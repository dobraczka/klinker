import ast
import functools
import hashlib
import json
import logging
import os
import pickle
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, get_args

import click
import numpy as np

import torch
from klinker import KlinkerBlockManager, KlinkerDataset
from klinker.blockers import (
    DeepBlocker,
    EmbeddingBlocker,
    MinHashLSHBlocker,
    RelationalDeepBlocker,
    RelationalMinHashLSHBlocker,
    SimpleRelationalTokenBlocker,
    TokenBlocker,
)
from klinker.blockers.base import Blocker
from klinker.blockers.embedding.blockbuilder import (
    EmbeddingBlockBuilder,
    KiezEmbeddingBlockBuilder,
    block_builder_resolver,
)
from klinker.encoders import (
    FrameEncoder,
    GCNDeepBlockerFrameEncoder,
    GCNFrameEncoder,
    LightEADeepBlockerFrameEncoder,
    LightEAFrameEncoder,
    frame_encoder_resolver,
)
from klinker.encoders.deepblocker import (
    DeepBlockerFrameEncoder,
    deep_blocker_encoder_resolver,
)
from klinker.encoders.pretrained import (
    TokenizedFrameEncoder,
)
from klinker.eval import Evaluation
from klinker.trackers import ConsoleResultTracker, ResultTracker, WANDBResultTracker
from nephelai import upload
from sylloge import OAEI, MovieGraphBenchmark, OpenEA
from sylloge.base import MultiSourceEADataset
from sylloge.moviegraph_benchmark_loader import GraphPair as movie_graph_pairs
from sylloge.oaei_loader import OAEI_TASK_NAME
from sylloge.open_ea_loader import GRAPH_PAIRS as open_ea_graph_pairs
from sylloge.open_ea_loader import GRAPH_SIZES as open_ea_graph_sizes
from sylloge.open_ea_loader import GRAPH_VERSIONS as open_ea_graph_versions

logger = logging.getLogger("KlinkerExperiment")

KIEZ_FAISS_DEFAULT_KEY = "faissflat"
KIEZ_FAISS_DEFAULT = {
    "algorithm": "Faiss",
    "algorithm_kwargs": {
        "index_key": "Flat",
        "use_gpu": True,
    },
}

KIEZ_FAISS_HNSW_KEY = "faisshnsw"
KIEZ_FAISS_HNSW = {
    "algorithm": "Faiss",
    "algorithm_kwargs": {
        "index_key": "HNSW",
        "use_gpu": False,
    },
}


def parse_bb_kwargs(
    block_builder_kwargs: str,
    n_neighbors: int,
    block_builder: str,
    n_candidates: Optional[int] = None,
) -> Dict[str, Any]:
    bb_kwargs: Dict[str, Any] = {}
    if block_builder_kwargs:
        if block_builder == "kiez":
            if block_builder_kwargs == KIEZ_FAISS_DEFAULT_KEY:
                bb_kwargs = KIEZ_FAISS_DEFAULT
            elif block_builder_kwargs == KIEZ_FAISS_HNSW_KEY:
                bb_kwargs = KIEZ_FAISS_HNSW
        elif block_builder_kwargs != KIEZ_FAISS_DEFAULT_KEY:
            bb_kwargs = ast.literal_eval(block_builder_kwargs)
    bb_kwargs["n_neighbors"] = n_neighbors
    if n_candidates:
        bb_kwargs["n_candidates"] = n_candidates
    return bb_kwargs


def embedding_options(f):
    @click.option(
        "--inner-encoder",
        type=click.Choice(
            [
                "sifembeddingtokenized",
                "averageembeddingtokenized",
                "transformertokenized",
                "sentencetransformertokenized",
            ]
        ),
    )
    @click.option("--embeddings", type=str, default="fasttext")
    @click.option("--inner-encoder-batch-size", type=int, default=256)
    @click.option("--reduce-transformer-dim-to", type=int, default=-1)
    @click.option("--reduce-sample-perc", type=float, default=0.3)
    @block_builder_resolver.get_option(
        "--block-builder", default="kiez", as_string=True
    )
    @click.option("--block-builder-kwargs", type=str, default=KIEZ_FAISS_DEFAULT_KEY)
    @click.option("--n-neighbors", type=int, default=100)
    @click.option("--n-candidates", type=int, default=None)
    @click.option("--force", type=bool, default=True)
    @click.option("--save-emb", type=bool, default=False)
    @functools.wraps(f)
    def wrapper_common_options(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper_common_options


def create_inner_encoder(
    inner_encoder: Type[TokenizedFrameEncoder],
    embeddings: str,
    inner_encoder_batch_size: int,
    reduce_dim_to: int,
    reduce_sample_perc: float,
) -> FrameEncoder:
    attribute_encoder_kwargs: Dict = {}
    if inner_encoder in (
        "transformertokenized",
        "sentencetransformertokenized",
    ):
        attribute_encoder_kwargs = {"batch_size": inner_encoder_batch_size}
        if embeddings != "fasttext":
            attribute_encoder_kwargs["model_name"] = embeddings
        if inner_encoder == "sentencetransformertokenized":
            if reduce_dim_to > 0:
                attribute_encoder_kwargs["reduce_dim_to"] = reduce_dim_to
                attribute_encoder_kwargs["reduce_sample_perc"] = reduce_sample_perc
    elif inner_encoder in (
        "averageembeddingtokenized",
        "sifembeddingtokenized",
    ):
        attribute_encoder_kwargs = {
            "tokenized_word_embedder_kwargs": {"embedding_fn": embeddings}
        }
    return frame_encoder_resolver.make(inner_encoder, attribute_encoder_kwargs)


def set_random_seed(seed: Optional[int] = None):
    if seed is None:
        seed = np.random.randint(0, 2**16)
        logger.info(f"No random seed provided. Using {seed}")
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)
    random.seed(seed)
    return seed


@dataclass
class ExperimentInfo:
    params: Dict
    tracker: ResultTracker
    experiment_artifact_dir: str
    artifact_name: str
    blocks_path: str
    encodings_dir: Optional[str]
    params_artifact_path: Optional[str]


def _get_encoder_times(instance, known: Dict[str, float]) -> Dict[str, float]:
    if isinstance(instance, RelationalDeepBlocker):
        known_attr = {
            f"attr_{key}": val
            for key, val in _get_encoder_times(
                instance._attribute_blocker, known
            ).items()
        }
        known_rel = {
            f"rel_{key}": val
            for key, val in _get_encoder_times(
                instance._relation_blocker, known
            ).items()
        }
        return {**known_attr, **known_rel}
    for value in instance.__dict__.values():
        if isinstance(value, FrameEncoder) and hasattr(value, "_encoding_time"):
            known[value.__class__.__name__] = value._encoding_time
            known.update(_get_encoder_times(value, known))
    return known


def _create_artifact_path(artifact_name: str, artifact_dir: str, suffix: str) -> str:
    return os.path.join(artifact_dir, f"{artifact_name}{suffix}")


def _create_artifact_name(tracker: ResultTracker, params: Dict) -> str:
    if isinstance(tracker, WANDBResultTracker):
        return str(tracker.run.id)
    else:
        # see https://stackoverflow.com/a/22003440
        return hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()


def _handle_artifacts(
    wandb: bool,
    params: Dict,
    artifact_name: str,
    artifact_dir: str,
    nextcloud: bool,
) -> None:
    if not wandb:
        params_artifact_path = _create_artifact_path(
            artifact_name, artifact_dir, suffix="_params.pkl"
        )
        with open(params_artifact_path, "wb") as out_file:
            pickle.dump(params, out_file)
        logger.info(f"Saved parameters artifact in {params_artifact_path}")
    if nextcloud:
        upload(artifact_dir, artifact_dir)


def _handle_encodings_dir(blocker, artifact_name, experiment_artifact_dir):
    if isinstance(blocker, (EmbeddingBlocker, RelationalDeepBlocker)):
        if blocker.force:
            encodings_dir = _create_artifact_path(
                artifact_name, experiment_artifact_dir, suffix="_encoded"
            )
        else:
            encodings_dir = _create_artifact_path(
                "ignoring_params", experiment_artifact_dir, suffix="_encoded"
            )
            if not os.path.exists(encodings_dir):
                os.makedirs(encodings_dir)
                run_info_path = _create_artifact_path(
                    f"created_by_{artifact_name}",
                    encodings_dir,
                    suffix="_encoded",
                )
                Path(run_info_path).touch()
        blocker.save = True
        blocker.save_dir = encodings_dir
        return encodings_dir
    else:
        return None


def prepare(
    blocker: Blocker,
    dataset: MultiSourceEADataset,
    params: Dict,
    wandb: bool,
    seed: int,
) -> ExperimentInfo:
    # clean names
    blocker_name = blocker.__class__.__name__
    dataset_name = dataset.canonical_name
    params["dataset_name"] = dataset.canonical_name
    if isinstance(blocker, EmbeddingBlocker):
        blocker_name = blocker.frame_encoder.__class__.__name__.replace(
            "FrameEncoder", ""
        )
    if isinstance(blocker, RelationalDeepBlocker):
        blocker_name = (
            "Relational"
            + blocker._attribute_blocker.frame_encoder.__class__.__name__.replace(
                "FrameEncoder", ""
            )
        )

    if "kiez" in params.values():
        # TODO remove
        params["improved_time"] = True
    params["blocker_name"] = blocker_name
    params["random_seed"] = seed

    # create tracker
    tracker: ResultTracker
    params_artifact_path = None
    if wandb:
        tracker = WANDBResultTracker(
            project="klinker", entity="dobraczka", config=params
        )
    else:
        tracker = ConsoleResultTracker()

    # create paths
    experiment_artifact_dir = os.path.join(
        "experiment_artifacts", dataset_name, blocker_name
    )
    if not os.path.exists(experiment_artifact_dir):
        os.makedirs(experiment_artifact_dir)
    tracker.start_run()
    artifact_name = _create_artifact_name(tracker, params)
    encodings_dir = _handle_encodings_dir(
        blocker, artifact_name, experiment_artifact_dir
    )

    params_artifact_path = (
        _create_artifact_path(
            artifact_name, experiment_artifact_dir, suffix="_params.pkl"
        )
        if not wandb
        else None
    )
    blocks_path = _create_artifact_path(
        artifact_name, experiment_artifact_dir, suffix="_blocks.parquet"
    )
    return ExperimentInfo(
        params=params,
        tracker=tracker,
        experiment_artifact_dir=experiment_artifact_dir,
        artifact_name=artifact_name,
        blocks_path=blocks_path,
        params_artifact_path=params_artifact_path,
        encodings_dir=encodings_dir,
    )


def prepare_dask_slurm_cluster(
    num_cores: int,
    memory: str,
    walltime: str,
    num_clusters: int,
    local_directory: str = "$TMPDIR",
    project: str = "p_scads_knowledgegraphs",
):
    from dask.distributed import Client
    from dask_jobqueue import SLURMCluster
    import subprocess as sp
    from time import sleep

    # setting up the dashboard
    uid = int(sp.check_output("id -u", shell=True).decode("utf-8").replace("\n", ""))
    portdash = 10001 + uid

    # create a Slurm cluster, please specify your project
    cluster = SLURMCluster(
        cores=num_cores,
        project=project,
        memory=memory,
        walltime=walltime,
        local_directory=local_directory,
        scheduler_options={"dashboard_address": f":{portdash}"},
    )

    # submit the job to the scheduler with the number of nodes (here 2) requested:
    cluster.scale(num_clusters)

    # wait for Slurm to allocate a resources
    sleep(120)

    # check resources
    client = Client(cluster)
    client


@click.group(chain=True)
@click.option("--clean/--no-clean", default=True)
@click.option("--wandb/--no-wandb", is_flag=True, default=False)
@click.option("--nextcloud/--no-nextcloud", is_flag=True, default=False)
@click.option("--random-seed", type=int, default=None)
@click.option("--partition-size", type=str, default="100MB")
@click.option("--use-cluster", type=bool, default=False)
@click.option("--num-cores", type=int, default=1)
@click.option("--memory", type=str, default="8GB")
@click.option("--walltime", type=str, default="01:00:00")
@click.option("--num-clusters", type=int, default=2)
@click.option("--local-directory", type=str, default="$TMPDIR")
def cli(
    clean: bool,
    wandb: bool,
    nextcloud: bool,
    random_seed: Optional[int],
    partition_size: Optional[str],
    use_cluster: bool,
    num_cores: int,
    memory: str,
    walltime: str,
    num_clusters: int,
    local_directory: str,
):
    if use_cluster:
        prepare_dask_slurm_cluster(
            num_cores=num_cores,
            memory=memory,
            walltime=walltime,
            num_clusters=num_clusters,
        )
    pass


@cli.result_callback()
def process_pipeline(
    blocker_and_dataset: List,
    clean: bool,
    wandb: bool,
    nextcloud: bool,
    random_seed: Optional[int],
    partition_size: Optional[str],
    use_cluster: bool,
    num_cores: int,
    memory: str,
    walltime: str,
    num_clusters: int,
    local_directory: str,
):
    seed = set_random_seed(random_seed)
    assert (
        len(blocker_and_dataset) == 2
    ), "Only 1 dataset and 1 blocker command can be used!"
    if not isinstance(blocker_and_dataset[0][0], MultiSourceEADataset):
        raise ValueError("First command must be dataset command!")
    if not isinstance(blocker_and_dataset[1][0], Blocker):
        raise ValueError("Second command must be blocker command!")
    dataset_with_params, blocker_with_params = blocker_and_dataset
    dataset, ds_params = dataset_with_params
    blocker, bl_params, blocker_creation_time = blocker_with_params
    klinker_dataset = KlinkerDataset.from_sylloge(
        dataset, clean=clean, partition_size=partition_size
    )
    params = {**ds_params, **bl_params}

    experiment_info = prepare(
        blocker=blocker, dataset=dataset, params=params, wandb=wandb, seed=seed
    )
    tracker = experiment_info.tracker

    start = time.time()
    blocks = blocker.assign(
        left=klinker_dataset.left,
        right=klinker_dataset.right,
        left_rel=klinker_dataset.left_rel,
        right_rel=klinker_dataset.right_rel,
    )
    blocks.to_parquet(experiment_info.blocks_path, overwrite=True)

    end = time.time()
    run_time = end - start
    run_time += blocker_creation_time
    logger.info(f"Execution took: {run_time} seconds")
    logger.info(f"Wrote blocks to {experiment_info.blocks_path}")
    blocks = KlinkerBlockManager.read_parquet(experiment_info.blocks_path)
    ev = Evaluation.from_dataset(blocks=blocks, dataset=klinker_dataset)
    encoder_times: Dict[str, float] = {
        f"encoder_times_{key.lower()}": value
        for key, value in _get_encoder_times(blocker, {}).items()
    }
    results = {
        **ev.to_dict(),
        "time_in_s": run_time,
        "blocker_creation_time": blocker_creation_time,
        **encoder_times,
    }
    if isinstance(blocker, EmbeddingBlocker) and isinstance(
        blocker.embedding_block_builder, KiezEmbeddingBlockBuilder
    ):
        results["nn_search_time"] = blocker.embedding_block_builder._nn_search_time

    if hasattr(blocker, "_loading_time"):
        results["encoded_loading_time"] = blocker._loading_time
    tracker.log_metrics(results)

    _handle_artifacts(
        wandb=wandb,
        params=experiment_info.params,
        artifact_name=experiment_info.artifact_name,
        artifact_dir=experiment_info.experiment_artifact_dir,
        nextcloud=nextcloud,
    )
    # cleanup
    if os.path.exists("/tmp/left/"):
        shutil.rmtree("/tmp/left")
        shutil.rmtree("/tmp/right")
    tracker.end_run()


@cli.command()
@click.option("--graph-pair", type=click.Choice(open_ea_graph_pairs), default="D_W")
@click.option("--size", type=click.Choice(open_ea_graph_sizes), default="15K")
@click.option("--version", type=click.Choice(open_ea_graph_versions), default="V1")
@click.option("--backend", type=str, default="pandas")
def open_ea_dataset(
    graph_pair: str, size: str, version: str, backend: str
) -> Tuple[MultiSourceEADataset, Dict]:
    return (
        OpenEA(
            graph_pair=graph_pair,
            size=size,
            version=version,
            backend=backend,
        ),
        click.get_current_context().params,
    )


@cli.command()
@click.option(
    "--graph-pair", type=click.Choice(get_args(movie_graph_pairs)), default="imdb-tmdb"
)
def movie_graph_benchmark_dataset(
    graph_pair: str,
) -> Tuple[MultiSourceEADataset, Dict]:
    return (
        MovieGraphBenchmark(graph_pair=graph_pair),
        click.get_current_context().params,
    )


@cli.command()
@click.option(
    "--task", type=click.Choice(get_args(OAEI_TASK_NAME)), default="starwars-swg"
)
def oaei_dataset(task: str) -> Tuple[MultiSourceEADataset, Dict]:
    return (
        OAEI(task=task),
        click.get_current_context().params,
    )


@cli.command()
@click.option("--threshold", type=float, default=0.5)
@click.option("--num-perm", type=int, default=128)
@click.option("--fn-weight", type=float, default=0.5)
def lsh_blocker(
    threshold: float, num_perm: int, fn_weight: float
) -> Tuple[Blocker, Dict, float]:
    fp_weight = 1.0 - fn_weight
    start = time.time()
    blocker = MinHashLSHBlocker(
        threshold=threshold, num_perm=num_perm, weights=(fp_weight, fn_weight)
    )
    end = time.time()
    return (blocker, click.get_current_context().params, end - start)


@cli.command()
@click.option("--attr-threshold", type=float, default=0.5)
@click.option("--attr-num-perm", type=int, default=128)
@click.option("--attr-fn-weight", type=float, default=0.5)
@click.option("--rel-threshold", type=float, default=0.7)
@click.option("--rel-num-perm", type=int, default=128)
@click.option("--rel-fn-weight", type=float, default=0.5)
@click.option("--top-n-a", type=int, default=None)
@click.option("--top-n-r", type=int, default=None)
def relational_lsh_blocker(
    attr_threshold: float,
    attr_num_perm: int,
    attr_fn_weight: float,
    rel_threshold: float,
    rel_num_perm: int,
    rel_fn_weight: float,
    top_n_a: Optional[int],
    top_n_r: Optional[int],
) -> Tuple[Blocker, Dict, float]:
    attr_fp_weight = 1.0 - attr_fn_weight
    rel_fp_weight = 1.0 - rel_fn_weight
    if top_n_a < 0:
        top_n_a = None
    if top_n_r < 0:
        top_n_r = None
    start = time.time()
    blocker = RelationalMinHashLSHBlocker(
        attr_threshold=attr_threshold,
        attr_num_perm=attr_num_perm,
        attr_weights=(attr_fp_weight, attr_fn_weight),
        rel_threshold=rel_threshold,
        rel_num_perm=rel_num_perm,
        rel_weights=(rel_fp_weight, rel_fn_weight),
        top_n_a=top_n_a,
        top_n_r=top_n_r,
    )
    end = time.time()
    return (blocker, click.get_current_context().params, end - start)


@cli.command()
@deep_blocker_encoder_resolver.get_option(
    "--encoder", default="autoencoder", as_string=True
)
@click.option("--batch-size", type=int, default=256)
@click.option("--num-epochs", type=int, default=50)
@click.option("--learning-rate", type=float, default=1e-3)
@click.option("--synth-tuples-per-tuple", type=int, default=5)
@click.option("--pos-to-neg-ratio", type=float, default=1.0)
@click.option("--max-perturbation", type=float, default=0.4)
@click.option("--embedding-dimension", type=int, default=300)
@click.option("--hidden-dimension", type=int, default=150)
@block_builder_resolver.get_option("--block-builder", default="kiez", as_string=True)
@embedding_options
def deepblocker(
    encoder: Type[DeepBlockerFrameEncoder],
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    synth_tuples_per_tuple: int,
    pos_to_neg_ratio: float,
    max_perturbation: float,
    embedding_dimension: int,
    hidden_dimension: int,
    inner_encoder: Type[TokenizedFrameEncoder],
    inner_encoder_batch_size: int,
    reduce_transformer_dim_to: int,
    reduce_sample_perc: float,
    embeddings: str,
    block_builder: Type[EmbeddingBlockBuilder],
    block_builder_kwargs: str,
    n_neighbors: int,
    n_candidates: Optional[int],
    force: bool,
    save_emb: bool,
) -> Tuple[Blocker, Dict, float]:
    inner_encoder_inst = create_inner_encoder(
        inner_encoder,
        embeddings,
        inner_encoder_batch_size,
        reduce_transformer_dim_to,
        reduce_sample_perc,
    )
    encoder_kwargs = {
        "frame_encoder": inner_encoder_inst,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "hidden_dimensions": (embedding_dimension, hidden_dimension),
    }
    if encoder != "autoencoder":
        encoder_kwargs.update(
            {
                "synth_tuples_per_tuple": synth_tuples_per_tuple,
                "pos_to_neg_ratio": pos_to_neg_ratio,
                "max_perturbation": max_perturbation,
            }
        )
    bb_kwargs = parse_bb_kwargs(
        block_builder_kwargs, n_neighbors, block_builder, n_candidates=n_candidates
    )
    start = time.time()
    blocker = DeepBlocker(
        frame_encoder=encoder,
        frame_encoder_kwargs=encoder_kwargs,
        embedding_block_builder=block_builder,
        embedding_block_builder_kwargs=bb_kwargs,
        force=force,
        save=save_emb,
    )
    end = time.time()
    return (blocker, click.get_current_context().params, end - start)


@cli.command()
@deep_blocker_encoder_resolver.get_option(
    "--encoder", default="autoencoder", as_string=True
)
@click.option("--num-epochs", type=int, default=50)
@click.option("--batch-size", type=int, default=256)
@click.option("--learning-rate", type=float, default=1e-3)
@click.option("--synth-tuples-per-tuple", type=int, default=5)
@click.option("--pos-to-neg-ratio", type=float, default=1.0)
@click.option("--max-perturbation", type=float, default=0.4)
@click.option("--embedding-dimension", type=int, default=300)
@click.option("--hidden-dimension", type=int, default=150)
@click.option("--rel-n-neighbors", type=int, default=100)
@click.option("--top-n-a", type=int, default=None)
@click.option("--top-n-r", type=int, default=None)
@embedding_options
def relational_deepblocker(
    encoder: Type[DeepBlockerFrameEncoder],
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    synth_tuples_per_tuple: int,
    pos_to_neg_ratio: float,
    max_perturbation: float,
    embedding_dimension: int,
    hidden_dimension: int,
    rel_n_neighbors: int,
    top_n_a: Optional[int],
    top_n_r: Optional[int],
    inner_encoder: Type[TokenizedFrameEncoder],
    embeddings: str,
    inner_encoder_batch_size: int,
    reduce_transformer_dim_to: int,
    reduce_sample_perc: float,
    block_builder: Type[EmbeddingBlockBuilder],
    block_builder_kwargs: str,
    n_neighbors: int,
    n_candidates: Optional[int],
    force: bool,
    save_emb: bool,
) -> Tuple[Blocker, Dict, float]:
    if top_n_a < 0:
        top_n_a = None
    if top_n_r < 0:
        top_n_r = None
    inner_encoder_inst = create_inner_encoder(
        inner_encoder,
        embeddings,
        inner_encoder_batch_size,
        reduce_transformer_dim_to,
        reduce_sample_perc,
    )
    encoder_kwargs = {
        "frame_encoder": inner_encoder_inst,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "hidden_dimensions": (embedding_dimension, hidden_dimension),
    }
    if encoder != "autoencoder":
        encoder_kwargs.update(
            {
                "synth_tuples_per_tuple": synth_tuples_per_tuple,
                "pos_to_neg_ratio": pos_to_neg_ratio,
                "max_perturbation": max_perturbation,
            }
        )
    bb_kwargs = parse_bb_kwargs(block_builder_kwargs, 100, block_builder, n_candidates)
    # deep-copy
    attr_bb_kwargs = {**bb_kwargs}
    rel_bb_kwargs = {**bb_kwargs}
    attr_bb_kwargs["n_neighbors"] = n_neighbors
    rel_bb_kwargs["n_neighbors"] = rel_n_neighbors
    start = time.time()
    blocker = RelationalDeepBlocker(
        attr_frame_encoder=encoder,
        attr_frame_encoder_kwargs=encoder_kwargs,
        attr_embedding_block_builder=block_builder,
        attr_embedding_block_builder_kwargs=attr_bb_kwargs,
        rel_frame_encoder=encoder,
        rel_frame_encoder_kwargs=encoder_kwargs,
        rel_embedding_block_builder=block_builder,
        rel_embedding_block_builder_kwargs=rel_bb_kwargs,
        force=force,
        save=save_emb,
        top_n_a=top_n_a,
        top_n_r=top_n_r,
    )
    end = time.time()
    return (blocker, click.get_current_context().params, end - start)


@cli.command()
@click.option("--min-token-length", type=int, default=3)
def token_blocker(min_token_length: int) -> Tuple[Blocker, Dict, float]:
    start = time.time()
    blocker = TokenBlocker(min_token_length=min_token_length)
    end = time.time()
    return (blocker, click.get_current_context().params, end - start)


@cli.command()
@click.option("--min-token-length", type=int, default=3)
@click.option("--top-n-a", type=int, default=None)
@click.option("--top-n-r", type=int, default=None)
def relational_token_blocker(
    min_token_length: int,
    top_n_a: Optional[int],
    top_n_r: Optional[int],
) -> Tuple[Blocker, Dict, float]:
    if top_n_a and top_n_a < 0:
        top_n_a = None
    if top_n_r and top_n_r < 0:
        top_n_r = None
    start = time.time()
    blocker = SimpleRelationalTokenBlocker(
        min_token_length=min_token_length, top_n_a=top_n_a, top_n_r=top_n_r
    )
    end = time.time()
    return (blocker, click.get_current_context().params, end - start)


@cli.command()
@click.option("--depth", type=int, default=2)
@click.option("--mini-dim", type=int, default=16)
@click.option("--rel-dim", type=int)
@click.option("--batch-size", type=int)
@block_builder_resolver.get_option("--block-builder", default="kiez", as_string=True)
@embedding_options
def light_ea_blocker(
    depth: int,
    mini_dim: int,
    rel_dim: Optional[int],
    batch_size: Optional[int],
    inner_encoder: Type[TokenizedFrameEncoder],
    embeddings: str,
    inner_encoder_batch_size: int,
    reduce_transformer_dim_to: int,
    reduce_sample_perc: float,
    block_builder: Type[EmbeddingBlockBuilder],
    block_builder_kwargs: str,
    n_neighbors: int,
    n_candidates: Optional[int],
    force: bool,
    save_emb: bool,
) -> Tuple[Blocker, Dict, float]:
    inner_encoder_inst = create_inner_encoder(
        inner_encoder,
        embeddings,
        inner_encoder_batch_size,
        reduce_transformer_dim_to,
        reduce_sample_perc,
    )
    bb_kwargs = parse_bb_kwargs(block_builder_kwargs, 100, block_builder, n_candidates)
    print(bb_kwargs)
    start = time.time()
    blocker = EmbeddingBlocker(
        frame_encoder=LightEAFrameEncoder(
            depth=depth,
            mini_dim=mini_dim,
            attribute_encoder=inner_encoder_inst,
        ),
        embedding_block_builder=block_builder,
        embedding_block_builder_kwargs=bb_kwargs,
        force=force,
        save=save_emb,
    )
    end = time.time()
    return (blocker, click.get_current_context().params, end - start)


@cli.command()
@click.option("--batch-size", type=int)
@click.option("--depth", type=int, default=2)
@click.option("--edge-weight", type=float, default=1.0)
@click.option("--self-loop-weight", type=float, default=2.0)
@click.option("--layer-dims", type=int, default=300)
@click.option("--bias", type=bool, default=True)
@click.option("--use-weight-layers", type=bool, default=True)
@click.option("--aggr", type=str, default="sum")
@embedding_options
def gcn_blocker(
    batch_size: Optional[int],
    depth: int,
    edge_weight: float,
    self_loop_weight: float,
    layer_dims: int,
    bias: bool,
    use_weight_layers: bool,
    aggr: str,
    inner_encoder: Type[TokenizedFrameEncoder],
    embeddings: str,
    inner_encoder_batch_size: int,
    reduce_transformer_dim_to: int,
    reduce_sample_perc: float,
    block_builder: Type[EmbeddingBlockBuilder],
    block_builder_kwargs: str,
    n_neighbors: int,
    n_candidates: Optional[int],
    force: bool,
    save_emb: bool,
) -> Tuple[Blocker, Dict, float]:
    inner_encoder_inst = create_inner_encoder(
        inner_encoder,
        embeddings,
        inner_encoder_batch_size,
        reduce_transformer_dim_to,
        reduce_sample_perc,
    )
    bb_kwargs = parse_bb_kwargs(
        block_builder_kwargs, n_neighbors, block_builder, n_candidates
    )
    start = time.time()
    blocker = EmbeddingBlocker(
        frame_encoder=GCNFrameEncoder(
            depth=depth,
            edge_weight=edge_weight,
            self_loop_weight=self_loop_weight,
            layer_dims=layer_dims,
            bias=bias,
            use_weight_layers=use_weight_layers,
            aggr=aggr,
            attribute_encoder=inner_encoder_inst,
        ),
        embedding_block_builder=block_builder,
        embedding_block_builder_kwargs=bb_kwargs,
        force=force,
        save=save_emb,
    )
    end = time.time()
    return (blocker, click.get_current_context().params, end - start)


@cli.command()
@click.option("--te-batch-size", type=int)
@click.option("--depth", type=int, default=2)
@click.option("--edge-weight", type=float, default=1.0)
@click.option("--self-loop-weight", type=float, default=2.0)
@click.option("--bias", type=bool, default=True)
@click.option("--use-weight-layers", type=bool, default=True)
@click.option("--aggr", type=str, default="sum")
@deep_blocker_encoder_resolver.get_option(
    "--deepblocker-encoder", default="autoencoder", as_string=True
)
@click.option("--num-epochs", type=int, default=50)
@click.option("--batch-size", type=int, default=256)
@click.option("--learning-rate", type=float, default=1e-3)
@click.option("--synth-tuples-per-tuple", type=int, default=5)
@click.option("--pos-to-neg-ratio", type=float, default=1.0)
@click.option("--max-perturbation", type=float, default=0.4)
@click.option("--embedding-dimension", type=int, default=300)
@click.option("--hidden-dimension", type=int, default=75)
@embedding_options
def gcn_deepblocker(
    te_batch_size: Optional[int],
    depth: int,
    edge_weight: float,
    self_loop_weight: float,
    bias: bool,
    use_weight_layers: bool,
    aggr: str,
    deepblocker_encoder: str,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    synth_tuples_per_tuple: int,
    pos_to_neg_ratio: float,
    max_perturbation: float,
    embedding_dimension: int,
    hidden_dimension: int,
    inner_encoder: Type[TokenizedFrameEncoder],
    embeddings: str,
    inner_encoder_batch_size: int,
    reduce_transformer_dim_to: int,
    reduce_sample_perc: float,
    block_builder: Type[EmbeddingBlockBuilder],
    block_builder_kwargs: str,
    n_neighbors: int,
    n_candidates: Optional[int],
    force: bool,
    save_emb: bool,
) -> Tuple[Blocker, Dict, float]:
    inner_encoder_inst = create_inner_encoder(
        inner_encoder,
        embeddings,
        inner_encoder_batch_size,
        reduce_transformer_dim_to,
        reduce_sample_perc,
    )

    deepblocker_encoder_kwargs = {
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
    }
    if deepblocker_encoder != "autoencoder":
        deepblocker_encoder_kwargs.update(
            {
                "synth_tuples_per_tuple": synth_tuples_per_tuple,
                "pos_to_neg_ratio": pos_to_neg_ratio,
                "max_perturbation": max_perturbation,
            }
        )
    bb_kwargs = parse_bb_kwargs(
        block_builder_kwargs, n_neighbors, block_builder, n_candidates
    )

    start = time.time()
    blocker = EmbeddingBlocker(
        frame_encoder=GCNDeepBlockerFrameEncoder(
            depth=depth,
            edge_weight=edge_weight,
            self_loop_weight=self_loop_weight,
            embedding_dimension=embedding_dimension,
            hidden_dimension=hidden_dimension,
            bias=bias,
            use_weight_layers=use_weight_layers,
            aggr=aggr,
            inner_encoder=inner_encoder_inst,
            deepblocker_encoder=deepblocker_encoder,
            deepblocker_encoder_kwargs=deepblocker_encoder_kwargs,
        ),
        embedding_block_builder=block_builder,
        embedding_block_builder_kwargs=bb_kwargs,
        force=force,
        save=save_emb,
    )
    end = time.time()
    return (blocker, click.get_current_context().params, end - start)


@cli.command()
@click.option("--depth", type=int, default=2)
@click.option("--mini-dim", type=int, default=16)
@deep_blocker_encoder_resolver.get_option(
    "--deepblocker-encoder", default="autoencoder", as_string=True
)
@click.option("--num-epochs", type=int, default=50)
@click.option("--batch-size", type=int, default=256)
@click.option("--learning-rate", type=float, default=1e-3)
@click.option("--synth-tuples-per-tuple", type=int, default=5)
@click.option("--pos-to-neg-ratio", type=float, default=1.0)
@click.option("--max-perturbation", type=float, default=0.4)
@click.option("--embedding-dimension", type=int, default=300)
@click.option("--hidden-dimension", type=int, default=75)
@embedding_options
def light_ea_deepblocker(
    depth: int,
    mini_dim: int,
    deepblocker_encoder: str,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    synth_tuples_per_tuple: int,
    pos_to_neg_ratio: float,
    max_perturbation: float,
    embedding_dimension: int,
    hidden_dimension: int,
    inner_encoder: Type[TokenizedFrameEncoder],
    embeddings: str,
    inner_encoder_batch_size: int,
    reduce_transformer_dim_to: int,
    reduce_sample_perc: float,
    block_builder: Type[EmbeddingBlockBuilder],
    block_builder_kwargs: str,
    n_neighbors: int,
    n_candidates: Optional[int],
    force: bool,
    save_emb: bool,
) -> Tuple[Blocker, Dict, float]:
    inner_encoder_inst = create_inner_encoder(
        inner_encoder,
        embeddings,
        inner_encoder_batch_size,
        reduce_transformer_dim_to,
        reduce_sample_perc,
    )

    deepblocker_encoder_kwargs = {
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
    }
    if deepblocker_encoder != "autoencoder":
        deepblocker_encoder_kwargs.update(
            {
                "synth_tuples_per_tuple": synth_tuples_per_tuple,
                "pos_to_neg_ratio": pos_to_neg_ratio,
                "max_perturbation": max_perturbation,
            }
        )
    bb_kwargs = parse_bb_kwargs(
        block_builder_kwargs, n_neighbors, block_builder, n_candidates
    )

    start = time.time()
    blocker = EmbeddingBlocker(
        frame_encoder=LightEADeepBlockerFrameEncoder(
            depth=depth,
            mini_dim=mini_dim,
            embedding_dimension=embedding_dimension,
            hidden_dimension=hidden_dimension,
            inner_encoder=inner_encoder_inst,
            deepblocker_encoder=deepblocker_encoder,
            deepblocker_encoder_kwargs=deepblocker_encoder_kwargs,
        ),
        embedding_block_builder=block_builder,
        embedding_block_builder_kwargs=bb_kwargs,
        force=force,
        save=save_emb,
    )
    end = time.time()
    return (blocker, click.get_current_context().params, end - start)


@cli.command()
@embedding_options
def only_embeddings_blocker(
    inner_encoder: str,
    embeddings: str,
    inner_encoder_batch_size: int,
    reduce_transformer_dim_to: int,
    reduce_sample_perc: float,
    block_builder: Type[EmbeddingBlockBuilder],
    block_builder_kwargs: str,
    n_neighbors: int,
    n_candidates: Optional[int],
    force: bool,
    save_emb: bool,
) -> Tuple[Blocker, Dict, float]:
    print("save_emb=%s" % (save_emb))
    inner_encoder_inst = create_inner_encoder(
        inner_encoder,
        embeddings,
        inner_encoder_batch_size,
        reduce_transformer_dim_to,
        reduce_sample_perc,
    )
    bb_kwargs = parse_bb_kwargs(
        block_builder_kwargs, n_neighbors, block_builder, n_candidates
    )
    start = time.time()
    blocker = EmbeddingBlocker(
        frame_encoder=inner_encoder_inst,
        embedding_block_builder=block_builder,
        embedding_block_builder_kwargs=bb_kwargs,
        force=force,
        save=save_emb,
    )
    end = time.time()
    return (blocker, click.get_current_context().params, end - start)


if __name__ == "__main__":
    # import dask
    # dask.config.set(scheduler='processes')
    # from dask.distributed import Client, LocalCluster
    # cluster = LocalCluster()  # Launches a scheduler and workers locally
    # client = Client(cluster)  # Connect to distributed cluster and override default
    cli()
