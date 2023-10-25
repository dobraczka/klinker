import ast
import hashlib
import json
import logging
import os
import pickle
import random
import shutil
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type, get_args

import click
import numpy as np
import torch
from nephelai import upload
from sylloge import OAEI, MovieGraphBenchmark, OpenEA
from sylloge.base import EADataset
from sylloge.moviegraph_benchmark_loader import GraphPair as movie_graph_pairs
from sylloge.oaei_loader import OAEI_TASK_NAME
from sylloge.open_ea_loader import GRAPH_PAIRS as open_ea_graph_pairs
from sylloge.open_ea_loader import GRAPH_SIZES as open_ea_graph_sizes
from sylloge.open_ea_loader import GRAPH_VERSIONS as open_ea_graph_versions

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
    block_builder_resolver,
)
from klinker.encoders import (
    AverageEmbeddingTokenizedFrameEncoder,
    FrameEncoder,
    GCNFrameEncoder,
    LightEAFrameEncoder,
    SIFEmbeddingTokenizedFrameEncoder,
    TransformerTokenizedFrameEncoder,
)
from klinker.encoders.deepblocker import (
    DeepBlockerFrameEncoder,
    deep_blocker_encoder_resolver,
)
from klinker.encoders.pretrained import (
    TokenizedFrameEncoder,
    tokenized_frame_encoder_resolver,
)
from klinker.eval import Evaluation
from klinker.trackers import ConsoleResultTracker, ResultTracker, WANDBResultTracker

logger = logging.getLogger("KlinkerExperiment")


def set_random_seed(seed: Optional[int] = None):
    if seed is None:
        seed = np.random.randint(0, 2**16)
        logger.info(f"No random seed provided. Using {seed}")
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)
    random.seed(seed)


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
    for _, value in instance.__dict__.items():
        if isinstance(value, FrameEncoder):
            if hasattr(value, "_encoding_time"):
                known[value.__class__.__name__] = value._encoding_time
                known.update(_get_encoder_times(value, known))
    return known


def _create_artifact_path(artifact_name: str, artifact_dir: str, suffix: str) -> str:
    return os.path.join(os.path.join(artifact_dir, f"{artifact_name}{suffix}"))


def _create_artifact_name(tracker: ResultTracker, params: Dict) -> str:
    if isinstance(tracker, WANDBResultTracker):
        return str(tracker.run.id)
    else:
        # see https://stackoverflow.com/a/22003440
        return hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()


# def _create_block_artifact_path(artifact_name: str, artifact_dir: str, tracker: ResultTracker, params: Dict) -> str:
#     if isinstance(tracker, WANDBResultTracker):
#         artifact_file_path = _create_artifact_path(artifact_name, artifact_dir)
#     else:
#         params_artifact_path = _create_artifact_path(
#             artifact_name, artifact_dir, suffix="_params.pkl"
#         )
#         with open(params_artifact_path, "wb") as out_file:
#             pickle.dump(params, out_file)
#         logger.info(f"Saved parameters artifact in {params_artifact_path}")
#         counter = 0
#         while True:
#             counter_artifact_name = f"{artifact_name}_{counter}"
#             artifact_file_path = _create_artifact_path(
#                 counter_artifact_name, artifact_dir
#             )
#             if os.path.exists(artifact_file_path):
#                 counter += 1
#             else:
#                 break
#     return artifact_file_path


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


def prepare(
    blocker: Blocker, dataset: EADataset, params: Dict, wandb: bool
) -> ExperimentInfo:

    # clean names
    blocker_name = blocker.__class__.__name__
    dataset_name = dataset.canonical_name
    params["dataset_name"] = dataset.canonical_name
    if isinstance(blocker, EmbeddingBlocker):
        blocker_name = blocker.frame_encoder.__class__.__name__.replace(
            "FrameEncoder", ""
        )
    params["blocker_name"] = blocker_name

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
    encodings_dir = None
    if isinstance(blocker, EmbeddingBlocker):
        encodings_dir = _create_artifact_path(
            artifact_name, experiment_artifact_dir, suffix="_encoded"
        )
        blocker.save = True
        blocker.save_dir = encodings_dir

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


@click.group(chain=True)
@click.option("--clean/--no-clean", default=True)
@click.option("--wandb/--no-wandb", is_flag=True, default=False)
@click.option("--nextcloud/--no-nextcloud", is_flag=True, default=False)
@click.option("--random-seed", type=int, default=None)
def cli(clean: bool, wandb: bool, nextcloud: bool, random_seed: Optional[int]):
    pass


@cli.result_callback()
def process_pipeline(
    blocker_and_dataset: List,
    clean: bool,
    wandb: bool,
    nextcloud: bool,
    random_seed: Optional[int],
):
    set_random_seed(random_seed)
    assert (
        len(blocker_and_dataset) == 2
    ), "Only 1 dataset and 1 blocker command can be used!"
    if not isinstance(blocker_and_dataset[0][0], EADataset):
        raise ValueError("First command must be dataset command!")
    if not isinstance(blocker_and_dataset[1][0], Blocker):
        raise ValueError("Second command must be blocker command!")
    dataset_with_params, blocker_with_params = blocker_and_dataset
    dataset, ds_params = dataset_with_params
    blocker, bl_params, blocker_creation_time = blocker_with_params
    klinker_dataset = KlinkerDataset.from_sylloge(dataset, clean=clean)
    params = {**ds_params, **bl_params}

    experiment_info = prepare(
        blocker=blocker, dataset=dataset, params=params, wandb=wandb
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
@click.option("--npartitions", type=int, default=1)
def open_ea_dataset(
    graph_pair: str, size: str, version: str, backend: str, npartitions: int
) -> Tuple[EADataset, Dict]:
    return (
        OpenEA(
            graph_pair=graph_pair,
            size=size,
            version=version,
            backend=backend,
            npartitions=npartitions,
        ),
        click.get_current_context().params,
    )


@cli.command()
@click.option(
    "--graph-pair", type=click.Choice(get_args(movie_graph_pairs)), default="imdb-tmdb"
)
@click.option("--backend", type=str, default="pandas")
@click.option("--npartitions", type=int, default=1)
def movie_graph_benchmark_dataset(
    graph_pair: str, backend: str, npartitions: int
) -> Tuple[EADataset, Dict]:
    return (
        MovieGraphBenchmark(
            graph_pair=graph_pair, backend=backend, npartitions=npartitions
        ),
        click.get_current_context().params,
    )


@cli.command()
@click.option(
    "--task", type=click.Choice(get_args(OAEI_TASK_NAME)), default="starwars-swg"
)
@click.option("--backend", type=str, default="pandas")
@click.option("--npartitions", type=int, default=1)
def oaei_dataset(task: str, backend: str, npartitions: int) -> Tuple[EADataset, Dict]:
    return (
        OAEI(task=task, backend=backend, npartitions=npartitions),
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
def relational_lsh_blocker(
    attr_threshold: float,
    attr_num_perm: int,
    attr_fn_weight: float,
    rel_threshold: float,
    rel_num_perm: int,
    rel_fn_weight: float,
) -> Tuple[Blocker, Dict, float]:
    attr_fp_weight = 1.0 - attr_fn_weight
    rel_fp_weight = 1.0 - rel_fn_weight
    start = time.time()
    blocker = RelationalMinHashLSHBlocker(
        attr_threshold=attr_threshold,
        attr_num_perm=attr_num_perm,
        attr_weights=(attr_fp_weight, attr_fn_weight),
        rel_threshold=rel_threshold,
        rel_num_perm=rel_num_perm,
        rel_weights=(rel_fp_weight, rel_fn_weight),
    )
    end = time.time()
    return (blocker, click.get_current_context().params, end - start)


@cli.command()
@deep_blocker_encoder_resolver.get_option(
    "--encoder", default="autoencoder", as_string=True
)
@tokenized_frame_encoder_resolver.get_option(
    "--inner-encoder", default="SIFEmbeddingTokenizedFrameEncoder", as_string=True
)
@click.option("--embeddings", type=str, default="glove")
@click.option("--num-epochs", type=int, default=50)
@click.option("--batch-size", type=int, default=256)
@click.option("--learning_rate", type=float, default=1e-3)
@click.option("--synth-tuples-per-tuple", type=int, default=5)
@click.option("--pos-to-neg-ratio", type=float, default=1.0)
@click.option("--max-perturbation", type=float, default=0.4)
@click.option("--embedding-dimension", type=int, default=300)
@click.option("--hidden-dimension", type=int, default=150)
@block_builder_resolver.get_option("--block-builder", default="kiez", as_string=True)
@click.option("--block-builder-kwargs", type=str)
@click.option("--n-neighbors", type=int, default=100)
@click.option("--force", type=bool, default=True)
def deepblocker(
    encoder: Type[DeepBlockerFrameEncoder],
    inner_encoder: Type[TokenizedFrameEncoder],
    embeddings: str,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    synth_tuples_per_tuple: int,
    pos_to_neg_ratio: float,
    max_perturbation: float,
    embedding_dimension: int,
    hidden_dimension: int,
    block_builder: Type[EmbeddingBlockBuilder],
    block_builder_kwargs: str,
    n_neighbors: int,
    force: bool,
) -> Tuple[Blocker, Dict, float]:
    attribute_encoder_kwargs: Dict = {}
    if inner_encoder == TransformerTokenizedFrameEncoder:
        attribute_encoder_kwargs = dict(batch_size=batch_size)
    elif (
        inner_encoder == AverageEmbeddingTokenizedFrameEncoder
        or inner_encoder == SIFEmbeddingTokenizedFrameEncoder
    ):
        attribute_encoder_kwargs = dict(
            tokenized_word_embedder_kwargs=dict(embedding_fn=embeddings)
        )
    encoder_kwargs = {
        "frame_encoder": inner_encoder,
        "frame_encoder_kwargs": attribute_encoder_kwargs,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "hidden_dimensions": (embedding_dimension, hidden_dimension),
    }
    if not encoder == "autoencoder":
        encoder_kwargs.update(
            {
                "synth_tuples_per_tuple": synth_tuples_per_tuple,
                "pos_to_neg_ratio": pos_to_neg_ratio,
                "max_perturbation": max_perturbation,
            }
        )
    bb_kwargs: Dict[str, Any] = {}
    if block_builder_kwargs:
        bb_kwargs = ast.literal_eval(block_builder_kwargs)
    bb_kwargs["n_neighbors"] = n_neighbors
    start = time.time()
    blocker = DeepBlocker(
        frame_encoder=encoder,
        frame_encoder_kwargs=encoder_kwargs,
        embedding_block_builder=block_builder,
        embedding_block_builder_kwargs=bb_kwargs,
        force=force,
    )
    end = time.time()
    return (blocker, click.get_current_context().params, end - start)


@cli.command()
@deep_blocker_encoder_resolver.get_option(
    "--encoder", default="autoencoder", as_string=True
)
@tokenized_frame_encoder_resolver.get_option(
    "--inner-encoder", default="TransformerTokenizedFrameEncoder", as_string=True
)
@click.option("--embeddings", type=str, default="glove")
@click.option("--num-epochs", type=int, default=50)
@click.option("--batch-size", type=int, default=256)
@click.option("--learning_rate", type=float, default=1e-3)
@click.option("--synth-tuples-per-tuple", type=int, default=5)
@click.option("--pos-to-neg-ratio", type=float, default=1.0)
@click.option("--max-perturbation", type=float, default=0.4)
@block_builder_resolver.get_option("--block-builder", default="kiez", as_string=True)
@click.option("--block-builder-kwargs", type=str)
@click.option("--attr-n-neighbors", type=int, default=100)
@click.option("--rel-n-neighbors", type=int, default=100)
@click.option("--force", type=bool, default=True)
def relational_deepblocker(
    encoder: Type[DeepBlockerFrameEncoder],
    inner_encoder: Type[TokenizedFrameEncoder],
    embeddings: str,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    synth_tuples_per_tuple: int,
    pos_to_neg_ratio: float,
    max_perturbation: float,
    block_builder: Type[EmbeddingBlockBuilder],
    block_builder_kwargs: str,
    attr_n_neighbors: int,
    rel_n_neighbors: int,
    force: bool,
) -> Tuple[Blocker, Dict, float]:
    attribute_encoder_kwargs: Dict = {}
    if inner_encoder == TransformerTokenizedFrameEncoder:
        attribute_encoder_kwargs = dict(batch_size=batch_size)
    elif (
        inner_encoder == AverageEmbeddingTokenizedFrameEncoder
        or inner_encoder == SIFEmbeddingTokenizedFrameEncoder
    ):
        attribute_encoder_kwargs = dict(
            tokenized_word_embedder_kwargs=dict(embedding_fn=embeddings)
        )
    encoder_kwargs = {
        "frame_encoder": inner_encoder,
        "frame_encoder_kwargs": attribute_encoder_kwargs,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
    }
    if not encoder == "autoencoder":
        encoder_kwargs.update(
            {
                "synth_tuples_per_tuple": synth_tuples_per_tuple,
                "pos_to_neg_ratio": pos_to_neg_ratio,
                "max_perturbation": max_perturbation,
            }
        )
    bb_kwargs: Dict[str, Any] = {}
    if block_builder_kwargs:
        bb_kwargs = ast.literal_eval(block_builder_kwargs)
    attr_bb_kwargs = bb_kwargs
    rel_bb_kwargs = bb_kwargs
    attr_bb_kwargs["n_neighbors"] = attr_n_neighbors
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
    )
    end = time.time()
    return (blocker, click.get_current_context().params, end - start)


@cli.command()
@click.option("--min-token-length", type=int, default=3)
@click.option("--intermediate-saving", type=bool, default=False)
def token_blocker(
    min_token_length: int, intermediate_saving: bool
) -> Tuple[Blocker, Dict, float]:
    start = time.time()
    blocker = TokenBlocker(
        min_token_length=min_token_length, intermediate_saving=intermediate_saving
    )
    end = time.time()
    return (blocker, click.get_current_context().params, end - start)


@cli.command()
@click.option("--min-token-length", type=int, default=3)
@click.option("--intermediate-saving", type=bool, default=False)
def relational_token_blocker(
    min_token_length: int, intermediate_saving: bool
) -> Tuple[Blocker, Dict, float]:
    start = time.time()
    blocker = SimpleRelationalTokenBlocker(
        min_token_length=min_token_length, intermediate_saving=intermediate_saving
    )
    end = time.time()
    return (blocker, click.get_current_context().params, end - start)


@cli.command()
@tokenized_frame_encoder_resolver.get_option(
    "--inner-encoder", default="TransformerTokenizedFrameEncoder", as_string=True
)
@click.option("--embeddings", type=str, default="glove")
@click.option("--ent-dim", type=int, default=256)
@click.option("--depth", type=int, default=2)
@click.option("--mini-dim", type=int, default=16)
@click.option("--rel-dim", type=int)
@click.option("--batch-size", type=int)
@block_builder_resolver.get_option("--block-builder", default="kiez", as_string=True)
@click.option("--block-builder-kwargs", type=str)
@click.option("--n-neighbors", type=int, default=100)
@click.option("--force", type=bool, default=True)
def light_ea_blocker(
    inner_encoder: Type[TokenizedFrameEncoder],
    embeddings: str,
    ent_dim: int,
    depth: int,
    mini_dim: int,
    rel_dim: Optional[int],
    batch_size: Optional[int],
    block_builder: Type[EmbeddingBlockBuilder],
    block_builder_kwargs: str,
    n_neighbors: int,
    force: bool,
) -> Tuple[Blocker, Dict, float]:
    attribute_encoder_kwargs: Dict = {}
    if inner_encoder == TransformerTokenizedFrameEncoder:
        attribute_encoder_kwargs = dict(batch_size=batch_size)
    elif (
        inner_encoder == AverageEmbeddingTokenizedFrameEncoder
        or inner_encoder == SIFEmbeddingTokenizedFrameEncoder
    ):
        attribute_encoder_kwargs = dict(
            tokenized_word_embedder_kwargs=dict(embedding_fn=embeddings)
        )
    bb_kwargs: Dict[str, Any] = {}
    if block_builder_kwargs:
        bb_kwargs = ast.literal_eval(block_builder_kwargs)
    bb_kwargs["n_neighbors"] = n_neighbors
    start = time.time()
    blocker = EmbeddingBlocker(
        frame_encoder=LightEAFrameEncoder(
            ent_dim=ent_dim,
            depth=depth,
            mini_dim=mini_dim,
            attribute_encoder=inner_encoder,
            attribute_encoder_kwargs=attribute_encoder_kwargs,
        ),
        embedding_block_builder=block_builder,
        embedding_block_builder_kwargs=bb_kwargs,
        force=force,
    )
    end = time.time()
    return (blocker, click.get_current_context().params, end - start)


@cli.command()
@tokenized_frame_encoder_resolver.get_option(
    "--inner-encoder", default="SIFEmbeddingTokenizedFrameEncoder", as_string=True
)
@click.option("--batch-size", type=int)
@click.option("--embeddings", type=str, default="glove")
@click.option("--depth", type=int, default=2)
@click.option("--edge-weight", type=float, default=1.0)
@click.option("--self-loop-weight", type=float, default=2.0)
@click.option("--layer-dims", type=int, default=300)
@click.option("--bias", type=bool, default=True)
@click.option("--use-weight-layers", type=bool, default=True)
@click.option("--aggr", type=str, default="sum")
@block_builder_resolver.get_option("--block-builder", default="kiez", as_string=True)
@click.option("--block-builder-kwargs", type=str)
@click.option("--n-neighbors", type=int, default=100)
@click.option("--force", type=bool, default=True)
def gcn_blocker(
    inner_encoder: Type[TokenizedFrameEncoder],
    batch_size: Optional[int],
    embeddings: str,
    depth: int,
    edge_weight: float,
    self_loop_weight: float,
    layer_dims: int,
    bias: bool,
    use_weight_layers: bool,
    aggr: str,
    block_builder: Type[EmbeddingBlockBuilder],
    block_builder_kwargs: str,
    n_neighbors: int,
    force: bool,
) -> Tuple[Blocker, Dict, float]:
    attribute_encoder_kwargs: Dict = {}
    if inner_encoder == TransformerTokenizedFrameEncoder:
        attribute_encoder_kwargs = dict(batch_size=batch_size)
    elif (
        inner_encoder == AverageEmbeddingTokenizedFrameEncoder
        or inner_encoder == SIFEmbeddingTokenizedFrameEncoder
    ):
        attribute_encoder_kwargs = dict(
            tokenized_word_embedder_kwargs=dict(embedding_fn=embeddings)
        )
    bb_kwargs: Dict[str, Any] = {}
    if block_builder_kwargs:
        bb_kwargs = ast.literal_eval(block_builder_kwargs)
    bb_kwargs["n_neighbors"] = n_neighbors
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
            attribute_encoder=inner_encoder,
            attribute_encoder_kwargs=attribute_encoder_kwargs,
        ),
        embedding_block_builder=block_builder,
        embedding_block_builder_kwargs=bb_kwargs,
        force=force,
    )
    end = time.time()
    return (blocker, click.get_current_context().params, end - start)


@cli.command()
@click.option(
    "--encoder",
    type=click.Choice(["sifembeddingtokenized", "averageembeddingtokenized"]),
)
@click.option("--embeddings", type=str, default="fasttext")
@block_builder_resolver.get_option("--block-builder", default="kiez", as_string=True)
@click.option("--block-builder-kwargs", type=str)
@click.option("--n-neighbors", type=int, default=100)
@click.option("--force", type=bool, default=True)
def only_embeddings_blocker(
    encoder: str,
    embeddings: str,
    block_builder: Type[EmbeddingBlockBuilder],
    block_builder_kwargs: str,
    n_neighbors: int,
    force: bool,
) -> Tuple[Blocker, Dict, float]:
    frame_encoder_kwargs = dict(
        tokenized_word_embedder_kwargs=dict(embedding_fn=embeddings)
    )
    bb_kwargs: Dict[str, Any] = {}
    if block_builder_kwargs:
        bb_kwargs = ast.literal_eval(block_builder_kwargs)
    bb_kwargs["n_neighbors"] = n_neighbors
    start = time.time()
    blocker = EmbeddingBlocker(
        frame_encoder=encoder,
        frame_encoder_kwargs=frame_encoder_kwargs,
        embedding_block_builder=block_builder,
        embedding_block_builder_kwargs=bb_kwargs,
        force=force,
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
