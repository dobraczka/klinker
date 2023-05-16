import ast
import hashlib
import json
import os
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple, Type

import click
import pandas as pd
from pykeen.trackers import ConsoleResultTracker, ResultTracker, WANDBResultTracker
from sylloge import OAEI, MovieGraphBenchmark, OpenEA
from sylloge.base import EADataset

from klinker import KlinkerDataset
from klinker.blockers import (
    DeepBlocker,
    EmbeddingBlocker,
    MinHashLSHBlocker,
    RelationalMinHashLSHBlocker,
    RelationalTokenBlocker,
    TokenBlocker,
)
from klinker.blockers.base import Blocker
from klinker.blockers.embedding.blockbuilder import (
    EmbeddingBlockBuilder,
    block_builder_resolver,
)
from klinker.encoders import (
    AverageEmbeddingTokenizedFrameEncoder,
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
from klinker.eval_metrics import Evaluation


def _create_artifact_path(
    artifact_name: str, artifact_dir: str, suffix="_blocks.pkl"
) -> str:
    return os.path.join(os.path.join(artifact_dir, f"{artifact_name}{suffix}"))


def _handle_artifacts(
    blocks: pd.DataFrame, tracker: ResultTracker, params: Dict, artifact_dir: str
) -> None:
    if isinstance(tracker, WANDBResultTracker):
        artifact_name = str(tracker.run.id)
        artifact_file_path = _create_artifact_path(artifact_name, artifact_dir)
        blocks.to_pickle(artifact_file_path)
        # artifact = wandb.Artifact(name="blocks", type="result")
        # artifact.add_file(local_path=artifact_file_path)
        # tracker.run.log_artifact(artifact)
    else:
        # see https://stackoverflow.com/a/22003440
        hashed_config_params = hashlib.sha1(
            json.dumps(params, sort_keys=True).encode()
        ).hexdigest()
        params_artifact_path = _create_artifact_path(
            hashed_config_params, artifact_dir, suffix="_params.pkl"
        )
        with open(params_artifact_path, "wb") as out_file:
            pickle.dump(params, out_file)
        counter = 0
        while True:
            artifact_name = f"{hashed_config_params}_{counter}"
            artifact_file_path = _create_artifact_path(artifact_name, artifact_dir)
            if os.path.exists(artifact_file_path):
                counter += 1
            else:
                break
        blocks.to_pickle(artifact_file_path)


@click.group(chain=True)
@click.option("--clean/--no-clean", default=True)
@click.option("--wandb/--no-wandb", is_flag=True, default=False)
def cli(clean: bool, wandb: bool):
    pass


@cli.result_callback()
def process_pipeline(blocker_and_dataset: List, clean: bool, wandb: bool):
    assert (
        len(blocker_and_dataset) == 2
    ), "Only 1 dataset and 1 blocker command can be used!"
    if not isinstance(blocker_and_dataset[0][0], EADataset):
        raise ValueError("First command must be dataset command!")
    if not isinstance(blocker_and_dataset[1][0], Blocker):
        raise ValueError("First command must be blocker command!")
    dataset_with_params, blocker_with_params = blocker_and_dataset
    dataset, ds_params = dataset_with_params
    blocker, bl_params = blocker_with_params
    klinker_dataset = KlinkerDataset.from_sylloge(dataset, clean=clean)
    params = {**ds_params, **bl_params}

    dataset_name = dataset.canonical_name
    blocker_name = blocker.__class__.__name__
    experiment_artifact_dir = os.path.join(
        "experiment_artifacts", dataset_name, blocker_name
    )
    if not os.path.exists(experiment_artifact_dir):
        os.makedirs(experiment_artifact_dir)
    tracker: ResultTracker
    if wandb:
        tracker = WANDBResultTracker(
            project="klinker", entity="dobraczka", config=params
        )
    else:
        tracker = ConsoleResultTracker()
    tracker.start_run()

    start = time.time()
    blocks = blocker.assign(
        left=klinker_dataset.left,
        right=klinker_dataset.right,
        left_rel=klinker_dataset.left_rel,
        right_rel=klinker_dataset.right_rel,
    )
    end = time.time()
    ev = Evaluation.from_dataset(blocks=blocks, dataset=klinker_dataset)
    run_time = end - start
    tracker.log_metrics({**ev.to_dict(), "time in s": run_time})

    _handle_artifacts(blocks, tracker, params, experiment_artifact_dir)
    tracker.end_run()


@cli.command()
@click.option("--graph-pair", type=str, default="D_W")
@click.option("--size", type=str, default="15K")
@click.option("--version", type=str, default="V1")
def open_ea_dataset(graph_pair: str, size: str, version: str) -> Tuple[EADataset, Dict]:
    return (
        OpenEA(graph_pair=graph_pair, size=size, version=version),
        click.get_current_context().params,
    )


@cli.command()
@click.option("--graph-pair", type=str, default="imdb-tmdb")
def movie_graph_benchmark_dataset(graph_pair: str) -> Tuple[EADataset, Dict]:
    return (
        MovieGraphBenchmark(graph_pair=graph_pair),
        click.get_current_context().params,
    )


@cli.command()
@click.option("--task", type=str, default="starwars-swg")
def oaei_dataset(task: str) -> Tuple[EADataset, Dict]:
    return (
        OAEI(task=task, backend="pandas"),
        click.get_current_context().params,
    )


@cli.command()
@click.option("--threshold", type=float, default=0.5)
@click.option("--num-perm", type=int, default=128)
@click.option("--fn-weight", type=float, default=0.5)
def lsh_blocker(
    threshold: float, num_perm: int, fn_weight: float
) -> Tuple[Blocker, Dict]:
    fp_weight = 1.0 - fn_weight
    return (
        MinHashLSHBlocker(
            threshold=threshold, num_perm=num_perm, weights=(fp_weight, fn_weight)
        ),
        click.get_current_context().params,
    )


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
) -> Tuple[Blocker, Dict]:
    attr_fp_weight = 1.0 - attr_fn_weight
    rel_fp_weight = 1.0 - rel_fn_weight
    return (
        RelationalMinHashLSHBlocker(
            attr_threshold=attr_threshold,
            attr_num_perm=attr_num_perm,
            attr_weights=(attr_fp_weight, attr_fn_weight),
            rel_threshold=rel_threshold,
            rel_num_perm=rel_num_perm,
            rel_weights=(rel_fp_weight, rel_fn_weight),
        ),
        click.get_current_context().params,
    )


@cli.command()
@deep_blocker_encoder_resolver.get_option("--encoder", default="autoencoder")
@tokenized_frame_encoder_resolver.get_option(
    "--inner-encoder", default="TransformerTokenizedFrameEncoder"
)
@click.option("--embeddings", type=str, default="glove")
@click.option("--num-epochs", type=int, default=50)
@click.option("--batch-size", type=int, default=256)
@click.option("--learning_rate", type=float, default=1e-3)
@click.option("--synth-tuples-per-tuple", type=int, default=5)
@click.option("--pos-to-neg-ratio", type=float, default=1.0)
@click.option("--max-perturbation", type=float, default=0.4)
@block_builder_resolver.get_option("--block-builder", default="kiez")
@click.option("--block-builder-kwargs", type=str)
@click.option("--n-neighbors", type=int, default=100)
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
    block_builder: Type[EmbeddingBlockBuilder],
    block_builder_kwargs: str,
    n_neighbors: int,
) -> Tuple[Blocker, Dict]:
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
    bb_kwargs["n_neighbors"] = n_neighbors
    return (
        DeepBlocker(
            frame_encoder=encoder,
            frame_encoder_kwargs=encoder_kwargs,
            embedding_block_builder=block_builder,
            embedding_block_builder_kwargs=bb_kwargs,
        ),
        click.get_current_context().params,
    )


@cli.command()
@click.option("--min-token-length", type=int, default=3)
def token_blocker(min_token_length: int):
    return (
        TokenBlocker(min_token_length=min_token_length),
        click.get_current_context().params,
    )


@cli.command()
@click.option("--attr-min-token-length", type=int, default=3)
@click.option("--rel-min-token-length", type=int, default=3)
def relational_token_blocker(attr_min_token_length: int, rel_min_token_length: int):
    return (
        RelationalTokenBlocker(
            attr_min_token_length=attr_min_token_length,
            rel_min_token_length=rel_min_token_length,
        ),
        click.get_current_context().params,
    )


@cli.command()
@tokenized_frame_encoder_resolver.get_option(
    "--inner-encoder", default="TransformerTokenizedFrameEncoder"
)
@click.option("--embeddings", type=str, default="glove")
@click.option("--ent-dim", type=int, default=256)
@click.option("--depth", type=int, default=2)
@click.option("--mini-dim", type=int, default=16)
@click.option("--rel-dim", type=int)
@click.option("--batch-size", type=int)
@block_builder_resolver.get_option("--block-builder", default="kiez")
@click.option("--block-builder-kwargs", type=str)
@click.option("--n-neighbors", type=int, default=100)
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
) -> Tuple[Blocker, Dict]:
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
    return (
        EmbeddingBlocker(
            frame_encoder=LightEAFrameEncoder(
                ent_dim=ent_dim,
                depth=depth,
                mini_dim=mini_dim,
                attribute_encoder=inner_encoder,
                attribute_encoder_kwargs=attribute_encoder_kwargs,
            ),
            embedding_block_builder=block_builder,
            embedding_block_builder_kwargs=bb_kwargs,
        ),
        click.get_current_context().params,
    )


@cli.command()
@tokenized_frame_encoder_resolver.get_option(
    "--inner-encoder", default="TransformerTokenizedFrameEncoder"
)
@click.option("--embeddings", type=str, default="glove")
@click.option("--depth", type=int, default=2)
@click.option("--batch-size", type=int)
@block_builder_resolver.get_option("--block-builder", default="kiez")
@click.option("--block-builder-kwargs", type=str)
@click.option("--n-neighbors", type=int, default=100)
def gcn_blocker(
    inner_encoder: Type[TokenizedFrameEncoder],
    embeddings: str,
    depth: int,
    batch_size: Optional[int],
    block_builder: Type[EmbeddingBlockBuilder],
    block_builder_kwargs: str,
    n_neighbors: int,
) -> Tuple[Blocker, Dict]:
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
    return (
        EmbeddingBlocker(
            frame_encoder=GCNFrameEncoder(
                depth=depth,
                attribute_encoder=inner_encoder,
                attribute_encoder_kwargs=attribute_encoder_kwargs,
            ),
            embedding_block_builder=block_builder,
            embedding_block_builder_kwargs=bb_kwargs,
        ),
        click.get_current_context().params,
    )


if __name__ == "__main__":
    cli()
