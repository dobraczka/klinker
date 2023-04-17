import time
from typing import Dict, List, Tuple, Type

import click
from pykeen.trackers import ConsoleResultTracker, ResultTracker, WANDBResultTracker
from sylloge import MovieGraphBenchmark, OpenEA
from sylloge.base import EADataset

from klinker.blockers import DeepBlocker, MinHashLSHBlocker, TokenBlocker
from klinker.blockers.base import Blocker
from klinker.blockers.embedding.blockbuilder import block_builder_resolver, EmbeddingBlockBuilder
from klinker import KlinkerTripleFrame, KlinkerDataset
from klinker.encoders.deepblocker import deep_blocker_encoder_resolver, DeepBlockerFrameEncoder
from klinker.encoders.base import FrameEncoder
from klinker.encoders.pretrained import tokenized_frame_encoder_resolver, TokenizedFrameEncoder
from klinker.eval_metrics import Evaluation


@click.group(chain=True)
@click.option("--clean/--no-clean", default=False)
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

    tracker: ResultTracker
    if wandb:
        tracker = WANDBResultTracker(
            project="klinker", entity="dobraczka", config=params
        )
    else:
        tracker = ConsoleResultTracker()
    tracker.start_run()
    start = time.time()
    blocks = blocker.assign(left=klinker_dataset.left, right=klinker_dataset.right)
    end = time.time()
    ev = Evaluation.from_dataset(blocks=blocks, dataset=klinker_dataset)
    run_time = end - start
    tracker.log_metrics({**ev.to_dict(), "time in s": run_time})


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
def movie_graph_benchmark_dataset(graph_pair: str) -> EADataset:
    return MovieGraphBenchmark(graph_pair=graph_pair)


@cli.command()
@click.option("--threshold", type=float, default=0.5)
@click.option("--num-perm", type=int, default=128)
def lsh_blocker(threshold: float, num_perm: int) -> Tuple[Blocker, Dict]:
    return (
        MinHashLSHBlocker(threshold=threshold, num_perm=num_perm),
        click.get_current_context().params,
    )


@cli.command()
@deep_blocker_encoder_resolver.get_option("--encoder", default="autoencoder")
@tokenized_frame_encoder_resolver.get_option("--inner-encoder", default="TransformerTokenizedFrameEncoder")
@click.option("--num-epochs", type=int, default=50)
@click.option("--batch-size", type=int, default=256)
@click.option("--learning_rate", type=float, default=1e-3)
@click.option("--synth-tuples-per-tuple", type=int, default=5)
@click.option("--pos-to-neg-ratio", type=float, default=1.0)
@click.option("--max-perturbation", type=float, default=0.4)
@block_builder_resolver.get_option("--block-builder", default="kiez")
@click.option("--n-neighbors", type=int, default=100)
def deepblocker(
    encoder: Type[DeepBlockerFrameEncoder],
    inner_encoder: Type[TokenizedFrameEncoder],
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    synth_tuples_per_tuple: int,
    pos_to_neg_ratio: float,
    max_perturbation: float,
    block_builder: Type[EmbeddingBlockBuilder],
    n_neighbors: int,
) -> Tuple[Blocker, Dict]:
    encoder_kwargs = {
        "frame_encoder": inner_encoder,
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
    return (
        DeepBlocker(
            frame_encoder=encoder,
            frame_encoder_kwargs=encoder_kwargs,
            embedding_block_builder=block_builder,
            embedding_block_builder_kwargs=dict(n_neighbors=n_neighbors),
        ),
        click.get_current_context().params,
    )

@cli.command()
def token_blocker():
    return TokenBlocker(), {}


if __name__ == "__main__":
    cli()
