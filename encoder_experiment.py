import os
import time
from typing import Any, Dict, Tuple, Type, Union

import click
import numpy as np
from pykeen.trackers import WANDBResultTracker
from sylloge import OpenEA

from klinker.blockers.embedding import KiezEmbeddingBlockBuilder
from klinker.data import KlinkerDataset, NamedVector
from klinker.encoders import (
    FrameEncoder,
    GCNFrameEncoder,
    LightEAFrameEncoder,
    frame_encoder_resolver,
)
from klinker.encoders.deepblocker import DeepBlockerFrameEncoder
from klinker.eval_metrics import Evaluation


def _load_or_create_path(
    base_dir: str, encoder: Type[FrameEncoder]
) -> Union[Tuple[str, str], Tuple[NamedVector, NamedVector]]:
    encoder_name = str(encoder).split(".")[-1][:-2]
    encoder_dir = os.path.join(base_dir, encoder_name)
    if not os.path.exists(encoder_dir):
        os.makedirs(encoder_dir)
    left_enc_path = os.path.join(encoder_dir, "left.pkl")
    right_enc_path = os.path.join(encoder_dir, "right.pkl")
    if os.path.exists(left_enc_path) and os.path.exists(right_enc_path):
        return (
            NamedVector.from_pickle(left_enc_path),
            NamedVector.from_pickle(right_enc_path),
        )
    return left_enc_path, right_enc_path


@click.command()
@frame_encoder_resolver.get_option("--encoder")
@click.option("--n-neighbors", type=int, required=True)
@click.option("--base-dir", type=str, default="encoder_experiment_open_ea_d_y_15k_v1")
def run_experiment(
    encoder: Type[FrameEncoder],
    n_neighbors: int,
    base_dir: str,
):
    tracker = WANDBResultTracker(project="klinker", entity="dobraczka", config=click.get_current_context().params)
    tracker.start_run()
    path_or_encoded = _load_or_create_path(base_dir, encoder)
    ds = KlinkerDataset.from_sylloge(
        OpenEA(graph_pair="D_Y", size="15K", version="V1"), clean=True
    )

    encoder_run_time = 0.0
    if isinstance(path_or_encoded[0], str):
        print(f"Calculating with {encoder}")
        encoder_kwargs: Dict[str, Any] = {}
        if isinstance(
            encoder, (GCNFrameEncoder, LightEAFrameEncoder, DeepBlockerFrameEncoder)
        ):
            inner_encoder = "sifembedding"
            if isinstance(encoder, (GCNFrameEncoder, LightEAFrameEncoder)):
                encoder_kwargs["attribute_encoder"] = inner_encoder
            else:
                encoder_kwargs["inner_encoder"] = inner_encoder

        left = ds.left.concat_values()
        right = ds.right.concat_values()
        left_reduced = left.set_index(left.id_col)[left.non_id_columns]
        right_reduced = right.set_index(right.id_col)[right.non_id_columns]

        start = time.time()
        enc_inst = frame_encoder_resolver.make(encoder, encoder_kwargs)
        left_enc, right_enc = enc_inst.encode(
            left=left_reduced,
            right=right_reduced,
            left_rel=ds.left_rel,
            right_rel=ds.right_rel,
        )

        end = time.time()
        encoder_run_time = end - start
        left_enc.to_pickle(path_or_encoded[0])
        right_enc.to_pickle(path_or_encoded[1])
    else:
        print("loaded from pickle!")
        left_enc, right_enc = path_or_encoded

    start = time.time()
    kiez = KiezEmbeddingBlockBuilder(
        n_neighbors=n_neighbors,
        algorithm="Faiss",
        algorithm_kwargs={"index_key": "HNSW32", "index_param": "efSearch=918"},
    )
    blocks = kiez.build_blocks(
        left_enc.subset(ds.gold["left"][:100]),
        right_enc.subset(ds.gold["right"][:100]),
        left_name=ds.gold.columns[0],
        right_name=ds.gold.columns[1],
    )
    block_builder_run_time = time.time() - start
    ev = Evaluation.from_dataset(blocks=blocks, dataset=ds)
    tracker.log_metrics(
        {
            **ev.to_dict(),
            "encoder_run_time": encoder_run_time,
            "block_builder_run_time": block_builder_run_time,
        }
    )


if __name__ == "__main__":
    run_experiment()
