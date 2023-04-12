from typing import List, Tuple, Union

from class_resolver import HintOrType, OptionalKwargs

from klinker.encoders.deepblocker import (
    DeepBlockerFrameEncoder,
    deep_blocker_encoder_resolver,
)

from .blockbuilder import EmbeddingBlockBuilder
from .blocker import EmbeddingBlocker


class DeepBlocker(EmbeddingBlocker):
    def __init__(
        self,
        wanted_cols: Union[
            str, List[str], Tuple[Union[str, List[str]], Union[str, List[str]]]
        ] = None,
        frame_encoder: HintOrType[DeepBlockerFrameEncoder] = None,
        frame_encoder_kwargs: OptionalKwargs = None,
        embedding_block_builder: HintOrType[EmbeddingBlockBuilder] = None,
        embedding_block_builder_kwargs: OptionalKwargs = None,
    ):
        frame_encoder = deep_blocker_encoder_resolver.make(
            frame_encoder, frame_encoder_kwargs
        )
        super().__init__(
            wanted_cols=wanted_cols,
            frame_encoder=frame_encoder,
            embedding_block_builder=embedding_block_builder,
            embedding_block_builder_kwargs=embedding_block_builder_kwargs,
        )
