from typing import List, Tuple, Union

import pandas as pd
from class_resolver import HintOrType, OptionalKwargs

from klinker.blockers.base import SchemaAgnosticBlocker
from klinker.blockers.embedding import EmbeddingBlockBuilder, block_builder_resolver, FrameEncoder, frame_encoder_resolver
from klinker.data import KlinkerFrame


class EmbeddingBlocker(SchemaAgnosticBlocker):
    def __init__(
        self,
        wanted_cols: Union[
            str, List[str], Tuple[Union[str, List[str]], Union[str, List[str]]]
        ] = None,
        frame_encoder: HintOrType[FrameEncoder] = None,
        frame_encoder_kwargs: OptionalKwargs = None,
        embedding_block_builder: HintOrType[EmbeddingBlockBuilder] = None,
        embedding_block_builder_kwargs: OptionalKwargs = None,
    ):
        super().__init__(wanted_cols=wanted_cols)
        self.frame_encoder = frame_encoder_resolver.make(
            frame_encoder, frame_encoder_kwargs
        )
        self.embedding_block_builder = block_builder_resolver.make(
            embedding_block_builder, embedding_block_builder_kwargs
        )

    def _assign(self, left: KlinkerFrame, right: KlinkerFrame) -> pd.DataFrame:
        left_reduced = left[self._actual_wanted_cols[0]]
        right_reduced = right[self._actual_wanted_cols[1]]

        left_emb, right_emb = self.frame_encoder.encode(
            left=left_reduced, right=right_reduced
        )
        return self.embedding_block_builder.build_blocks(
            left=left_emb, right=right_emb, left_data=left, right_data=right
        )
