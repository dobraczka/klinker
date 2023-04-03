from typing import Tuple, List, Union
import pandas as pd
from class_resolver import HintOrType, OptionalKwargs
from klinker.data import KlinkerFrame
from klinker.blockers.base import SchemaAgnosticBlocker
from klinker.blockers.embedding.deep_blocker_encoder import deep_blocker_encoder_resolver
from klinker.blockers.embedding import block_builder_resolver, EmbeddingBlockBuilder
from klinker.blockers.embedding.deep_blocker_encoder import DeepBlockerEncoder

class DeepBlocker(SchemaAgnosticBlocker):

    def __init__(
        self,
        wanted_cols: Union[
            str, List[str], Tuple[Union[str, List[str]], Union[str, List[str]]]
        ] = None,
        frame_encoder: HintOrType[DeepBlockerEncoder] = None,
        frame_encoder_kwargs: OptionalKwargs = None,
        embedding_block_builder: HintOrType[EmbeddingBlockBuilder] = None,
        embedding_block_builder_kwargs: OptionalKwargs = None,
    ):
        super().__init__(wanted_cols=wanted_cols)
        self.frame_encoder = deep_blocker_encoder_resolver.make(
            frame_encoder, frame_encoder_kwargs
        )
        self.embedding_block_builder = block_builder_resolver.make(
            embedding_block_builder, embedding_block_builder_kwargs
        )

    def _preprocess(self, df):
        df.fillna(' ', inplace=True)
        df = df.astype(str)
        return df.agg(' '.join, axis=1).to_frame(name="_merged_text")

    def _assign(self, left: KlinkerFrame, right: KlinkerFrame) -> pd.DataFrame:
        left = self._preprocess(left[self._actual_wanted_cols[0]])
        right = self._preprocess(right[self._actual_wanted_cols[1]])

        left_emb, right_emb = self.frame_encoder.encode(
            left=left, right=right
        )
        return self.embedding_block_builder.build_blocks(
            left=left_emb, right=right_emb, left_data=left, right_data=right
        )


