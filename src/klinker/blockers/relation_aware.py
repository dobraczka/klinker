from typing import Callable, List, Optional, Tuple, Union

import pandas as pd
from class_resolver import HintOrType, OptionalKwargs
from nltk.tokenize import word_tokenize

from klinker.data import KlinkerFrame, KlinkerTriplePandasFrame, KlinkerBlockManager
from klinker.encoders.deepblocker import DeepBlockerFrameEncoder

from .base import Blocker
from .embedding.deepblocker import DeepBlocker
from .embedding.blockbuilder import EmbeddingBlockBuilder
from .lsh import MinHashLSHBlocker
from .token_blocking import TokenBlocker


def reverse_rel(rel_frame: pd.DataFrame) -> pd.DataFrame:
    orig_columns = rel_frame.columns
    rev_rel_frame = rel_frame[rel_frame.columns[::-1]]
    rev_rel_frame[rev_rel_frame.columns[1]] = (
        "_inv_" + rev_rel_frame[rev_rel_frame.columns[1]]
    )
    rev_rel_frame.columns = orig_columns
    return rev_rel_frame


def concat_neighbor_attributes(
    attribute_frame: KlinkerFrame, rel_frame: pd.DataFrame
) -> KlinkerFrame:
    """Return concatenated attributes of neighboring entities.

    Note:: If an entity does not show up in rel_frame it is not contained in the result! Also, the attributes of the entity itself are also not part of the concatenated attributes!

    :param attribute_frame: DataFrame with entity attributes
    :param rel_frame: DataFrame with relation triples
    :return: DataFrame with concatenated attribute values of neighboring entities
    """
    rev_rel_frame = reverse_rel(rel_frame)
    with_inv = pd.concat([rel_frame, rev_rel_frame])
    concat_attr = attribute_frame.concat_values().set_index(attribute_frame.id_col)
    return KlinkerTriplePandasFrame(
        with_inv.set_index(with_inv.columns[2]).join(concat_attr, how="left").dropna(),
        table_name=attribute_frame.table_name,
        id_col=rel_frame.columns[0],
    ).concat_values()


class RelationalBlocker(Blocker):
    _attribute_blocker: Blocker
    _relation_blocker: Blocker

    def _assign(
        self,
        left: KlinkerFrame,
        right: KlinkerFrame,
        left_rel: Optional[pd.DataFrame] = None,
        right_rel: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        attr_blocked = self._attribute_blocker.assign(left=left, right=right)
        left_rel_conc = concat_neighbor_attributes(left, left_rel)
        right_rel_conc = concat_neighbor_attributes(right, right_rel)
        rel_blocked = self._relation_blocker.assign(left_rel_conc, right_rel_conc)
        return KlinkerBlockManager.combine(attr_blocked, rel_blocked)


class RelationalMinHashLSHBlocker(RelationalBlocker):
    def __init__(
        self,
        tokenize_fn: Callable = word_tokenize,
        attr_threshold: float = 0.5,
        attr_num_perm: int = 128,
        attr_weights: Tuple[float, float] = (0.5, 0.5),
        rel_threshold: float = 0.7,
        rel_num_perm: int = 128,
        rel_weights: Tuple[float, float] = (0.5, 0.5),
        wanted_cols: Union[
            str, List[str], Tuple[Union[str, List[str]], Union[str, List[str]]]
        ] = None,
    ):
        self._attribute_blocker = MinHashLSHBlocker(
            tokenize_fn=tokenize_fn,
            threshold=attr_threshold,
            num_perm=attr_num_perm,
            wanted_cols=wanted_cols,
            weights=attr_weights,
        )
        self._relation_blocker = MinHashLSHBlocker(
            tokenize_fn=tokenize_fn,
            threshold=rel_threshold,
            num_perm=rel_num_perm,
            wanted_cols=wanted_cols,
            weights=rel_weights,
        )


class RelationalTokenBlocker(RelationalBlocker):
    def __init__(
        self,
        tokenize_fn: Callable[[str], List[str]] = word_tokenize,
        wanted_cols: Union[
            str, List[str], Tuple[Union[str, List[str]], Union[str, List[str]]]
        ] = None,
        attr_min_token_length: int = 3,
        rel_min_token_length: int = 3,
    ):
        self._attribute_blocker = TokenBlocker(
            tokenize_fn=tokenize_fn,
            wanted_cols=wanted_cols,
            min_token_length=attr_min_token_length,
        )
        self._relation_blocker = TokenBlocker(
            tokenize_fn=tokenize_fn,
            wanted_cols=wanted_cols,
            min_token_length=rel_min_token_length,
        )


class RelationalDeepBlocker(RelationalBlocker):
    def __init__(
        self,
        attr_frame_encoder: HintOrType[DeepBlockerFrameEncoder] = None,
        attr_frame_encoder_kwargs: OptionalKwargs = None,
        attr_embedding_block_builder: HintOrType[EmbeddingBlockBuilder] = None,
        attr_embedding_block_builder_kwargs: OptionalKwargs = None,
        rel_frame_encoder: HintOrType[DeepBlockerFrameEncoder] = None,
        rel_frame_encoder_kwargs: OptionalKwargs = None,
        rel_embedding_block_builder: HintOrType[EmbeddingBlockBuilder] = None,
        rel_embedding_block_builder_kwargs: OptionalKwargs = None,
    ):
        self._attribute_blocker = DeepBlocker(
            frame_encoder=attr_frame_encoder,
            frame_encoder_kwargs=attr_frame_encoder_kwargs,
            embedding_block_builder=attr_embedding_block_builder,
            embedding_block_builder_kwargs=attr_embedding_block_builder_kwargs,
        )
        self._relation_blocker = DeepBlocker(
            frame_encoder=rel_frame_encoder,
            frame_encoder_kwargs=rel_frame_encoder_kwargs,
            embedding_block_builder=rel_embedding_block_builder,
            embedding_block_builder_kwargs=rel_embedding_block_builder_kwargs,
        )
