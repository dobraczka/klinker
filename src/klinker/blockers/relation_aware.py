import pathlib
from typing import Callable, List, Optional, Tuple, TypeVar, Union

import dask.dataframe as dd
import pandas as pd
from class_resolver import HintOrType, OptionalKwargs
from nltk.tokenize import word_tokenize

from klinker.typing import SeriesType

from ..data import (
    KlinkerBlockManager,
    KlinkerFrame,
    KlinkerPandasFrame,
    KlinkerTripleDaskFrame,
    KlinkerTriplePandasFrame,
    combine_blocks,
)
from ..encoders.deepblocker import DeepBlockerFrameEncoder
from ..typing import Frame
from ..utils import concat_frames
from .base import Blocker, SchemaAgnosticBlocker
from .embedding.blockbuilder import EmbeddingBlockBuilder
from .embedding.deepblocker import DeepBlocker
from .lsh import MinHashLSHBlocker
from .token_blocking import TokenBlocker

FrameType = TypeVar("FrameType", dd.DataFrame, pd.DataFrame)


def reverse_rel(rel_frame: Frame, inverse_prefix: str = "_inv_") -> Frame:
    """Reverse the relations by switching first and last column.

    Args:
    ----
      rel_frame: Frame: Frame with relation triples.
      inverse_prefix: Prefix for new inverse relations

    Returns:
    -------
      rel_frame with reversed relations
    """
    orig_columns = rel_frame.columns
    rev_rel_frame = rel_frame[rel_frame.columns[::-1]]
    rev_rel_frame[rev_rel_frame.columns[1]] = (
        inverse_prefix + rev_rel_frame[rev_rel_frame.columns[1]]
    )
    rev_rel_frame.columns = orig_columns
    return rev_rel_frame


def _upgrade_to_triple(concat_attr: FrameType, conc_frame: FrameType) -> FrameType:
    # make into triple frame
    concat_attr[conc_frame.columns[1]] = "dummy_relation"
    # reorder for common triple format (because relation was added as last col)
    concat_attr = concat_attr[
        [concat_attr.columns[0], concat_attr.columns[2], concat_attr.columns[1]]
    ]
    # common column names for concat
    concat_attr.columns = conc_frame.columns
    return concat_attr


def concat_neighbor_attributes(
    attribute_frame: KlinkerFrame, rel_frame: Frame, include_own_attributes: bool = True
) -> SeriesType:
    """Return concatenated attributes of neighboring entities.

    Args:
    ----
      attribute_frame: KlinkerFrame with entity attributes
      rel_frame: Frame with relation triples
      include_own_attributes: if True also concatenates attributes of entity itself
      attribute_frame: KlinkerFrame:
      rel_frame: Frame:
      include_own_attributes: bool:  (Default value = True)

    Returns:
    -------
      Series with concatenated attribute values of neighboring entities

    """
    assert attribute_frame.table_name
    rev_rel_frame = reverse_rel(rel_frame)
    with_inv = concat_frames([rel_frame, rev_rel_frame])
    concat_attr = attribute_frame.concat_values().to_frame().reset_index()
    if isinstance(concat_attr, dd.DataFrame):
        concat_attr._meta = pd.DataFrame(
            [], columns=[attribute_frame.id_col, attribute_frame.table_name], dtype=str
        )

    conc_frame = (
        with_inv.set_index(with_inv.columns[2])
        .join(concat_attr.set_index(attribute_frame.id_col), how="left")
        .dropna()
    )

    if isinstance(attribute_frame, KlinkerPandasFrame):
        if include_own_attributes:
            concat_attr = _upgrade_to_triple(concat_attr, conc_frame)
            conc_frame = pd.concat([conc_frame, concat_attr])
        return KlinkerTriplePandasFrame(
            conc_frame,
            table_name=attribute_frame.table_name,
            id_col=rel_frame.columns[0],
        ).concat_values()
    else:
        if include_own_attributes:
            concat_attr = _upgrade_to_triple(concat_attr, conc_frame)
            conc_frame = dd.concat([conc_frame, concat_attr])
        return KlinkerTripleDaskFrame.from_dask_dataframe(
            conc_frame,
            table_name=attribute_frame.table_name,
            id_col=rel_frame.columns[0],
            construction_class=KlinkerTriplePandasFrame,
        ).concat_values()


class BaseSimpleRelationalBlocker(Blocker):
    """Uses one blocking strategy on entity attribute values and concatenation of neighboring values."""

    _blocker: SchemaAgnosticBlocker

    def concat_relational_info(
        self,
        left: KlinkerFrame,
        right: KlinkerFrame,
        left_rel: KlinkerFrame,
        right_rel: KlinkerFrame,
    ) -> Tuple[SeriesType, SeriesType]:
        """Concatenate neighbor entity attribute values with own.

        Args:
        ----
          left: KlinkerFrame: Frame with attribute info of left dataset.
          right: KlinkerFrame: Frame with attribute info of right dataset.
          left_rel: KlinkerFrame: Relation triples of left dataset.
          right_rel: KlinkerFrame: Relation triples of right dataset.

        Returns:
        -------
            (left_conc, right_conc) Concatenated entity attribute values for left and right
        """
        left_conc = concat_neighbor_attributes(
            left, left_rel, include_own_attributes=True
        )
        right_conc = concat_neighbor_attributes(
            right, right_rel, include_own_attributes=True
        )
        return left_conc, right_conc

    def assign(
        self,
        left: KlinkerFrame,
        right: KlinkerFrame,
        left_rel: Optional[KlinkerFrame] = None,
        right_rel: Optional[KlinkerFrame] = None,
    ) -> KlinkerBlockManager:
        """Assign entity ids to blocks.

        Will concat all entity attribute information and neighboring info before proceeding.

        Args:
        ----
          left: KlinkerFrame: Contains entity attribute information of left dataset.
          right: KlinkerFrame: Contains entity attribute information of right dataset.
          left_rel: Optional[KlinkerFrame]:  (Default value = None) Contains relational information of left dataset.
          right_rel: Optional[KlinkerFrame]:  (Default value = None) Contains relational information of left dataset.

        Returns:
        -------
            KlinkerBlockManager: instance holding the resulting blocks.
        """
        assert left_rel is not None
        assert right_rel is not None
        left_conc, right_conc = self.concat_relational_info(
            left=left, right=right, left_rel=left_rel, right_rel=right_rel
        )
        return self._blocker._assign(left=left_conc, right=right_conc)


class SimpleRelationalTokenBlocker(BaseSimpleRelationalBlocker):
    """Token blocking on concatenation of entity attribute values and neighboring values.

    Examples
    --------
        >>> # doctest: +SKIP
        >>> from sylloge import MovieGraphBenchmark
        >>> from klinker.data import KlinkerDataset
        >>> ds = KlinkerDataset.from_sylloge(MovieGraphBenchmark(),clean=True)
        >>> from klinker.blockers import SimpleRelationalTokenBlocker
        >>> blocker = SimpleRelationalTokenBlocker()
        >>> blocks = blocker.assign(left=ds.left, right=ds.right, left_rel=ds.left_rel, right_rel=ds.right_rel)
    """

    def __init__(
        self,
        tokenize_fn: Callable[[str], List[str]] = word_tokenize,
        min_token_length: int = 3,
    ):
        self._blocker = TokenBlocker(
            tokenize_fn=tokenize_fn,
            min_token_length=min_token_length,
        )


class SimpleRelationalMinHashLSHBlocker(BaseSimpleRelationalBlocker):
    """MinHashLSH blocking on concatenation of entity attribute values and neighboring values.

    Examples
    --------
        >>> # doctest: +SKIP
        >>> from sylloge import MovieGraphBenchmark
        >>> from klinker.data import KlinkerDataset
        >>> ds = KlinkerDataset.from_sylloge(MovieGraphBenchmark(),clean=True)
        >>> from klinker.blockers import SimpleRelationalTokenBlocker
        >>> blocker = SimpleRelationalMinHashLSHBlocker()
        >>> blocks = blocker.assign(left=ds.left, right=ds.right, left_rel=ds.left_rel, right_rel=ds.right_rel)
    """

    def __init__(
        self,
        tokenize_fn: Callable = word_tokenize,
        threshold: float = 0.5,
        num_perm: int = 128,
        weights: Tuple[float, float] = (0.5, 0.5),
    ):
        self._blocker = MinHashLSHBlocker(
            tokenize_fn=tokenize_fn,
            threshold=threshold,
            num_perm=num_perm,
            weights=weights,
        )


class RelationalBlocker(Blocker):
    """Uses seperate blocker for entity attribute values and concatenation of neighboring entity attribute values."""

    _attribute_blocker: SchemaAgnosticBlocker
    _relation_blocker: SchemaAgnosticBlocker

    def assign(
        self,
        left: KlinkerFrame,
        right: KlinkerFrame,
        left_rel: Optional[KlinkerFrame] = None,
        right_rel: Optional[KlinkerFrame] = None,
    ) -> KlinkerBlockManager:
        """Assign entity ids to blocks.

        Will concat all entity attribute information before proceeding.
        Then uses `_attribute_blocker` for entity attribute values and
        `_relation_blocker` for concatenated neighboring entity attribute values.

        Args:
        ----
          left: KlinkerFrame: Contains entity attribute information of left dataset.
          right: KlinkerFrame: Contains entity attribute information of right dataset.
          left_rel: Optional[KlinkerFrame]:  (Default value = None) Contains relational information of left dataset.
          right_rel: Optional[KlinkerFrame]:  (Default value = None) Contains relational information of left dataset.

        Returns:
        -------
            KlinkerBlockManager: instance holding the resulting blocks.
        """
        attr_blocked = self._attribute_blocker.assign(left=left, right=right)
        left_rel_conc = concat_neighbor_attributes(
            left, left_rel, include_own_attributes=False
        )
        right_rel_conc = concat_neighbor_attributes(
            right, right_rel, include_own_attributes=False
        )
        rel_blocked = self._relation_blocker._assign(left_rel_conc, right_rel_conc)
        return combine_blocks(attr_blocked, rel_blocked)


class RelationalMinHashLSHBlocker(RelationalBlocker):
    """Seperate MinHashLSH blocking on concatenation of entity attribute values and neighboring values.

    Examples
    --------
        >>> # doctest: +SKIP
        >>> from sylloge import MovieGraphBenchmark
        >>> from klinker.data import KlinkerDataset
        >>> ds = KlinkerDataset.from_sylloge(MovieGraphBenchmark(),clean=True)
        >>> from klinker.blockers import RelationalMinHashLSHBlocker
        >>> blocker = RelationalMinHashLSHBlocker(attr_threshold=0.7, rel_threshold=0.9)
        >>> blocks = blocker.assign(left=ds.left, right=ds.right, left_rel=ds.left_rel, right_rel=ds.right_rel)
    """

    def __init__(
        self,
        tokenize_fn: Callable = word_tokenize,
        attr_threshold: float = 0.5,
        attr_num_perm: int = 128,
        attr_weights: Tuple[float, float] = (0.5, 0.5),
        rel_threshold: float = 0.7,
        rel_num_perm: int = 128,
        rel_weights: Tuple[float, float] = (0.5, 0.5),
    ):
        self._attribute_blocker = MinHashLSHBlocker(
            tokenize_fn=tokenize_fn,
            threshold=attr_threshold,
            num_perm=attr_num_perm,
            weights=attr_weights,
        )
        self._relation_blocker = MinHashLSHBlocker(
            tokenize_fn=tokenize_fn,
            threshold=rel_threshold,
            num_perm=rel_num_perm,
            weights=rel_weights,
        )


class RelationalTokenBlocker(RelationalBlocker):
    """Seperate Tokenblocking on concatenation of entity attribute values and neighboring values.

    Examples
    --------
        >>> # doctest: +SKIP
        >>> from sylloge import MovieGraphBenchmark
        >>> from klinker.data import KlinkerDataset
        >>> ds = KlinkerDataset.from_sylloge(MovieGraphBenchmark(),clean=True)
        >>> from klinker.blockers import RelationalTokenBlocker
        >>> blocker = RelationalTokenBlocker(attr_min_token_length=3, rel_min_token_length=5)
        >>> blocks = blocker.assign(left=ds.left, right=ds.right, left_rel=ds.left_rel, right_rel=ds.right_rel)

    """

    def __init__(
        self,
        tokenize_fn: Callable[[str], List[str]] = word_tokenize,
        attr_min_token_length: int = 3,
        rel_min_token_length: int = 3,
    ):
        self._attribute_blocker = TokenBlocker(
            tokenize_fn=tokenize_fn,
            min_token_length=attr_min_token_length,
        )
        self._relation_blocker = TokenBlocker(
            tokenize_fn=tokenize_fn,
            min_token_length=rel_min_token_length,
        )


class RelationalDeepBlocker(RelationalBlocker):
    """Seperate DeepBlocker strategy on concatenation of entity attribute values and neighboring values.

    Examples
    --------
        >>> # doctest: +SKIP
        >>> from sylloge import MovieGraphBenchmark
        >>> from klinker.data import KlinkerDataset
        >>> ds = KlinkerDataset.from_sylloge(MovieGraphBenchmark(),clean=True)
        >>> from klinker.blockers import RelationalDeepBlocker
        >>> blocker = RelationalDeepBlocker(attr_frame_encoder="autoencoder", rel_frame_encoder="autoencoder")
        >>> blocks = blocker.assign(left=ds.left, right=ds.right, left_rel=ds.left_rel, right_rel=ds.right_rel)
    """

    _attribute_blocker: DeepBlocker
    _relation_blocker: DeepBlocker

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
        save: bool = True,
        save_dir: Optional[Union[str, pathlib.Path]] = None,
        force: bool = False,
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
        # set after instatiating seperate blocker to use setter
        self.save = save
        self.force = force
        self.save_dir = save_dir

    @property
    def save(self) -> bool:
        return self._save

    @save.setter
    def save(self, value: bool):
        self._save = value
        self._attribute_blocker.save = value
        self._relation_blocker.save = value

    @property
    def force(self) -> bool:
        return self._force

    @force.setter
    def force(self, value: bool):
        self._force = value
        self._attribute_blocker.force = value
        self._relation_blocker.force = value

    @property
    def save_dir(self) -> Optional[Union[str, pathlib.Path]]:
        return self._save_dir

    @save_dir.setter
    def save_dir(self, value: Optional[Union[str, pathlib.Path]]):
        if value is None:
            self._save_dir = None
            self._attribute_blocker.save_dir = None
            self._relation_blocker.save_dir = None
        else:
            sd = pathlib.Path(value)
            self._save_dir = sd
            self._attribute_blocker.save_dir = sd.joinpath("attributes")
            self._relation_blocker.save_dir = sd.joinpath("relation")
