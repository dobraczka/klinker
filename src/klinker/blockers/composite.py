from .relation_aware import (
    BaseRelationalBlocker,
    SimpleRelationalTokenBlocker,
    concat_neighbor_attributes,
)
import logging
from ..encoders import TokenizedFrameEncoder
from klinker.typing import SeriesType
from .attribute_clustering import (
    AttributeClusteringTokenBlocker,
    TokenClusteringTokenBlocker,
    TokenClusteringMinHashLSHBlocker,
    AttributeClusteringMinHashLSHBlocker,
    NoiseClusterHandling,
)
from class_resolver import HintOrType, OptionalKwargs
from typing import Callable, List, Optional, Type, Tuple

import dask.dataframe as dd
import pandas as pd
from nltk.tokenize import word_tokenize


from ..data import (
    KlinkerBlockManager,
    KlinkerFrame,
    combine_blocks,
)
from .base import SchemaAgnosticBlocker
from .token_blocking import TokenBlocker, UniqueNameBlocker

logger = logging.getLogger(__name__)


def filter_with_unique(conc, unique_blocks_side):
    was_series = True
    if isinstance(conc, (pd.Series, dd.Series)):
        value_df = conc.to_frame("values")
    else:
        was_series = False
        value_df = conc.set_index(conc.id_col)
    filter_df = unique_blocks_side.explode().to_frame("filter")
    if not isinstance(value_df, dd.DataFrame):
        filter_df = filter_df.compute()
    joined = value_df.merge(
        filter_df, left_index=True, right_on="filter", how="left", indicator=True
    )
    mask = joined[joined["_merge"] == "left_only"]["filter"]
    if was_series:
        return conc.loc[mask]
    return value_df.loc[mask].reset_index()


class BaseCompositeUniqueNameBlocker(BaseRelationalBlocker):
    _attribute_blocker: SchemaAgnosticBlocker
    _relation_blocker: SchemaAgnosticBlocker
    _attr_blocker_cls: Type[SchemaAgnosticBlocker]
    _rel_blocker_cls: Type[SchemaAgnosticBlocker]

    def __init__(
        self,
        top_n_a: Optional[int] = None,
        top_n_r: Optional[int] = None,
        attr_blocker_kwargs=None,
        rel_blocker_kwargs=None,
        use_unique_name: bool = True,
    ):
        super().__init__(top_n_a=top_n_a, top_n_r=top_n_r)
        attr_blocker_kwargs = {} if not attr_blocker_kwargs else attr_blocker_kwargs
        rel_blocker_kwargs = {} if not rel_blocker_kwargs else rel_blocker_kwargs
        self._attribute_blocker = self.__class__._attr_blocker_cls(
            **attr_blocker_kwargs
        )
        self._relation_blocker = self.__class__._rel_blocker_cls(**rel_blocker_kwargs)
        self.use_unique_name = use_unique_name

    def _compute_attr_blocks(self, left, right, unique_blocks) -> KlinkerBlockManager:
        if self.use_unique_name:
            left_attr_filtered = filter_with_unique(
                left, unique_blocks.blocks[left.table_name]
            )
            right_attr_filtered = filter_with_unique(
                right, unique_blocks.blocks[right.table_name]
            )
            if len(left_attr_filtered) == 0 or len(right_attr_filtered) == 0:
                logging.info(
                    "Nothing left to do for attr_blocks because unique got everything!"
                )
                return unique_blocks
            return combine_blocks(
                unique_blocks,
                self._attribute_blocker.assign(left_attr_filtered, right_attr_filtered),
            )
        return self._attribute_blocker.assign(left, right)

    def _compute_rel_blocks(
        self, left, right, left_rel, right_rel, unique_blocks
    ) -> Optional[KlinkerBlockManager]:
        left_conc, right_conc = self.concat_relational_info(
            left=left, right=right, left_rel=left_rel, right_rel=right_rel
        )
        left_filtered = left
        right_filtered = right
        if self.use_unique_name:
            logger.info("Got relational information")
            left_filtered = filter_with_unique(
                left_conc, unique_blocks.blocks[left.table_name]
            )
            right_filtered = filter_with_unique(
                right_conc, unique_blocks.blocks[right.table_name]
            )
            logger.info("Filtered relational information")
            if len(left_filtered) == 0 or len(right_filtered) == 0:
                logging.info(
                    "Nothing left to do for rel_blocks because unique got everything!"
                )
                return None
        return self._relation_blocker._assign(left=left_filtered, right=right_filtered)

    def assign(
        self,
        left: KlinkerFrame,
        right: KlinkerFrame,
        left_rel: Optional[KlinkerFrame] = None,
        right_rel: Optional[KlinkerFrame] = None,
    ) -> KlinkerBlockManager:
        assert left_rel is not None
        assert right_rel is not None
        unique_blocks = None
        if self.use_unique_name:
            unique_blocks = UniqueNameBlocker().assign(left, right)
            unique_blocks.blocks.persist()
            logger.info("Done with unique!")

        attr_blocks = self._compute_attr_blocks(left, right, unique_blocks)
        logger.info("Done with attr!")
        rel_blocks = self._compute_rel_blocks(
            left, right, left_rel, right_rel, unique_blocks
        )
        logger.info("Done with rel!")
        if rel_blocks is None:
            return attr_blocks
        return combine_blocks(attr_blocks, rel_blocks)


class BaseAttrTokenCompositeUniqueNameBlocker(BaseCompositeUniqueNameBlocker):
    _attr_blocker_cls = TokenBlocker

    def __init__(
        self,
        top_n_a: Optional[int] = None,
        top_n_r: Optional[int] = None,
        tokenize_fn: Callable[[str], List[str]] = word_tokenize,
        min_token_length: int = 3,
        rel_blocker_kwargs=None,
    ):
        super().__init__(
            top_n_a=top_n_a,
            top_n_r=top_n_r,
            attr_blocker_kwargs=dict(
                tokenize_fn=tokenize_fn, min_token_length=min_token_length
            ),
            rel_blocker_kwargs=rel_blocker_kwargs,
        )


class CompositeRelationalTokenBlocker(SimpleRelationalTokenBlocker):
    def __init__(
        self,
        tokenize_fn: Callable[[str], List[str]] = word_tokenize,
        min_token_length: int = 3,
        top_n_a: Optional[int] = None,
        top_n_r: Optional[int] = None,
    ):
        super().__init__(
            tokenize_fn=tokenize_fn,
            min_token_length=min_token_length,
            top_n_a=top_n_a,
            top_n_r=top_n_r,
        )
        self._unique_blocker = UniqueNameBlocker()

    def assign(
        self,
        left: KlinkerFrame,
        right: KlinkerFrame,
        left_rel: Optional[KlinkerFrame] = None,
        right_rel: Optional[KlinkerFrame] = None,
    ) -> KlinkerBlockManager:
        unique_blocks = self._unique_blocker.assign(left, right)
        unique_blocks.blocks.persist()
        left_conc, right_conc = self.concat_relational_info(
            left=left, right=right, left_rel=left_rel, right_rel=right_rel
        )
        left_filtered = filter_with_unique(
            left_conc, unique_blocks.blocks[left.table_name]
        )
        right_filtered = filter_with_unique(
            right_conc, unique_blocks.blocks[right.table_name]
        )
        return combine_blocks(
            unique_blocks,
            self._blocker._assign(left=left_filtered, right=right_filtered),
        )


class BaseCompositeRelationalClusteringBlocker(BaseCompositeUniqueNameBlocker):
    _relation_blocker: AttributeClusteringTokenBlocker
    _rel_blocker_cls = AttributeClusteringTokenBlocker

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
            left,
            left_rel,
            include_own_attributes=False,
            top_n_a=self.top_n_a,
            top_n_r=self.top_n_r,
            do_not_concat_values=True,
        )
        right_conc = concat_neighbor_attributes(
            right,
            right_rel,
            include_own_attributes=False,
            top_n_a=self.top_n_a,
            top_n_r=self.top_n_r,
            do_not_concat_values=True,
        )
        return left_conc, right_conc

    def _compute_rel_blocks(
        self, left, right, left_rel, right_rel, unique_blocks
    ) -> KlinkerBlockManager:
        logger.info("Gathering rel info")
        left_conc, right_conc = self.concat_relational_info(
            left=left, right=right, left_rel=left_rel, right_rel=right_rel
        )
        logger.info("Got rel info")
        left_conc = left_conc.drop_duplicates()
        right_conc = right_conc.drop_duplicates()
        logger.info("Deduplicated")
        left_filtered = left_conc
        right_filtered = right_conc
        if self.use_unique_name:
            left_filtered = filter_with_unique(
                left_conc, unique_blocks.blocks[left.table_name]
            )
            right_filtered = filter_with_unique(
                right_conc, unique_blocks.blocks[right.table_name]
            )
        return self._relation_blocker.assign(left=left_filtered, right=right_filtered)


class CompositeRelationalAttributeClusteringBlocker(
    BaseCompositeRelationalClusteringBlocker
):
    _attribute_blocker: TokenBlocker
    _attr_blocker_cls = TokenBlocker
    _relation_blocker: AttributeClusteringTokenBlocker
    _rel_blocker_cls = AttributeClusteringTokenBlocker

    def __init__(
        self,
        top_n_a: Optional[int] = None,
        top_n_r: Optional[int] = None,
        tokenize_fn: Callable[[str], List[str]] = word_tokenize,
        stop_words: Optional[List[str]] = None,
        min_token_length: int = 3,
        encoder: HintOrType[TokenizedFrameEncoder] = None,
        encoder_kwargs: OptionalKwargs = None,
        min_cluster_size: int = 5,
        min_samples: Optional[int] = None,
        cluster_selection_epsilon: float = 0.0,
        metric: str = "euclidean",
        alpha: float = 1.0,
        p: Optional[float] = None,
        cluster_selection_method: str = "eom",
        noise_cluster_handling: NoiseClusterHandling = "remove",
        use_unique_name: bool = True,
    ):
        super().__init__(
            top_n_a=top_n_a,
            top_n_r=top_n_r,
            use_unique_name=use_unique_name,
            attr_blocker_kwargs=dict(
                tokenize_fn=tokenize_fn,
                stop_words=stop_words,
                min_token_length=min_token_length,
            ),
            rel_blocker_kwargs=dict(
                encoder=encoder,
                encoder_kwargs=encoder_kwargs,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_epsilon=cluster_selection_epsilon,
                metric=metric,
                alpha=alpha,
                p=p,
                cluster_selection_method=cluster_selection_method,
                noise_cluster_handling=noise_cluster_handling,
                tokenize_fn=tokenize_fn,
                stop_words=stop_words,
                min_token_length=min_token_length,
            ),
        )


class CompositeRelationalTokenClusteringBlocker(
    BaseCompositeRelationalClusteringBlocker
):
    _attribute_blocker: TokenBlocker
    _attr_blocker_cls = TokenBlocker
    _relation_blocker: TokenClusteringTokenBlocker
    _rel_blocker_cls = TokenClusteringTokenBlocker

    def __init__(
        self,
        top_n_a: Optional[int] = None,
        top_n_r: Optional[int] = None,
        tokenize_fn: Callable[[str], List[str]] = word_tokenize,
        stop_words: Optional[List[str]] = None,
        min_token_length: int = 3,
        encoder: HintOrType[TokenizedFrameEncoder] = None,
        encoder_kwargs: OptionalKwargs = None,
        min_cluster_size: int = 5,
        min_samples: Optional[int] = None,
        cluster_selection_epsilon: float = 0.0,
        metric: str = "euclidean",
        alpha: float = 1.0,
        p: Optional[float] = None,
        cluster_selection_method: str = "eom",
        noise_cluster_handling: NoiseClusterHandling = "remove",
        use_unique_name: bool = True,
    ):
        super().__init__(
            top_n_a=top_n_a,
            top_n_r=top_n_r,
            use_unique_name=use_unique_name,
            attr_blocker_kwargs=dict(
                tokenize_fn=tokenize_fn,
                stop_words=stop_words,
                min_token_length=min_token_length,
            ),
            rel_blocker_kwargs=dict(
                encoder=encoder,
                encoder_kwargs=encoder_kwargs,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_epsilon=cluster_selection_epsilon,
                metric=metric,
                alpha=alpha,
                p=p,
                cluster_selection_method=cluster_selection_method,
                noise_cluster_handling=noise_cluster_handling,
                tokenize_fn=tokenize_fn,
                stop_words=stop_words,
                min_token_length=min_token_length,
            ),
        )


class CompositeRelationalAttributeClusteringLSHBlocker(
    BaseCompositeRelationalClusteringBlocker
):
    _attribute_blocker: TokenBlocker
    _attr_blocker_cls = TokenBlocker
    _relation_blocker: AttributeClusteringMinHashLSHBlocker
    _rel_blocker_cls = AttributeClusteringMinHashLSHBlocker

    def __init__(
        self,
        top_n_a: Optional[int] = None,
        top_n_r: Optional[int] = None,
        tokenize_fn: Callable[[str], List[str]] = word_tokenize,
        stop_words: Optional[List[str]] = None,
        min_token_length: int = 3,
        encoder: HintOrType[TokenizedFrameEncoder] = None,
        encoder_kwargs: OptionalKwargs = None,
        min_cluster_size: int = 5,
        min_samples: Optional[int] = None,
        cluster_selection_epsilon: float = 0.0,
        metric: str = "euclidean",
        alpha: float = 1.0,
        p: Optional[float] = None,
        cluster_selection_method: str = "eom",
        noise_cluster_handling: NoiseClusterHandling = "remove",
        rel_threshold: float = 0.5,
        rel_num_perm: int = 128,
        rel_weights: Tuple[float, float] = (0.5, 0.5),
        use_unique_name: bool = True,
    ):
        super().__init__(
            top_n_a=top_n_a,
            top_n_r=top_n_r,
            use_unique_name=use_unique_name,
            attr_blocker_kwargs=dict(
                tokenize_fn=tokenize_fn,
                stop_words=stop_words,
                min_token_length=min_token_length,
            ),
            rel_blocker_kwargs=dict(
                encoder=encoder,
                encoder_kwargs=encoder_kwargs,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_epsilon=cluster_selection_epsilon,
                metric=metric,
                alpha=alpha,
                p=p,
                cluster_selection_method=cluster_selection_method,
                noise_cluster_handling=noise_cluster_handling,
                threshold=rel_threshold,
                num_perm=rel_num_perm,
                weights=rel_weights,
            ),
        )


class CompositeRelationalTokenClusteringLSHBlocker(
    BaseCompositeRelationalClusteringBlocker
):
    _attribute_blocker: TokenBlocker
    _attr_blocker_cls = TokenBlocker
    _relation_blocker: TokenClusteringMinHashLSHBlocker
    _rel_blocker_cls = TokenClusteringMinHashLSHBlocker

    def __init__(
        self,
        top_n_a: Optional[int] = None,
        top_n_r: Optional[int] = None,
        tokenize_fn: Callable[[str], List[str]] = word_tokenize,
        stop_words: Optional[List[str]] = None,
        min_token_length: int = 3,
        encoder: HintOrType[TokenizedFrameEncoder] = None,
        encoder_kwargs: OptionalKwargs = None,
        min_cluster_size: int = 5,
        min_samples: Optional[int] = None,
        cluster_selection_epsilon: float = 0.0,
        metric: str = "euclidean",
        alpha: float = 1.0,
        p: Optional[float] = None,
        cluster_selection_method: str = "eom",
        noise_cluster_handling: NoiseClusterHandling = "remove",
        rel_threshold: float = 0.5,
        rel_num_perm: int = 128,
        rel_weights: Tuple[float, float] = (0.5, 0.5),
        use_unique_name: bool = True,
    ):
        super().__init__(
            top_n_a=top_n_a,
            top_n_r=top_n_r,
            use_unique_name=use_unique_name,
            attr_blocker_kwargs=dict(
                tokenize_fn=tokenize_fn,
                stop_words=stop_words,
                min_token_length=min_token_length,
            ),
            rel_blocker_kwargs=dict(
                encoder=encoder,
                encoder_kwargs=encoder_kwargs,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_epsilon=cluster_selection_epsilon,
                metric=metric,
                alpha=alpha,
                p=p,
                cluster_selection_method=cluster_selection_method,
                noise_cluster_handling=noise_cluster_handling,
                tokenize_fn=tokenize_fn,
                stop_words=stop_words,
                min_token_length=min_token_length,
                threshold=rel_threshold,
                num_perm=rel_num_perm,
                weights=rel_weights,
            ),
        )


if __name__ == "__main__":
    from klinker.data import KlinkerDataset
    from sylloge import OpenEA
    from klinker.eval import Evaluation
    from klinker.blockers import SimpleRelationalTokenBlocker

    ds = KlinkerDataset.from_sylloge(OpenEA(), clean=True).sample(0.01)
    # blocks = SimpleRelationalTokenBlocker().assign(
    #     ds.left, ds.right, ds.left_rel, ds.right_rel
    # )
    # print("\n====================")
    # print("Basic")
    # print(Evaluation.from_dataset(blocks, ds).to_dict())
    # print("====================\n")

    # blocks = CompositeRelationalTokenBlocker().assign(
    #     ds.left, ds.right, ds.left_rel, ds.right_rel
    # )
    # print("\n====================")
    # print("Basic Composite")
    # print(Evaluation.from_dataset(blocks, ds).to_dict())
    # print("====================\n")

    # blocks = CompositeRelationalAttributeClusteringBlocker().assign(
    #     ds.left, ds.right, ds.left_rel, ds.right_rel
    # )
    # print("\n====================")
    # print("Attribute + SIF")
    # print(Evaluation.from_dataset(blocks, ds).to_dict())
    # print("====================\n")

    # blocks = CompositeRelationalTokenClusteringBlocker().assign(
    #     ds.left, ds.right, ds.left_rel, ds.right_rel
    # )
    # print("\n====================")
    # print("Token + SIF")
    # print(Evaluation.from_dataset(blocks, ds).to_dict())
    # print("====================\n")

    # blocks = CompositeRelationalAttributeClusteringBlocker(
    #     encoder="sentencetransformertokenized"
    # ).assign(ds.left, ds.right, ds.left_rel, ds.right_rel)
    # print("\n====================")
    # print("Attribute + SenTrans")
    # print(Evaluation.from_dataset(blocks, ds).to_dict())
    # print("====================\n")
    # blocks = CompositeRelationalTokenClusteringBlocker(
    #     encoder="sentencetransformertokenized"
    # ).assign(ds.left, ds.right, ds.left_rel, ds.right_rel)
    # print("\n====================")
    # print("Token + SenTrans")
    # print(Evaluation.from_dataset(blocks, ds).to_dict())
    # print("====================\n")

    threshold = 0.3
    blocks = CompositeRelationalAttributeClusteringLSHBlocker(
        rel_threshold=threshold
    ).assign(ds.left, ds.right, ds.left_rel, ds.right_rel)
    print("\n====================")
    print("LSH Attribute + SIF")
    print(Evaluation.from_dataset(blocks, ds).to_dict())
    print("====================\n")

    blocks = CompositeRelationalTokenClusteringLSHBlocker(
        rel_threshold=threshold
    ).assign(ds.left, ds.right, ds.left_rel, ds.right_rel)
    print("\n====================")
    print("LSH Token + SIF")
    print(Evaluation.from_dataset(blocks, ds).to_dict())
    print("====================\n")

    blocks = CompositeRelationalAttributeClusteringLSHBlocker(
        rel_threshold=threshold, encoder="sentencetransformertokenized"
    ).assign(ds.left, ds.right, ds.left_rel, ds.right_rel)
    print("\n====================")
    print("LSH Attribute + SenTrans")
    print(Evaluation.from_dataset(blocks, ds).to_dict())
    print("====================\n")
    blocks = CompositeRelationalTokenClusteringLSHBlocker(
        rel_threshold=threshold, encoder="sentencetransformertokenized"
    ).assign(ds.left, ds.right, ds.left_rel, ds.right_rel)
    print("\n====================")
    print("LSH Token + SenTrans")
    print(Evaluation.from_dataset(blocks, ds).to_dict())
    print("====================\n")
