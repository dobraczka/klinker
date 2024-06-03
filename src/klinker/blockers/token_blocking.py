import logging
from typing import Dict, Any, Type, Literal
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
from typing import Callable, List, Optional
from ..utils import concat_frames

import dask.dataframe as dd
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from ..data import KlinkerFrame
from ..data.blocks import KlinkerBlockManager
from ..typing import Frame, SeriesType
from .base import SchemaAgnosticBlocker, Blocker

logger = logging.getLogger(__name__)


class FilteredTokenizer:
    def __init__(self, tokenize_fn=None, min_token_length=3, stop_words=None):
        if not tokenize_fn:
            tokenize_fn = word_tokenize
        if not stop_words:
            stop_words = stopwords.words("english")
        self.tokenize_fn = tokenize_fn
        self.stop_words = stop_words
        self.min_token_length = min_token_length

    def tokenize(self, x, return_set: bool = False):
        tokens = filter(
            lambda tok: len(tok) >= self.min_token_length
            and tok not in self.stop_words,
            self.tokenize_fn(str(x.lower())),
        )
        if return_set:
            return set(tokens)
        return list(tokens)


# def tokenize_series(x, tokenize_fn, min_token_length, stop_words):
#     """Tokenize a series and return set.

#     Args:
#     ----
#       x: series with values to tokenize
#       tokenize_fn: tokenization function
#       min_token_length: minimum length of tokens
#       stop_words: words to ignore

#     Returns:
#     -------
#         set of tokens
#     """
#     return set(
#         filter(
#             lambda tok: len(tok) >= min_token_length and tok not in stop_words,
#             tokenize_fn(str(x.lower())),
#         )
#     )


class TokenBlocker(SchemaAgnosticBlocker):
    """Concatenates and tokenizes entity attribute values and blocks on tokens.

    Examples
    --------
        >>> # doctest: +SKIP
        >>> from sylloge import MovieGraphBenchmark
        >>> from klinker.data import KlinkerDataset
        >>> ds = KlinkerDataset.from_sylloge(MovieGraphBenchmark(),clean=True)
        >>> from klinker.blockers import TokenBlocker
        >>> blocker = TokenBlocker()
        >>> blocks = blocker.assign(left=ds.left, right=ds.right)

    """

    def __init__(
        self,
        tokenize_fn: Callable[[str], List[str]] = word_tokenize,
        stop_words: Optional[List[str]] = None,
        min_token_length: int = 3,
    ):
        self.tokenizer = FilteredTokenizer(
            tokenize_fn=tokenize_fn,
            min_token_length=min_token_length,
            stop_words=stop_words,
        )

    def _tok_block(self, tab: SeriesType) -> Frame:
        """Perform token blocking on this series.

        Args:
        ----
          tab: SeriesType: series on which token blocking should be done.

        Returns:
        -------
            token blocked series.
        """
        name = tab.name
        id_col_name = tab.index.name
        # TODO figure out why this hack is needed
        # i.e. why does dask assume later for the join, that this is named 0
        # no matter what it is actually named
        tok_name = "tok"
        tok_kwargs: Dict[str, Any] = {
            "return_set": True,
        }
        collect_ids_kwargs = {"id_col": id_col_name}
        if isinstance(tab, dd.Series):
            tok_kwargs["meta"] = (tab.name, "O")
            collect_ids_kwargs["meta"] = pd.Series(
                [],
                name=tab.name,
                dtype="O",
                index=pd.Series([], dtype="O", name=tok_name),
            )
        return (
            tab.apply(self.tokenizer.tokenize, **tok_kwargs)
            .explode()
            .dropna()
            .to_frame()
            .reset_index()
            .rename(columns={name: tok_name})  # avoid same name for col and index
            .groupby(tok_name)
            .apply(lambda x, id_col: list(set(x[id_col])), **collect_ids_kwargs)
            .to_frame(name=name)
        )

    def _assign(
        self,
        left: SeriesType,
        right: SeriesType,
        left_rel: Optional[KlinkerFrame] = None,
        right_rel: Optional[KlinkerFrame] = None,
    ) -> KlinkerBlockManager:
        """Assign entity ids to blocks.

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
        left_tok = self._tok_block(left)
        right_tok = self._tok_block(right)
        left_tok.to_parquet("/tmp/tb_left_blocks.parquet")
        right_tok.to_parquet("/tmp/tb_right_blocks.parquet")
        pd_blocks = left_tok.join(right_tok, how="inner")
        if isinstance(pd_blocks, dd.DataFrame):
            return KlinkerBlockManager(pd_blocks)
        return KlinkerBlockManager.from_pandas(pd_blocks)


class UniqueNameBlocker(Blocker):
    def assign(
        self,
        left: KlinkerFrame,
        right: KlinkerFrame,
        left_rel: Optional[KlinkerFrame] = None,
        right_rel: Optional[KlinkerFrame] = None,
    ) -> KlinkerBlockManager:
        def filter_nonunique(attr_frame: KlinkerFrame, head_col, tail_col):
            if isinstance(attr_frame, pd.DataFrame):
                return attr_frame.groupby(tail_col).filter(
                    lambda x: x[head_col].nunique() == 1
                )
            return attr_frame.groupby(tail_col).apply(
                lambda x: x if x[head_col].nunique() == 1 else None,
                meta=attr_frame._meta,
            )

        if not list(left.columns) == list(right.columns):
            raise ValueError(
                f"Need identical column names but got {left.columns} and {right.columns}"
            )
        head_col = left.columns[0]
        tail_col = left.columns[2]
        left_unique = filter_nonunique(left, head_col, tail_col)
        right_unique = filter_nonunique(right, head_col, tail_col)
        lhead = f"{head_col}_x"
        rhead = f"{head_col}_y"
        res = left_unique.merge(right_unique, on=tail_col, how="inner")[
            [lhead, rhead]
        ].rename(columns={lhead: left.table_name, rhead: right.table_name})
        # avoid problems downstream because of non-str index
        res = res.set_index("unique_" + res.index.astype(str))
        res = res.applymap(lambda x: [x])
        if isinstance(res, pd.DataFrame):
            return KlinkerBlockManager.from_pandas(res)
        return KlinkerBlockManager(res)


class _MyVectorizerMixin:
    vectorizer_cls: Type

    def __init__(self, **vectorizer_kwargs):
        if vectorizer_kwargs == {}:
            vectorizer_kwargs["analyzer"] = FilteredTokenizer().tokenize
        self.vectorizer_kwargs = vectorizer_kwargs

    def vectorize(self, left: KlinkerFrame, right: KlinkerFrame):
        vec = self.__class__.vectorizer_cls(**self.vectorizer_kwargs)
        all_conc = concat_frames([left, right])
        return vec.fit_transform(all_conc), vec


class PartitioningTokenBlocker(_MyVectorizerMixin, SchemaAgnosticBlocker):
    vectorizer_cls = CountVectorizer

    def _assign(
        self,
        left: KlinkerFrame,
        right: KlinkerFrame,
        left_rel: Optional[KlinkerFrame] = None,
        right_rel: Optional[KlinkerFrame] = None,
    ) -> KlinkerBlockManager:
        X, _ = super().vectorize(left, right)
        X_left = X[: len(left)]
        X_right = X[len(left) :]

        # get overlaps via mm
        res = X_left @ X_right.T

        # get right entities by utilizing sparse matrix construction
        # see https://stackoverflow.com/a/52299730
        # and https://stackoverflow.com/a/24792612
        right_val = np.split(right.index[res.indices].to_numpy(), res.indptr[1:-1])
        blocks = (
            pd.Series(right_val, index=left.index)
            .to_frame(right.name)
            .reset_index(names=["tmp"])
        )
        blocks[left.name] = blocks["tmp"].apply(lambda x: [x])
        return KlinkerBlockManager.from_pandas(blocks[[left.name, right.name]])


class AttributeClusteringTokenBlocker(TokenBlocker):
    vectorizer_cls = CountVectorizer

    def __init__(self):
        super().__init__(min_token_length=0, tokenize_fn=lambda x: x.split(" "))

    def _conc_cluster_labels(self, frame, labels):
        from klinker.data.enhanced_df import KlinkerPandasFrame

        entity_cluster_labels = pd.DataFrame.from_dict(
            dict(head=frame["head"].values, labels=labels)
        )
        entity_cluster_labels["labels"] = entity_cluster_labels["labels"].replace(
            -1, np.nan
        )
        return KlinkerPandasFrame.from_df(
            entity_cluster_labels, id_col="head", table_name=frame.table_name
        ).concat_values()

    def assign(
        self,
        left: KlinkerFrame,
        right: KlinkerFrame,
        left_rel: Optional[KlinkerFrame] = None,
        right_rel: Optional[KlinkerFrame] = None,
    ) -> KlinkerBlockManager:
        from sentence_transformers import SentenceTransformer

        try:
            from cuml.cluster import HDBSCAN
        except ImportError:
            from hdbscan import HDBSCAN
        st = SentenceTransformer("gtr-t5-base")
        left_enc = st.encode(left["tail"].values)
        right_enc = st.encode(right["tail"].values)

        labels = HDBSCAN().fit_predict(np.concatenate([left_enc, right_enc]))
        labels = np.random.randint(-1, 50, size=len(left) + len(right))

        left_conc = self._conc_cluster_labels(left, labels[: len(left)])
        right_conc = self._conc_cluster_labels(right, labels[len(left) :])
        return super()._assign(left_conc, right_conc)


class TfIdfFilteredTokenBlocker(_MyVectorizerMixin, SchemaAgnosticBlocker):
    vectorizer_cls = TfidfVectorizer

    def __init__(self, threshold: float, **vectorizer_kwargs):
        self.threshold = threshold
        super().__init__(**vectorizer_kwargs)

    def _build_side(self, X_part, conc, tokens):
        return (
            pd.Series(
                np.split(tokens[X_part.indices], X_part.indptr[1:-1]), index=conc.index
            )
            .explode()
            .dropna()
            .to_frame("tok")
            .reset_index()
            .groupby("tok")
            .apply(lambda x: list(set(x["head"])))
            .to_frame(name=conc.name)
        )

    def _assign(
        self,
        left: SeriesType,
        right: SeriesType,
        left_rel: Optional[KlinkerFrame] = None,
        right_rel: Optional[KlinkerFrame] = None,
    ) -> KlinkerBlockManager:
        X, vec = super().vectorize(left, right)
        if self.threshold > 0.0:
            X = X >= self.threshold
        else:
            logger.warn(
                "Threshold is {self.threshold} and cannot be used! This turns this into a simple token blocker! Set threshold to > 0!"
            )
        tokens = vec.get_feature_names_out()
        left_blocks = self._build_side(X[: len(left)], left, tokens)
        right_blocks = self._build_side(X[len(left) :], right, tokens)
        return KlinkerBlockManager.from_pandas(
            left_blocks.join(right_blocks, how="inner")
        )


class DaskTfidfVectorizer:
    def __init__(
        self,
        *,
        norm: Optional[Literal["l2", "l1"]] = "l2",
        use_idf: bool = True,
        smooth_idf: bool = True,
        sublinear_tf: bool = False,
        persist_idf_array: bool = True,
        **vectorizer_kwargs,
    ):
        try:
            from dask_tfidf import DaskTfidfTransformer
        except ImportError:
            raise ImportError("Please install dask_tfidf!")
        from dask_ml.feature_extraction.text import (
            CountVectorizer as DaskCountVectorizer,
        )

        self.count_vec = DaskCountVectorizer(**vectorizer_kwargs)
        self.tfidf_transformer = DaskTfidfTransformer(
            norm=norm,
            use_idf=use_idf,
            smooth_idf=smooth_idf,
            sublinear_tf=sublinear_tf,
            persist_idf_array=persist_idf_array,
        )

    def fit_transform(self, x):
        import dask.bag as db

        counts = self.count_vec.fit_transform(db.from_sequence(x))
        counts.persist()
        return self.tfidf_transformer.fit_transform(counts).compute().tocsr()

    def get_feature_names_out(self, input_features=None):
        return self.count_vec.get_feature_names_out(input_features=input_features)


class DaskTfIdfFilteredTokenBlocker(TfIdfFilteredTokenBlocker):
    vectorizer_cls = DaskTfidfVectorizer

    def _build_side(self, X_part, conc, tokens):
        name = conc.name
        id_col_name = conc.index.name
        # TODO figure out why this hack is needed
        # i.e. why does dask assume later for the join, that this is named 0
        # no matter what it is actually named
        # tok_name = "tok"
        collect_ids_kwargs = {"id_col": id_col_name}
        # collect_ids_kwargs["meta"] = pd.Series(
        #     [],
        #     name=conc.name,
        #     dtype="O",
        #     index=pd.Series([], dtype="O", name=tok_name),
        # )
        return (
            pd.Series(
                np.split(tokens[X_part.indices], X_part.indptr[1:-1]), index=conc.index
            )
            .explode()
            .dropna()
            .to_frame("tok")
            .reset_index()
            .groupby("tok")
            .apply(lambda x, id_col: list(set(x[id_col])), **collect_ids_kwargs)
            .to_frame(name=name)
        )

    def _assign(
        self,
        left: SeriesType,
        right: SeriesType,
        left_rel: Optional[KlinkerFrame] = None,
        right_rel: Optional[KlinkerFrame] = None,
    ) -> KlinkerBlockManager:
        left = left.persist()
        right = right.persist()
        X, vec = super().vectorize(left, right)
        if self.threshold > 0.0:
            X = X >= self.threshold
        else:
            logger.warn(
                "Threshold is {self.threshold} and cannot be used! This turns this into a simple token blocker! Set threshold to > 0!"
            )
        tokens = vec.get_feature_names_out()
        left_blocks = self._build_side(X[: len(left)], left, tokens)
        right_blocks = self._build_side(X[len(left) :], right, tokens)
        return KlinkerBlockManager.from_pandas(
            left_blocks.join(right_blocks, how="inner")
        )


if __name__ == "__main__":
    from klinker.data import KlinkerDataset
    from sylloge import OpenEA
    from klinker.eval import Evaluation

    ds = KlinkerDataset.from_sylloge(OpenEA(), clean=True)
    blocker = AttributeClusteringTokenBlocker()
    blocks = blocker.assign(ds.left, ds.right)
    print(Evaluation.from_dataset(blocks, ds).to_dict())

    print("TokenBlocker:")
    blocker = TokenBlocker()
    blocks = blocker.assign(ds.left, ds.right)
    print(Evaluation.from_dataset(blocks, ds).to_dict())
