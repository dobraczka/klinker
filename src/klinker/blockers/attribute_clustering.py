from .token_blocking import TokenBlocker
import os
import torch
import numpy as np
import logging
from .lsh import MinHashLSHBlocker
from typing import Callable, List, Optional, Tuple, Literal

import pandas as pd
from nltk.tokenize import word_tokenize

from ..data import KlinkerFrame
from ..data.blocks import KlinkerBlockManager
from klinker.data import KlinkerPandasFrame, KlinkerTriplePandasFrame
from ..encoders import TokenizedFrameEncoder, frame_encoder_resolver
from class_resolver import HintOrType, OptionalKwargs

try:
    from cuml.cluster import HDBSCAN
except ImportError:
    from hdbscan import HDBSCAN

NoiseClusterHandling = Literal["remove", "token", "keep"]

logger = logging.getLogger(__name__)


class TokenClusteringMixin:
    _CLUSTERING_LABEL_PREFIX = "clustering_label_"

    def __init__(
        self,
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
        save_dir: Optional[str] = None,
        save: bool = False,
        **kwargs,
    ):
        self.encoder = frame_encoder_resolver.make(encoder, encoder_kwargs)
        self.hdbscan = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            metric=metric,
            alpha=alpha,
            p=p,
            cluster_selection_method=cluster_selection_method,
        )
        self.noise_cluster_handling = noise_cluster_handling
        self.save = save
        self.save_dir = save_dir
        super().__init__(**kwargs)

    def _conc_cluster_labels(
        self,
        frame,
        labels,
        id_col: str = "head",
        frame_col: str = "tail",
        val_col="value",
        label_col="cluster_label",
    ):
        res = KlinkerTriplePandasFrame.from_df(
            frame.merge(labels, left_on=frame_col, right_on=val_col)[
                [id_col, frame_col, label_col]
            ],
            id_col=frame.id_col,
            table_name=frame.table_name,
        )
        return res

    def _get_all_embeddings(self, left, right, value_col_name):
        left_vals = left[value_col_name].unique()
        right_vals = right[value_col_name].unique()
        # TODO adapt for dask
        left_df = pd.Series(left_vals, index=left_vals).to_frame("values")
        right_df = pd.Series(right_vals, index=right_vals).to_frame("values")
        left_emb, right_emb = self.encoder.encode(left_df, right_df)
        return left_emb, right_emb

    def _handle_noise_cluster(self, val_cluster_label):
        if self.noise_cluster_handling == "keep":
            return val_cluster_label
        if self.noise_cluster_handling == "remove":
            return val_cluster_label[val_cluster_label["cluster_label"] > 0]
        if self.noise_cluster_handling == "token":
            val_cluster_label.loc[
                val_cluster_label["cluster_label"] == -1, "cluster_label"
            ] = val_cluster_label[val_cluster_label["cluster_label"] == -1]["value"]
            return val_cluster_label

    def _save_embeddings_if_wanted(self, left_emb, right_emb, left_name, right_name):
        if self.save_dir is None:
            self.save_dir = f"clustering_{left_name}_{right_name}"
        if self.save:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            left_emb.to_pickle(os.path.join(self.save_dir, "left_emb"))
            right_emb.to_pickle(os.path.join(self.save_dir, "right_emb"))
            logger.info(f"Saved embeddings in {self.save_dir}")

    def _save_labels_if_wanted(self, labels: np.ndarray):
        if self.save:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            np.save(os.path.join(self.save_dir, "cluster_labels"), labels)
            logger.info(f"Saved cluster_labels in {self.save_dir}")

    def embed_and_cluster(
        self,
        left: KlinkerFrame,
        right: KlinkerFrame,
        value_col_name: str = "tail",
    ) -> Tuple[KlinkerFrame, KlinkerFrame]:
        left_emb, right_emb = self._get_all_embeddings(left, right, value_col_name)
        self._save_embeddings_if_wanted(
            left_emb, right_emb, left.table_name, right.table_name
        )
        all_vec = left_emb._tensor_lib.concatenate(
            [left_emb.vectors, right_emb.vectors]
        )
        try:
            labels = self.hdbscan.fit_predict(all_vec)
        except TypeError as err:  # happens for torch float32 and cuml....
            if isinstance(all_vec, torch.Tensor):
                labels = self.hdbscan.fit_predict(all_vec.detach().cpu().numpy())
            else:
                raise err
        if not isinstance(labels, np.ndarray):
            labels = labels.get()
        self._save_labels_if_wanted(labels)
        # TODO adapt for dask
        val_cluster_label = pd.DataFrame.from_dict(
            dict(value=left_emb.names + right_emb.names, cluster_label=labels)
        )
        val_cluster_label = self._handle_noise_cluster(val_cluster_label)
        val_cluster_label["cluster_label"] = (
            self.__class__._CLUSTERING_LABEL_PREFIX
            + val_cluster_label["cluster_label"].astype(str)
        )
        left_conc = self._conc_cluster_labels(left, val_cluster_label)
        right_conc = self._conc_cluster_labels(right, val_cluster_label)
        return left_conc, right_conc


class AttributeClusteringTokenBlocker(TokenClusteringMixin, TokenBlocker):
    def __init__(
        self,
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
        tokenize_fn: Callable[[str], List[str]] = word_tokenize,
        stop_words: Optional[List[str]] = None,
        min_token_length: int = 3,
        save_dir: Optional[str] = None,
        save: bool = False,
    ):
        super().__init__(
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
            min_token_length=min_token_length,
            tokenize_fn=tokenize_fn,
            stop_words=stop_words,
            save_dir=save_dir,
            save=save,
        )

    def assign(
        self,
        left: KlinkerFrame,
        right: KlinkerFrame,
        left_rel: Optional[KlinkerFrame] = None,
        right_rel: Optional[KlinkerFrame] = None,
    ) -> KlinkerBlockManager:
        left_c, right_c = self.embed_and_cluster(left, right)
        return super().assign(left_c, right_c)


class AttributeClusteringMinHashLSHBlocker(TokenClusteringMixin, MinHashLSHBlocker):
    def __init__(
        self,
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
        threshold: float = 0.5,
        num_perm: int = 128,
        weights: Tuple[float, float] = (0.5, 0.5),
        tokenize_fn: Callable[[str], List[str]] = word_tokenize,
        stop_words: Optional[List[str]] = None,
        min_token_length: int = 3,
        save_dir: Optional[str] = None,
        save: bool = False,
    ):
        super().__init__(
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
            threshold=threshold,
            num_perm=num_perm,
            weights=weights,
            min_token_length=min_token_length,
            tokenize_fn=tokenize_fn,
            stop_words=stop_words,
            save_dir=save_dir,
            save=save,
        )

    def assign(
        self,
        left: KlinkerFrame,
        right: KlinkerFrame,
        left_rel: Optional[KlinkerFrame] = None,
        right_rel: Optional[KlinkerFrame] = None,
    ) -> KlinkerBlockManager:
        left_c, right_c = self.embed_and_cluster(left, right)
        return super().assign(left_c, right_c)


class TokenClusteringTokenBlocker(TokenClusteringMixin, TokenBlocker):
    def __init__(
        self,
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
        save_dir: Optional[str] = None,
        save: bool = False,
    ):
        super().__init__(
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
            min_token_length=min_token_length,
            stop_words=stop_words,
            save_dir=save_dir,
            save=save,
        )
        self._inner_token_blocker = TokenBlocker(
            tokenize_fn=tokenize_fn,
            min_token_length=min_token_length,
            stop_words=stop_words,
        )

    def assign(
        self,
        left: KlinkerFrame,
        right: KlinkerFrame,
        left_rel: Optional[KlinkerFrame] = None,
        right_rel: Optional[KlinkerFrame] = None,
    ) -> KlinkerBlockManager:
        left_tok = KlinkerPandasFrame.from_df(
            self._inner_token_blocker._create_exploded_token_frame(
                left.fillna("").concat_values()
            ).rename(columns={left.table_name: "tail"}),
            id_col=left.id_col,
            table_name=left.table_name,
        )
        right_tok = KlinkerPandasFrame.from_df(
            self._inner_token_blocker._create_exploded_token_frame(
                right.fillna("").concat_values()
            ).rename(columns={right.table_name: "tail"}),
            id_col=right.id_col,
            table_name=right.table_name,
        )
        left_c, right_c = self.embed_and_cluster(left_tok, right_tok)
        return super().assign(left_c, right_c)


class TokenClusteringMinHashLSHBlocker(TokenClusteringMixin, MinHashLSHBlocker):
    def __init__(
        self,
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
        threshold: float = 0.5,
        num_perm: int = 128,
        weights: Tuple[float, float] = (0.5, 0.5),
        save_dir: Optional[str] = None,
        save: bool = False,
    ):
        super().__init__(
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
            min_token_length=min_token_length,
            stop_words=stop_words,
            threshold=threshold,
            num_perm=num_perm,
            weights=weights,
            save_dir=save_dir,
            save=save,
        )
        self._inner_token_blocker = TokenBlocker(
            tokenize_fn=tokenize_fn,
            min_token_length=min_token_length,
            stop_words=stop_words,
        )

    def assign(
        self,
        left: KlinkerFrame,
        right: KlinkerFrame,
        left_rel: Optional[KlinkerFrame] = None,
        right_rel: Optional[KlinkerFrame] = None,
    ) -> KlinkerBlockManager:
        left_tok = KlinkerPandasFrame.from_df(
            self._inner_token_blocker._create_exploded_token_frame(
                left.fillna("").concat_values()
            ).rename(columns={left.table_name: "tail"}),
            id_col=left.id_col,
            table_name=left.table_name,
        )
        right_tok = KlinkerPandasFrame.from_df(
            self._inner_token_blocker._create_exploded_token_frame(
                right.fillna("").concat_values()
            ).rename(columns={right.table_name: "tail"}),
            id_col=right.id_col,
            table_name=right.table_name,
        )
        left_c, right_c = self.embed_and_cluster(left_tok, right_tok)
        return super().assign(left_c, right_c)


if __name__ == "__main__":
    from klinker.data import KlinkerDataset
    from sylloge import MovieGraphBenchmark

    ds = KlinkerDataset.from_sylloge(MovieGraphBenchmark(), clean=True).sample(0.1)
    blocks = TokenClusteringMinHashLSHBlocker(
        noise_cluster_handling="token", threshold=0.2, weights=(0.2, 0.8)
    ).assign(ds.left, ds.right)
    from klinker.eval import Evaluation

    print(Evaluation.from_dataset(blocks, ds).to_dict())
