from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
import torch
from class_resolver import ClassResolver
from kiez import Kiez
from kiez.hubness_reduction import HubnessReduction
from kiez.neighbors import NNAlgorithm

from ...data import KlinkerBlockManager, KlinkerFrame, NamedVector
from ...typing import GeneralVector

try:
    from cuml.cluster import HDBSCAN
except ImportError:
    from hdbscan import HDBSCAN


class EmbeddingBlockBuilder:
    def build_blocks(
        self,
        left: NamedVector,
        right: NamedVector,
        left_name: str,
        right_name: str,
    ) -> KlinkerBlockManager:
        raise NotImplementedError

class NearestNeighborEmbeddingBlockBuilder(EmbeddingBlockBuilder):
    def _get_neighbors(
        self,
        left: GeneralVector,
        right: GeneralVector,
    ) -> np.ndarray:
        raise NotImplementedError

    def build_blocks(
        self,
        left: NamedVector,
        right: NamedVector,
        left_name: str,
        right_name: str,
    ) -> KlinkerBlockManager:
        neighbors = self._get_neighbors(left=left.vectors, right=right.vectors)
        df = pd.DataFrame(neighbors)
        df[right_name] = df.applymap(
            lambda x, right: right.names[x],
            right=right,
        ).values.tolist()
        df[left_name] = left.names
        return KlinkerBlockManager.from_pandas(df[[left_name, right_name]])


class KiezEmbeddingBlockBuilder(NearestNeighborEmbeddingBlockBuilder):
    def __init__(
        self,
        n_neighbors: int = 5,
        algorithm: Optional[Union[str, NNAlgorithm, Type[NNAlgorithm]]] = None,
        algorithm_kwargs: Optional[Dict[str, Any]] = None,
        hubness: Optional[Union[str, HubnessReduction, Type[HubnessReduction]]] = None,
        hubness_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.kiez = Kiez(
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            algorithm_kwargs=algorithm_kwargs,
            hubness=hubness,
            hubness_kwargs=hubness_kwargs,
        )

    def _get_neighbors(
        self,
        left: GeneralVector,
        right: GeneralVector,
    ) -> np.ndarray:
        if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
            left = left.detach().cpu().numpy()
            right = right.detach().cpu().numpy()
        self.kiez.fit(left, right)
        neighs = self.kiez.kneighbors(return_distance=False)
        assert isinstance(neighs, np.ndarray)  # for mypy
        return neighs


class ClusteringEmbeddingBlockBuilder(EmbeddingBlockBuilder):
    def _cluster(
        self,
        left: GeneralVector,
        right: GeneralVector,
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    @staticmethod
    def blocks_side(
        cluster_labels: np.ndarray, names: List[str], data_name: str
    ) -> pd.DataFrame:
        blocked = pd.DataFrame([names, cluster_labels]).transpose().groupby(1).agg(set)
        blocked.columns = [data_name]
        return blocked

    def build_blocks(
        self,
        left: NamedVector,
        right: NamedVector,
        left_name: str,
        right_name: str,
    ) -> pd.DataFrame:
        left_cluster_labels, right_cluster_labels = self._cluster(
            left.vectors, right.vectors
        )
        left_blocks = ClusteringEmbeddingBlockBuilder.blocks_side(
            left_cluster_labels, left.names, left_name
        )
        right_blocks = ClusteringEmbeddingBlockBuilder.blocks_side(
            right_cluster_labels, right.names, right_name
        )
        return KlinkerBlockManager.from_pandas(left_blocks.join(right_blocks))


class HDBSCANEmbeddingBlockBuilder(ClusteringEmbeddingBlockBuilder):
    def __init__(
        self,
        min_cluster_size: int = 5,
        min_samples: int = None,
        cluster_selection_epsilon=0.0,
        metric="euclidean",
        alpha=1.0,
        p=None,
        cluster_selection_method="eom",
        **kwargs
    ):
        self.clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            metric=metric,
            alpha=alpha,
            p=p,
            **kwargs
        )

    def _cluster(
        self,
        left: GeneralVector,
        right: GeneralVector,
    ) -> Tuple[np.ndarray, np.ndarray]:
        cluster_labels = self.clusterer.fit_predict(np.concatenate([left, right]))
        return cluster_labels[: len(left)], cluster_labels[len(left) :]


block_builder_resolver = ClassResolver(
    [KiezEmbeddingBlockBuilder, HDBSCANEmbeddingBlockBuilder],
    base=EmbeddingBlockBuilder,
    default=KiezEmbeddingBlockBuilder,
)
