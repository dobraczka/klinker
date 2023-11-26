import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
import torch
from class_resolver import ClassResolver
from kiez import Kiez
from kiez.hubness_reduction import HubnessReduction
from kiez.neighbors import NNAlgorithm

from ...data import KlinkerBlockManager, NamedVector, NNBasedKlinkerBlockManager
from ...typing import GeneralVector
from ...utils import sparse_sinkhorn_sims_pytorch

try:
    from cuml.cluster import HDBSCAN
except ImportError:
    from hdbscan import HDBSCAN

logger = logging.getLogger(__name__)


class EmbeddingBlockBuilder:
    """Base class for building blocks from embeddings."""

    def build_blocks(
        self,
        left: NamedVector,
        right: NamedVector,
        left_name: str,
        right_name: str,
    ) -> KlinkerBlockManager:
        """Build blocks from given embeddings.

        Args:
          left: NamedVector: Left embeddings.
          right: NamedVector: Right embeddings.
          left_name: str: Name of left dataset.
          right_name: str: Name of right dataset.

        Returns:
            Blocks
        """
        raise NotImplementedError


class NearestNeighborEmbeddingBlockBuilder(EmbeddingBlockBuilder):
    """Build blocks from embeddings by using n-nearest neigbors as blocks."""

    def _get_neighbors(
        self,
        left: GeneralVector,
        right: GeneralVector,
    ) -> np.ndarray:
        """Get nearest neighbors of of left entities in right embeddings.

        Args:
          left: GeneralVector: Left embeddings.
          right: GeneralVector: Right embeddings.

        Returns:
            nearest neighbors
        """
        raise NotImplementedError

    def build_blocks(
        self,
        left: NamedVector,
        right: NamedVector,
        left_name: str,
        right_name: str,
    ) -> KlinkerBlockManager:
        """Build blocks from given embeddings.

        Args:
          left: NamedVector: Left embeddings.
          right: NamedVector: Right embeddings.
          left_name: str: Name of left dataset.
          right_name: str: Name of right dataset.

        Returns:
            Blocks
        """
        start = time.time()
        neighbors = self._get_neighbors(left=left.vectors, right=right.vectors)
        print(f"Neighbors shape: {neighbors.shape}")
        end = time.time()
        print(f"Got neighbors in {end - start}")
        reverse_mapping = np.vectorize(right.id_entity_mapping.get)
        df = pd.DataFrame(reverse_mapping(neighbors), index=left.names)
        # parquet does not like int column names
        df.columns = df.columns.astype(str)
        return NNBasedKlinkerBlockManager.from_pandas(df)


class KiezEmbeddingBlockBuilder(NearestNeighborEmbeddingBlockBuilder):
    """Use kiez for nearest neighbor calculation.

    Args:
        n_neighbors: number k nearest neighbors.
        n_candidates: number candidates, when using hubness reduction.
        algorithm: nearest neighbor algorithm.
        algorithm_kwargs: keyword arguments for initialising nearest neighbor algorithm.
        hubness: hubness reduction method if wanted.
        hubness_kwargs: keyword arguments for initialising hubness reduction.

    Examples:

        >>> import numpy as np
        >>> from klinker.data import NamedVector
        >>> from klinker.blockers.embedding import KiezEmbeddingBlockBuilder
        >>> left = np.random.rand(50,2)
        >>> right = np.random.rand(50,2)
        >>> left_names = [f"left_{i}" for i in range(10)]
        >>> left_names = [f"left_{i}" for i in range(len(left))]
        >>> right_names = [f"right_{i}" for i in range(len(right))]
        >>> left_v = NamedVector(left_names, left)
        >>> right_v = NamedVector(right_names, right)
        >>> emb_bb = KiezEmbeddingBlockBuilder()
        >>> blocks = emb_bb.build_blocks(left_v, right_v, "left", "right") # doctest: +SKIP
        >>> blocks[0].compute() # doctest: +SKIP
                       left                                              right
        0  [left_0]  [right_3, right_24, right_11, right_46, right_37]

    """

    def __init__(
        self,
        n_neighbors: int = 5,
        n_candidates: Optional[int] = None,
        algorithm: Optional[Union[str, NNAlgorithm, Type[NNAlgorithm]]] = None,
        algorithm_kwargs: Optional[Dict[str, Any]] = None,
        hubness: Optional[Union[str, HubnessReduction, Type[HubnessReduction]]] = None,
        hubness_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if n_candidates:
            if "n_candidates" in algorithm_kwargs:
                logger.warn(
                    f"Found n_candidates in algorithm_kwargs as well! Using n_candidates={n_candidates}"
                )
            algorithm_kwargs["n_candidates"] = n_candidates
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
        """Get nearest neighbors of of left entities in right embeddings.

        Args:
          left: GeneralVector: Left embeddings.
          right: GeneralVector: Right embeddings.

        Returns:
            nearest neighbors
        """
        if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
            left = left.detach().cpu().numpy()
            right = right.detach().cpu().numpy()
        self.kiez.fit(left, right)
        neighs = self.kiez.kneighbors(return_distance=False)
        assert isinstance(neighs, np.ndarray)  # for mypy
        return neighs


class SparseSinkhornEmbeddingBlockBuilder(NearestNeighborEmbeddingBlockBuilder):
    def __init__(
        self, n_neighbors=10, n_candidates=50, iteration=10, reg=0.05, device=None
    ):
        if top_k_candidates < n_neighbors:
            logger.warn(
                "n_candidates cannot be smaller than n_neighbors! Setting n_candidates = n_neighbors"
            )
            n_candidates = n_neighbors
        self.n_neighbors = n_neighbors
        self.n_candidates = n_candidates
        self.iteration = iteration
        self.reg = reg
        self.device = device

    def _get_neighbors(
        self,
        left: GeneralVector,
        right: GeneralVector,
    ) -> np.ndarray:
        neighs, _ = sparse_sinkhorn_sims_pytorch(
            left,
            right,
            top_k=self.n_candidates,
            iteration=self.iteration,
            reg=self.reg,
            device=self.device,
        )
        return neighs[:, : self.n_neighbors]


class ClusteringEmbeddingBlockBuilder(EmbeddingBlockBuilder):
    """Use clustering of embeddings for blockbuilding."""

    def _cluster(
        self,
        left: GeneralVector,
        right: GeneralVector,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Cluster embeddings.

        Args:
          left: GeneralVector: left embeddings.
          right: GeneralVector: right embeddings.

        Returns:
            cluster labels of left/right
        """
        raise NotImplementedError

    @staticmethod
    def blocks_side(
        cluster_labels: np.ndarray, names: List[str], data_name: str
    ) -> pd.DataFrame:
        """Create blocks form cluster labels for one side.

        Args:
          cluster_labels: np.ndarray: Cluster labels.
          names: List[str]: Entity names.
          data_name: str: Name of dataset.

        Returns:
            Blocks for one side as pandas DataFrame
        """
        blocked = pd.DataFrame([names, cluster_labels]).transpose().groupby(1).agg(set)
        blocked.columns = [data_name]
        blocked.index.name = "cluster"
        return blocked

    def build_blocks(
        self,
        left: NamedVector,
        right: NamedVector,
        left_name: str,
        right_name: str,
    ) -> pd.DataFrame:
        """Build blocks from given embeddings.

        Args:
          left: NamedVector: Left embeddings.
          right: NamedVector: Right embeddings.
          left_name: str: Name of left dataset.
          right_name: str: Name of right dataset.

        Returns:
            Blocks
        """
        left_cluster_labels, right_cluster_labels = self._cluster(
            left.vectors, right.vectors
        )
        left_blocks = ClusteringEmbeddingBlockBuilder.blocks_side(
            left_cluster_labels, left.names, left_name
        )
        right_blocks = ClusteringEmbeddingBlockBuilder.blocks_side(
            right_cluster_labels, right.names, right_name
        )
        return KlinkerBlockManager.from_pandas(
            left_blocks.join(right_blocks, how="inner")
        )


class HDBSCANEmbeddingBlockBuilder(ClusteringEmbeddingBlockBuilder):
    """Use HDBSCAN clustering for block building.

    For information about parameter selection visit <https://hdbscan.readthedocs.io/en/latest/parameter_selection.html>.

    Args:
        min_cluster_size: int: The minimum size of clusters.
        min_samples: Optional[int]: The number of samples in a neighbourhood for a point to be considered a core point.
        cluster_selection_epsilon: float: A distance threshold. Clusters below this value will be merged.
        metric: str: Distance metric to use.
        alpha: float: A distance scaling parameter as used in robust single linkage.
        p: Optional[float]: p value to use if using the minkowski metric.
        cluster_selection_method: str: The method used to select clusters from the condensed tree.
        kwargs: Arguments passed to the distance metric

    Examples:

        >>> import numpy as np
        >>> from klinker.data import NamedVector
        >>> from klinker.blockers.embedding.blockbuilder import HDBSCANEmbeddingBlockBuilder
        >>> left = np.random.rand(50,2)
        >>> right = np.random.rand(50,2)
        >>> left_names = [f"left_{i}" for i in range(len(left))]
        >>> right_names = [f"right_{i}" for i in range(len(right))]
        >>> left_v = NamedVector(left_names, left)
        >>> right_v = NamedVector(right_names, right)
        >>> emb_bb = HDBSCANEmbeddingBlockBuilder()
        >>> blocks = emb_bb.build_blocks(left_v, right_v, "left", "right")
        >>> blocks[0].compute() #doctest: +SKIP
                                      left                right
        cluster
        0        {left_22, left_3, left_7}  {right_6, right_27}

    """

    def __init__(
        self,
        min_cluster_size: int = 5,
        min_samples: Optional[int] = None,
        cluster_selection_epsilon: float = 0.0,
        metric: str = "euclidean",
        alpha: float = 1.0,
        p: Optional[float] = None,
        cluster_selection_method: str = "eom",
        **kwargs,
    ):
        self.clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            metric=metric,
            alpha=alpha,
            p=p,
            **kwargs,
        )

    def _cluster(
        self,
        left: GeneralVector,
        right: GeneralVector,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Cluster embeddings.

        Args:
          left: GeneralVector: left embeddings.
          right: GeneralVector: right embeddings.

        Returns:
            cluster labels of left/right
        """
        cluster_labels = self.clusterer.fit_predict(np.concatenate([left, right]))
        return cluster_labels[: len(left)], cluster_labels[len(left) :]


block_builder_resolver = ClassResolver(
    [
        KiezEmbeddingBlockBuilder,
        HDBSCANEmbeddingBlockBuilder,
        SparseSinkhornEmbeddingBlockBuilder,
    ],
    base=EmbeddingBlockBuilder,
    default=KiezEmbeddingBlockBuilder,
)
