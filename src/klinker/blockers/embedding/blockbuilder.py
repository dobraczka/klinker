from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, Union

import numpy as np
import pandas as pd
import torch
from kiez import Kiez
from kiez.hubness_reduction import HubnessReduction
from kiez.neighbors import NNAlgorithm

from klinker.data import KlinkerFrame


class EmbeddingBlockBuilder:
    def build_blocks(
        self,
        left: Union[np.ndarray, torch.Tensor],
        right: Union[np.ndarray, torch.Tensor],
        left_data: KlinkerFrame,
        right_data: KlinkerFrame,
    ) -> pd.DataFrame:
        raise NotImplementedError


class NearestNeighborEmbeddingBlockBuilder(EmbeddingBlockBuilder):
    def _get_neighbors(
        self,
        left: Union[np.ndarray, torch.Tensor],
        right: Union[np.ndarray, torch.Tensor],
    ) -> np.ndarray:
        raise NotImplementedError

    def build_blocks(
        self,
        left: Union[np.ndarray, torch.Tensor],
        right: Union[np.ndarray, torch.Tensor],
        left_data: KlinkerFrame,
        right_data: KlinkerFrame,
    ) -> pd.DataFrame:
        neighbors = self._get_neighbors(left=left, right=right)
        df = pd.DataFrame(neighbors)
        df[right_data.name] = df.applymap(
            lambda x, right_data: right_data.iloc[x][right_data.id_col],
            right_data=right_data,
        ).values.tolist()
        df[left_data.name] = left_data.id.values.tolist()
        return df[[left_data.name, right_data.name]]


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
        left: Union[np.ndarray, torch.Tensor],
        right: Union[np.ndarray, torch.Tensor],
    ) -> np.ndarray:
        if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
            left = left.detach().numpy()
            right = right.detach().numpy()
        self.kiez.fit(left, right)
        neighs = self.kiez.kneighbors(return_distance=False)
        assert isinstance(neighs, np.ndarray)  # for mypy
        return neighs
