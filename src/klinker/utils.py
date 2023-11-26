import logging
from typing import Callable, List, Literal, TypeVar, overload

import dask.dataframe as dd
import numpy as np
import pandas as pd
import torch
from nltk.tokenize import word_tokenize
from tqdm.auto import tqdm

from .typing import (
    DeviceHint,
    Frame,
    GeneralVector,
    GeneralVectorLiteral,
    NumpyVectorLiteral,
    TorchVectorLiteral,
)

X = TypeVar("X")

logger = logging.getLogger(__name__)


def resolve_device(device: DeviceHint = None) -> torch.device:
    """Resolve a torch.device given a desired device (string).

    Args:
      device: DeviceHint:  (Default value = None)

    Returns:

    """
    # copy-pasted from pykeen
    if device is None or device == "gpu":
        device = "cuda"
    if isinstance(device, str):
        device = torch.device(device)
    if not torch.cuda.is_available() and device.type == "cuda":
        device = torch.device("cpu")
        logger.warning("No cuda devices were available. The model runs on CPU")
    return device


@overload
def concat_frames(frames: List[pd.DataFrame]) -> pd.DataFrame:
    ...


@overload
def concat_frames(frames: List[dd.DataFrame]) -> dd.DataFrame:
    ...


def concat_frames(frames: List[Frame]) -> Frame:
    """Concatenate dask or pandas frames.

    Args:
      frames: List[Frame]: List of dataframes.

    Returns:
        concatenated dataframes
    """
    if isinstance(frames[0], pd.DataFrame):
        return pd.concat(frames)
    return dd.concat(frames)


@overload
def cast_general_vector(
    vector: GeneralVector, return_type: Literal["np"]
) -> np.ndarray:
    ...


@overload
def cast_general_vector(
    vector: GeneralVector,
    return_type: Literal["pt"],
) -> torch.Tensor:
    ...


def cast_general_vector(
    vector: GeneralVector,
    return_type: GeneralVectorLiteral,
) -> GeneralVector:
    """Cast a vector to the desired type

    Args:
      vector: GeneralVector: Vector to cast
      return_type: GeneralVectorLiteral: Wanted return type.

    Returns:
        Vector in desired format

    Examples:

        >>> from klinker.utils import cast_general_vector
        >>> import numpy as np
        >>> arr = np.array([1,2,3])
        >>> cast_general_vector(arr, "pt")
        tensor([1, 2, 3])
        >>> t_arr = cast_general_vector(arr, "pt")
        >>> t_arr
        tensor([1, 2, 3])
        >>> cast_general_vector(t_arr, "np")
        array([1, 2, 3])

    """
    if return_type == TorchVectorLiteral:
        return torch.tensor(vector) if not isinstance(vector, torch.Tensor) else vector
    elif return_type == NumpyVectorLiteral:
        return np.array(vector) if not isinstance(vector, np.ndarray) else vector
    else:
        raise ValueError(f"Unknown return_type: {return_type}!")


def tokenize_row(
    row: pd.Series,
    tokenize_fn: Callable[[str], List[str]] = word_tokenize,
    min_token_length: int = 1,
) -> List:
    """Tokenize rows of series.

    Args:
      row: pd.Series: row with values to tokenize
      tokenize_fn: Callable[[str], List[str]]: Tokenization function
      min_token_length: int: Discard tokens below this value

    Returns:
        List of tokens
    """
    res = []
    for value in row.values:
        res.extend(
            list(
                filter(
                    lambda x: len(x) >= min_token_length,
                    tokenize_fn(str(value)),
                )
            )
        )
    return list(set(res))


@torch.no_grad()
def sparse_sinkhorn_sims_pytorch(
    features_l,
    features_r,
    top_k=500,
    iteration=15,
    reg=0.02,
    device=None,
    use_faiss=False,
):
    import torch_scatter

    device = resolve_device(device)
    features_l = cast_general_vector(features_l, "np")
    features_r = cast_general_vector(features_r, "np")

    if use_faiss:
        import faiss

        faiss.normalize_L2(features_l)
        faiss.normalize_L2(features_r)

        dim, measure = features_l.shape[1], faiss.METRIC_INNER_PRODUCT
        param = "Flat"
        index = faiss.index_factory(dim, param, measure)
        faiss.StandardGpuResources()
        index = faiss.index_cpu_to_all_gpus(index)
        index.train(features_l)
        index.add(features_l)
        sims, index = index.search(features_r, top_k)
    else:
        from kiez import Kiez

        kiez = Kiez(n_neighbors=top_k, algorithm="Sklearnnn")
        kiez.fit(features_l, features_r)
        dist, index = kiez.kneighbors()
        x = -dist
        sims = (x - np.min(x)) / (np.max(x) - np.min(x))
    sims = torch.tensor(sims).to(device)
    index = torch.tensor(index).to(device)

    row_sims = torch.exp(sims.flatten() / reg)

    size = features_l.shape[0]
    row_index = (
        torch.stack(
            [
                torch.arange(size * top_k).to(device) // top_k,
                torch.flatten(index.to(torch.int64)),
                torch.arange(size * top_k).to(device),
            ]
        )
    ).t()
    col_index = row_index[torch.argsort(row_index[:, 1])]
    covert_idx = torch.argsort(col_index[:, 2])
    for _ in tqdm(range(iteration), desc="Sinkhorn Iterations"):
        row_sims = (
            row_sims
            / torch_scatter.scatter_add(row_sims, row_index[:, 0])[row_index[:, 0]]
        )
        col_sims = row_sims[col_index[:, 2]]
        col_sims = (
            col_sims
            / torch_scatter.scatter_add(col_sims, col_index[:, 1])[col_index[:, 1]]
        )
        row_sims = col_sims[covert_idx]

    sims = torch.reshape(row_sims, (-1, top_k))
    ranks = np.argsort(-sims.cpu().numpy(), -1)
    new_index = np.take_along_axis(index.cpu().numpy(), ranks, axis=-1)
    return new_index, sims
