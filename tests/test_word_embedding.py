from typing import Tuple

import numpy as np
import pandas as pd
import pytest
import torch
from mocks import MockGensimDownloader
from strawman import dummy_df

from klinker.encoders import (
    AverageEmbeddingTokenizedFrameEncoder,
    SIFEmbeddingTokenizedFrameEncoder,
)
from klinker.typing import NumpyVectorLiteral, TorchVectorLiteral


@pytest.fixture
def example() -> Tuple[pd.DataFrame, pd.DataFrame]:
    left = dummy_df(
        (10, 1), content_length=20, allowed_chars=" abcdefghijklmnopqrstuvw"
    )
    right = dummy_df(
        (10, 1), content_length=20, allowed_chars=" abcdefghijklmnopqrstuvw"
    )
    return left, right


@pytest.mark.parametrize(
    "cls", [AverageEmbeddingTokenizedFrameEncoder, SIFEmbeddingTokenizedFrameEncoder]
)
@pytest.mark.parametrize("return_type", [NumpyVectorLiteral, TorchVectorLiteral])
def test_word_embedding(cls, return_type, example, mocker):
    dimension = 3
    mocker.patch(
        "klinker.encoders.pretrained.gensim_downloader",
        MockGensimDownloader(dimension=dimension),
    )
    left, right = example
    left_enc, right_enc = cls().encode(left, right, return_type=return_type)
    assert left_enc.shape == (len(left), dimension)
    assert right_enc.shape == (len(right), dimension)
    if return_type == TorchVectorLiteral:
        assert isinstance(left_enc, torch.Tensor)
        assert isinstance(right_enc, torch.Tensor)
    elif return_type == NumpyVectorLiteral:
        assert isinstance(left_enc, np.ndarray)
        assert isinstance(right_enc, np.ndarray)
