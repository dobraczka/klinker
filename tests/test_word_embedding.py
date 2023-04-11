from typing import Tuple

import pandas as pd
import pytest
from mocks import MockGensimDownloader
from strawman import dummy_df

from klinker.encoders import (
    AverageEmbeddingTokenizedFrameEncoder,
    SIFEmbeddingTokenizedFrameEncoder,
)


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
def test_word_embedding(cls, example, mocker):
    dimension = 3
    mocker.patch(
        "klinker.encoders.pretrained.gensim_downloader",
        MockGensimDownloader(dimension=dimension),
    )
    left, right = example
    left_enc, right_enc = cls().encode(left, right)
    assert left_enc.shape == (len(left), dimension)
    assert right_enc.shape == (len(right), dimension)
