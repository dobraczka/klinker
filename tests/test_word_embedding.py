from typing import Tuple

import pandas as pd
import pytest
from mocks import MockGensimDownloader
from strawman import dummy_df

from klinker.blockers.embedding.word_embedding import (
    AverageEmbeddingFrameEncoder,
    SIFEmbeddingFrameEncoder,
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
    "cls", [AverageEmbeddingFrameEncoder, SIFEmbeddingFrameEncoder]
)
def test_word_embedding(cls, example, mocker):
    dimension = 3
    mocker.patch(
        "klinker.blockers.embedding.word_embedding.gensim_downloader",
        MockGensimDownloader(dimension=dimension),
    )
    left, right = example
    left_enc, right_enc = cls().encode(left, right)
    assert left_enc.shape == (len(left), dimension)
    assert right_enc.shape == (len(right), dimension)
