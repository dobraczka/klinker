import pandas as pd
import pytest
import torch

from klinker.encoders import FrameEncoder
from klinker.encoders.base import initialize_and_fill
from klinker.data import NamedVector


@pytest.fixture
def example_wrong_column_length() -> pd.DataFrame:
    return pd.DataFrame([["nnn", "nnn"]], columns=[1, 2])


@pytest.fixture
def example_correct_column_length() -> pd.DataFrame:
    return pd.DataFrame([["nnn"]], columns=[1])


@pytest.mark.parametrize(
    "example_left, example_right, should_fail",
    [
        ("example_wrong_column_length", "example_wrong_column_length", True),
        ("example_wrong_column_length", "example_correct_column_length", True),
        ("example_correct_column_length", "example_correct_column_length", False),
    ],
)
def test_validate_wrong(example_left, example_right, should_fail, request):
    left = request.getfixturevalue(example_left)
    right = request.getfixturevalue(example_right)
    if should_fail:
        with pytest.raises(ValueError):
            FrameEncoder().validate(left, right)
    else:
        FrameEncoder().validate(left, right)


def test_initialize_and_fill():
    all_names = ["a", "b", "c", "d", "e"]
    known_vectors = torch.zeros((2, 10))
    known = NamedVector(names=["b", "d"], vectors=known_vectors)
    nv = initialize_and_fill(known=known, all_names=all_names)
    assert (nv[["b", "d"]] == known_vectors).all()
    assert nv.vectors.shape == (len(all_names), 10)
    with pytest.raises(ValueError):
        initialize_and_fill(known=known, all_names=["l","m","n"])
