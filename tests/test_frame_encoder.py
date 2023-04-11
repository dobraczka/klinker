import pandas as pd
import pytest

from klinker.encoders import FrameEncoder


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
