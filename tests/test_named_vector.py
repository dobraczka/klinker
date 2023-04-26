import numpy as np
import pytest
import torch

from klinker.data.named_vector import NamedVector


@pytest.mark.parametrize("torchify", [True, False])
def test_named_vector(torchify):
    names = ["a", "b", "c", "d"]
    vector = np.random.rand(4, 10)
    if torchify:
        vector = torch.from_numpy(vector)
    nv = NamedVector(names=names, vectors=vector)
    nv[:3]

    assert (nv["b"] == vector[1]).all()
    assert (nv[1] == vector[1]).all()
    assert (nv[["a", "b"]] == vector[[0, 1]]).all()
    assert (nv[[0, 1]] == vector[[0, 1]]).all()
    assert (nv[:3] == vector[:3]).all()
