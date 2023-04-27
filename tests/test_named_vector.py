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

    for change_names in [True, False]:
        if change_names:
            names = ["a","c","d","e"]
            nv.names = ["a","c","d","e"]
        assert (nv[names[1]] == vector[1]).all()
        assert (nv[1] == vector[1]).all()
        assert (nv[[names[0], names[1]]] == vector[[0, 1]]).all()
        assert (nv[[0, 1]] == vector[[0, 1]]).all()
        assert (nv[:3] == vector[:3]).all()
        assert nv.names == names

    with pytest.raises(ValueError):
        NamedVector(names=names[:2], vectors=vector)
        NamedVector(names=names, vectors=vector[:3])
        nv.names = ["a","a","a","a"]
        nv.names = ["a","b"]
        nv.vectors = vector[:2]
