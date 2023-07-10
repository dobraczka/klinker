import numpy as np
import pytest
import torch

from klinker.data.named_vector import NamedVector


@pytest.mark.parametrize("as_dict", [False, True])
@pytest.mark.parametrize("change_names", [False, True])
@pytest.mark.parametrize("torchify", [True, False])
def test_named_vector(as_dict, change_names, torchify):
    names = ["a", "b", "c", "d"]
    names_mapping = {name: idx for idx, name in enumerate(names)}
    id_mapping = {idx: name for idx, name in enumerate(names)}
    orig_names = names
    vector = np.random.rand(4, 10)
    if torchify:
        vector = torch.from_numpy(vector)
    if as_dict:
        nv = NamedVector(names=names_mapping, vectors=vector)
    else:
        nv = NamedVector(names=names, vectors=vector)

    # test entity mapping
    assert nv.entity_id_mapping == names_mapping
    assert nv.id_entity_mapping == id_mapping

    if change_names:
        names = ["a", "c", "d", "e"]
        nv.names = names
    assert (nv[names[1]] == vector[1]).all()
    assert (nv[1] == vector[1]).all()
    assert (nv[[names[0], names[1]]] == vector[[0, 1]]).all()
    assert (nv[[0, 1]] == vector[[0, 1]]).all()
    assert (nv[:3] == vector[:3]).all()
    assert nv.names == names
    assert len(nv[[]]) == 0

    with pytest.raises(ValueError):
        # test wrong init
        NamedVector(names=names[:2], vectors=vector)
        NamedVector(names=names, vectors=vector[:3])
        NamedVector(names=[0, 1], vectors=vector[:2])
        NamedVector(names={"a": 0, "b": 2}, vectors=vector[:2])
        # test disallowed item setting
        nv.names = ["a", "a", "a", "a"]
        nv.names = ["a", "b"]
        nv.vectors = vector[:2]
        nv["a"] = "b"

    # reset orignal names
    nv.names = orig_names

    # test setting
    zeros = np.zeros(10)
    two_zeros = np.stack([zeros, zeros])
    all_zeros = np.zeros((4, 10))
    if torchify:
        zeros = torch.from_numpy(zeros)
        two_zeros = torch.from_numpy(two_zeros)
        all_zeros = torch.from_numpy(all_zeros)
    nv["a"] = zeros
    assert (nv["a"] == zeros).all()
    nv[1] = zeros
    assert (nv[1] == zeros).all()
    nv[["b", "c"]] = two_zeros
    assert (nv[["b", "c"]] == two_zeros).all()
    nv[:4] = all_zeros
    assert (nv.vectors == all_zeros).all()

    # test eq
    assert nv == NamedVector(names=nv.names, vectors=nv.vectors)
    assert nv != NamedVector(names=["w", "r", "o", "n"], vectors=nv.vectors)
    assert nv != NamedVector(names=nv.names, vectors=torch.rand(4, 10))

    # smoketest repr
    nv.__repr__()

    # test concat
    new_names = ["x", "y", "z"]
    new_vectors = np.random.rand(3, 10)
    if torchify:
        new_vectors = torch.from_numpy(new_vectors)
    new_nv_sub = NamedVector(names=new_names, vectors=new_vectors)
    new_nv = nv.concat(new_nv_sub)

    assert len(new_nv) == len(names) + len(new_names)
    assert (new_nv[orig_names] == vector).all()
    assert (new_nv[new_names] == new_vectors).all()

    # test subset
    assert new_nv.subset(orig_names) == nv
    assert new_nv.subset(new_names) == new_nv_sub
