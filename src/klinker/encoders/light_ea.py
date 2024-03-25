from typing import Optional

import numpy as np
import torch
from class_resolver import HintOrType, OptionalKwargs
from sklearn.preprocessing import normalize
from tqdm import trange

from klinker.utils import resolve_device

from ..data import NamedVector
from ..typing import GeneralVector
from .base import RelationFrameEncoder
from .pretrained import TokenizedFrameEncoder, tokenized_frame_encoder_resolver


@torch.no_grad()
def _my_norm(x):
    x /= torch.norm(x, p=2, dim=-1).view(-1, 1) + 1e-8
    return x


@torch.no_grad()
def _get_random_vec(*dims, device):
    random_vec = torch.randn(*dims).to(device)
    return _my_norm(random_vec)


@torch.no_grad()
def _random_projection(x, out_dim, device):
    random_vec = _get_random_vec(x.shape[-1], out_dim, device=device)
    return x @ random_vec


@torch.no_grad()
def _batch_sparse_matmul(
    sparse_tensor, dense_tensor, device, batch_size=32, save_mem=True
):
    if not isinstance(dense_tensor, torch.Tensor):
        dense_tensor = torch.from_numpy(dense_tensor).to(device)
    results = []
    for i in range(dense_tensor.shape[-1] // batch_size + 1):
        temp_result = torch.sparse.mm(
            sparse_tensor, dense_tensor[:, i * batch_size : (i + 1) * batch_size]
        )
        if save_mem:
            temp_result = temp_result.cpu().numpy()
        results.append(temp_result)
    if save_mem:
        return np.concatenate(results, -1)
    else:
        return torch.cat(results, dim=-1)


class LightEAFrameEncoder(RelationFrameEncoder):
    """Use LightEA algorithm to encode frame.

    Args:
    ----
        ent_dim: int: Entity dimensions
        depth: int: Number of hops
        mini_dim:int: Mini batching size
        rel_dim:int: relation embedding dimensions (same as ent_dim if None)
        attribute_encoder: HintOrType[TokenizedFrameEncoder]: Attribute encoder class
        attribute_encoder_kwargs: OptionalKwargs: Keyword arguments for initializing attribute encoder class

    Quote: Reference
        Mao et. al.,"LightEA: A Scalable, Robust, and Interpretable Entity Alignment Framework via Three-view Label Propagation", EMNLP 2022 <https://aclanthology.org/2022.emnlp-main.52.pdf>
    """

    def __init__(
        self,
        ent_dim: int = 256,
        depth: int = 2,
        mini_dim: int = 16,
        rel_dim: Optional[int] = None,
        attribute_encoder: HintOrType[TokenizedFrameEncoder] = None,
        attribute_encoder_kwargs: OptionalKwargs = None,
    ):
        # TODO ent dim is unused!
        self.ent_dim = ent_dim
        self.depth = depth
        self.device = resolve_device()
        self.mini_dim = mini_dim
        self.rel_dim = ent_dim if rel_dim is None else rel_dim
        self.attribute_encoder = tokenized_frame_encoder_resolver.make(
            attribute_encoder, attribute_encoder_kwargs
        )

    def _encode_rel(
        self,
        rel_triples_left: np.ndarray,
        rel_triples_right: np.ndarray,
        ent_features: NamedVector,
    ) -> GeneralVector:
        (
            node_size,
            rel_size,
            ent_tuple,
            triples_idx,
            ent_ent,
            ent_ent_val,
            rel_ent,
            ent_rel,
        ) = self._transform_graph(rel_triples_left, rel_triples_right)
        return self._get_features(
            node_size,
            rel_size,
            ent_tuple,
            triples_idx,
            ent_ent,
            ent_ent_val,
            rel_ent,
            ent_rel,
            ent_features.vectors,
        )

    def _transform_graph(
        self, rel_triples_left: np.ndarray, rel_triples_right: np.ndarray
    ):
        triples = []
        rel_size = 0
        for line in rel_triples_left:
            h, r, t = line
            triples.append([h, t, 2 * r])
            triples.append([t, h, 2 * r + 1])
            rel_size = max(rel_size, 2 * r + 1)
        for line in rel_triples_right:
            h, r, t = line
            triples.append([h, t, 2 * r])
            triples.append([t, h, 2 * r + 1])
            rel_size = max(rel_size, 2 * r + 1)
        triples = np.unique(triples, axis=0)
        node_size, rel_size = np.max(triples) + 1, np.max(triples[:, 2]) + 1  # type: ignore
        ent_tuple, triples_idx = [], []
        ent_ent_s, rel_ent_s, ent_rel_s = {}, set(), set()
        last, index = (-1, -1), -1

        for i in range(node_size):
            ent_ent_s[(i, i)] = 0

        for h, t, r in triples:
            ent_ent_s[(h, h)] += 1
            ent_ent_s[(t, t)] += 1

            if (h, t) != last:
                last = (h, t)
                index += 1
                ent_tuple.append([h, t])
                ent_ent_s[(h, t)] = 0

            triples_idx.append([index, r])
            ent_ent_s[(h, t)] += 1
            rel_ent_s.add((r, h))
            ent_rel_s.add((t, r))

        ent_tuple = np.array(ent_tuple)  # type: ignore
        triples_idx = np.unique(np.array(triples_idx), axis=0)  # type: ignore

        ent_ent = np.unique(np.array(list(ent_ent_s.keys())), axis=0)
        ent_ent_val = np.array([ent_ent_s[(x, y)] for x, y in ent_ent]).astype(
            "float32"
        )
        rel_ent = np.unique(np.array(list(rel_ent_s)), axis=0)
        ent_rel = np.unique(np.array(list(ent_rel_s)), axis=0)
        return (
            node_size,
            rel_size,
            ent_tuple,
            triples_idx,
            ent_ent,
            ent_ent_val,
            rel_ent,
            ent_rel,
        )

    @torch.no_grad()
    def _get_features(
        self,
        node_size,
        rel_size,
        ent_tuple,
        triples_idx,
        ent_ent,
        ent_ent_val,
        rel_ent,
        ent_rel,
        ent_feature,
    ):
        ent_feature = ent_feature.to(self.device)
        rel_feature = torch.zeros((rel_size, ent_feature.shape[-1])).to(self.device)

        ent_ent, ent_rel, rel_ent, ent_ent_val, triples_idx, ent_tuple = map(
            torch.tensor,
            [ent_ent, ent_rel, rel_ent, ent_ent_val, triples_idx, ent_tuple],
        )

        ent_ent = ent_ent.t()
        ent_rel = ent_rel.t()
        rel_ent = rel_ent.t()
        triples_idx = triples_idx.t()
        ent_tuple = ent_tuple.t()

        ent_ent_graph = torch.sparse_coo_tensor(
            indices=ent_ent, values=ent_ent_val, size=(node_size, node_size)
        ).to(self.device)
        rel_ent_graph = torch.sparse_coo_tensor(
            indices=rel_ent,
            values=torch.ones(rel_ent.shape[1]),
            size=(rel_size, node_size),
        ).to(self.device)
        ent_rel_graph = torch.sparse_coo_tensor(
            indices=ent_rel,
            values=torch.ones(ent_rel.shape[1]),
            size=(node_size, rel_size),
        ).to(self.device)

        ent_list, rel_list = [ent_feature], [rel_feature]
        for _ in trange(self.depth):
            new_rel_feature = torch.from_numpy(
                _batch_sparse_matmul(rel_ent_graph, ent_feature, self.device)
            ).to(self.device)
            new_rel_feature = _my_norm(new_rel_feature)

            new_ent_feature = torch.from_numpy(
                _batch_sparse_matmul(ent_ent_graph, ent_feature, self.device)
            ).to(self.device)
            new_ent_feature += torch.from_numpy(
                _batch_sparse_matmul(ent_rel_graph, rel_feature, self.device)
            ).to(self.device)
            new_ent_feature = _my_norm(new_ent_feature)

            ent_feature = new_ent_feature
            rel_feature = new_rel_feature
            ent_list.append(ent_feature)
            rel_list.append(rel_feature)

        ent_feature = torch.cat(ent_list, dim=1)
        rel_feature = torch.cat(rel_list, dim=1)

        ent_feature = _my_norm(ent_feature)
        rel_feature = _my_norm(rel_feature)
        rel_feature = _random_projection(rel_feature, self.rel_dim, self.device)
        batch_size = ent_feature.shape[-1] // self.mini_dim
        sparse_graph = torch.sparse_coo_tensor(
            indices=triples_idx,
            values=torch.ones(triples_idx.shape[1]),
            size=(torch.max(triples_idx).item() + 1, rel_size),
        ).to(self.device)
        adj_value = _batch_sparse_matmul(sparse_graph, rel_feature, self.device)
        del rel_feature

        features_list = []

        for batch in trange(self.rel_dim // batch_size + 1):
            temp_list = []
            for head in trange(batch_size):
                if batch * batch_size + head >= self.rel_dim:
                    break
                sparse_graph = torch.sparse_coo_tensor(
                    indices=ent_tuple,
                    values=adj_value[:, batch * batch_size + head],
                    size=(node_size, node_size),
                ).to(self.device)
                feature = _batch_sparse_matmul(
                    sparse_graph,
                    _random_projection(ent_feature, self.mini_dim, self.device).to(
                        self.device
                    ),
                    self.device,
                    batch_size=128,
                    save_mem=True,
                )
                temp_list.append(feature)
            if len(temp_list):
                features_list.append(np.concatenate(temp_list, axis=-1))
        features = np.concatenate(features_list, axis=-1)
        features = normalize(features)
        return np.concatenate([ent_feature.cpu().numpy(), features], axis=-1)
