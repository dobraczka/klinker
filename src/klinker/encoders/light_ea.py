from typing import Optional

import faiss
import numpy as np
import torch
from class_resolver import HintOrType, OptionalKwargs
from pykeen.utils import resolve_device
from tqdm import trange

from .base import RelationFrameEncoder
from .pretrained import TokenizedFrameEncoder, tokenized_frame_encoder_resolver
from ..data import NamedVector
from ..typing import GeneralVector


@torch.no_grad()
def my_norm(x):
    x /= torch.norm(x, p=2, dim=-1).view(-1, 1) + 1e-8
    return x


@torch.no_grad()
def get_random_vec(*dims, device):
    random_vec = torch.randn(*dims).to(device)
    random_vec = my_norm(random_vec)
    return random_vec


@torch.no_grad()
def random_projection(x, out_dim, device):
    random_vec = get_random_vec(x.shape[-1], out_dim, device=device)
    return x @ random_vec


@torch.no_grad()
def batch_sparse_matmul(
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
    def __init__(
        self,
        ent_dim: int = 256,
        depth: int = 2,
        mini_dim: int = 16,
        rel_dim: Optional[int] = None,
        attribute_encoder: HintOrType[TokenizedFrameEncoder] = None,
        attribute_encoder_kwargs: OptionalKwargs = None,
    ):
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
        node_size, rel_size = np.max(triples) + 1, np.max(triples[:, 2]) + 1
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

        ent_tuple = np.array(ent_tuple)
        triples_idx = np.unique(np.array(triples_idx), axis=0)

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
        ent_ent, ent_rel, rel_ent, triples_idx, ent_tuple = map(
            lambda x: x.t(), [ent_ent, ent_rel, rel_ent, triples_idx, ent_tuple]
        )
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
        for i in trange(self.depth):
            new_rel_feature = torch.from_numpy(
                batch_sparse_matmul(rel_ent_graph, ent_feature, self.device)
            ).to(self.device)
            new_rel_feature = my_norm(new_rel_feature)

            new_ent_feature = torch.from_numpy(
                batch_sparse_matmul(ent_ent_graph, ent_feature, self.device)
            ).to(self.device)
            new_ent_feature += torch.from_numpy(
                batch_sparse_matmul(ent_rel_graph, rel_feature, self.device)
            ).to(self.device)
            new_ent_feature = my_norm(new_ent_feature)

            ent_feature = new_ent_feature
            rel_feature = new_rel_feature
            ent_list.append(ent_feature)
            rel_list.append(rel_feature)

        ent_feature = torch.cat(ent_list, dim=1)
        rel_feature = torch.cat(rel_list, dim=1)

        ent_feature = my_norm(ent_feature)
        rel_feature = my_norm(rel_feature)
        rel_feature = random_projection(rel_feature, self.rel_dim, self.device)
        batch_size = ent_feature.shape[-1] // self.mini_dim
        sparse_graph = torch.sparse_coo_tensor(
            indices=triples_idx,
            values=torch.ones(triples_idx.shape[1]),
            size=(torch.max(triples_idx).item() + 1, rel_size),
        ).to(self.device)
        adj_value = batch_sparse_matmul(sparse_graph, rel_feature, self.device)
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
                feature = batch_sparse_matmul(
                    sparse_graph,
                    random_projection(ent_feature, self.mini_dim, self.device).to(
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

        faiss.normalize_L2(features)
        features = np.concatenate([ent_feature.cpu().numpy(), features], axis=-1)
        return features


if __name__ == "__main__":
    from sylloge import OpenEA

    from klinker import KlinkerDataset
    from klinker.blockers import EmbeddingBlocker
    from klinker.blockers.embedding import KiezEmbeddingBlockBuilder
    from klinker.eval_metrics import Evaluation

    sylloge_ds = OpenEA()
    ds = KlinkerDataset.from_sylloge(sylloge_ds, clean=True).sample(200)

    kiez = KiezEmbeddingBlockBuilder(
        n_neighbors=100,
        algorithm="Faiss",
        algorithm_kwargs=dict(index_key="HNSW32", index_param="efSearch=512"),
    )
    blocker = EmbeddingBlocker(
        frame_encoder=LightEAFrameEncoder(ent_dim=256, depth=2, mini_dim=16, attribute_encoder="TransformerTokenizedFrameEncoder"),
        embedding_block_builder=kiez,
    )

    blocks = blocker.assign(
        left=ds.left, right=ds.right, left_rel=ds.left_rel, right_rel=ds.right_rel
    )
    ev = Evaluation.from_dataset(blocks=blocks, dataset=ds)
    print(ev)
    # id_mapped = IdMappedEADataset.from_ea_dataset(ds)
    # inverse_mapping = {idx: uri for uri, idx in id_mapped.entity_mapping.items()}
    # left_ids = sorted(
    #     set(id_mapped.rel_triples_left[:, 0]).union(
    #         set(id_mapped.rel_triples_left[:, 2])
    #     )
    # )
    # left_data = KlinkerFrame(
    #     [inverse_mapping[idx] for idx in left_ids],
    #     index=left_ids,
    #     name="left",
    #     columns=["id"],
    # )
    # right_ids = sorted(
    #     set(id_mapped.rel_triples_right[:, 0]).union(
    #         set(id_mapped.rel_triples_right[:, 2])
    #     )
    # )
    # right_data = KlinkerFrame(
    #     [inverse_mapping[idx] for idx in right_ids],
    #     index=right_ids,
    #     name="right",
    #     columns=["id"],
    # )

    # ent_dim = 256
    # depth = 2
    # mini_dim = 16
    # (
    #     node_size,
    #     rel_size,
    #     ent_tuple,
    #     triples_idx,
    #     ent_ent,
    #     ent_ent_val,
    #     rel_ent,
    #     ent_rel,
    # ) = transform_graph(id_mapped)
    # features = get_features(
    #     id_mapped.folds[0].train,
    #     node_size,
    #     rel_size,
    #     ent_tuple,
    #     triples_idx,
    #     ent_ent,
    #     ent_ent_val,
    #     rel_ent,
    #     ent_rel,
    #     ent_dim,
    #     depth,
    #     "cpu",
    #     mini_dim,
    # )
    # left, right = features[:15000], features[15000:]
    # kiez = KiezEmbeddingBlockBuilder(
    #     n_neighbors=100,
    #     algorithm="Faiss",
    #     algorithm_kwargs=dict(index_key="HNSW32", index_param="efSearch=512"),
    # )
    # blocks = kiez.build_blocks(left, right, left_data, right_data)
    # import ipdb  # noqa: autoimport

    # ipdb.set_trace()  # BREAKPOINT
    # ev = Evaluation(
    #     blocks,
    #     ds.ent_links,
    #     left_data_len=len(left_data),
    #     right_data_len=len(right_data),
    # )
    # print(ev)
