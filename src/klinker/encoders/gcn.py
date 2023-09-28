import logging
from typing import List, Optional, Set, Tuple, Union

import numpy as np
import torch
from class_resolver import HintOrType, OptionalKwargs

try:
    from torch_scatter import scatter
    from torch_sparse import SparseTensor
    from torch_sparse import matmul as sparse_matmul

    TORCH_SCATTER = True
except ImportError:
    TORCH_SCATTER = False


from klinker.utils import resolve_device

from .base import RelationFrameEncoder
from .pretrained import TokenizedFrameEncoder, tokenized_frame_encoder_resolver
from ..data import NamedVector
from ..typing import GeneralVector

logger = logging.getLogger(__name__)

# adapted from https://github.com/pyg-team/pytorch_geometric/blob/2463371cf290a106e057c0c1f24f7a5a38318328/torch_geometric/utils/loop.py#L218
def _add_remaining_self_loops(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_attr: Optional[torch.Tensor] = None,
    fill_value: Optional[Union[float, torch.Tensor, str]] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    N = num_nodes
    mask = edge_index[0] != edge_index[1]

    loop_index = torch.arange(0, N, dtype=torch.long, device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)

    if edge_attr is not None:
        if fill_value is None:
            loop_attr = edge_attr.new_full((N,) + edge_attr.size()[1:], 1.0)

        elif isinstance(fill_value, (int, float)):
            loop_attr = edge_attr.new_full((N,) + edge_attr.size()[1:], fill_value)
        elif isinstance(fill_value, torch.Tensor):
            loop_attr = fill_value.to(edge_attr.device, edge_attr.dtype)
            if edge_attr.dim() != loop_attr.dim():
                loop_attr = loop_attr.unsqueeze(0)
            sizes = [N] + [1] * (loop_attr.dim() - 1)
            loop_attr = loop_attr.repeat(*sizes)

        elif isinstance(fill_value, str):
            loop_attr = scatter(
                edge_attr, edge_index[1], dim=0, dim_size=N, reduce=fill_value
            )
        else:
            raise AttributeError("No valid 'fill_value' provided")

        inv_mask = ~mask
        loop_attr[edge_index[0][inv_mask]] = edge_attr[inv_mask]

        edge_attr = torch.cat([edge_attr[mask], loop_attr], dim=0)

    edge_index = torch.cat([edge_index[:, mask], loop_index], dim=1)
    return edge_index, edge_attr


# adapted from https://github.com/pyg-team/pytorch_geometric/blob/2463371cf290a106e057c0c1f24f7a5a38318328/torch_geometric/nn/conv/gcn_conv.py
def _gcn_norm(
    edge_index,
    num_nodes: int,
    edge_weight=None,
    improved=True,
    add_self_loops=True,
    flow="source_to_target",
    dtype=None,
):
    fill_value = 2.0 if improved else 1.0
    assert flow in ["source_to_target", "target_to_source"]

    if edge_weight is None:
        edge_weight = torch.ones(
            (edge_index.size(1),), dtype=dtype, device=edge_index.device
        )

    if add_self_loops:
        edge_index, tmp_edge_weight = _add_remaining_self_loops(
            edge_index=edge_index,
            num_nodes=num_nodes,
            edge_attr=edge_weight,
            fill_value=fill_value,
        )
        assert tmp_edge_weight is not None
        edge_weight = tmp_edge_weight

    row, col = edge_index[0], edge_index[1]
    idx = col if flow == "source_to_target" else row
    deg = scatter(edge_weight, idx, dim=0, dim_size=num_nodes, reduce="sum")
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    return edge_index, edge_weight


class GCNFrameEncoder(RelationFrameEncoder):
    """Use untrained GCN for aggregating neighboring embeddings with self.

    Args:
        depth: How many hops of neighbors should be incorporated
        attribute_encoder: HintOrType[TokenizedFrameEncoder]: Base encoder class
        attribute_encoder_kwargs: OptionalKwargs: Keyword arguments for initializing encoder
    """

    def __init__(
        self,
        depth: int = 2,
        attribute_encoder: HintOrType[TokenizedFrameEncoder] = None,
        attribute_encoder_kwargs: OptionalKwargs = None,
    ):
        if not TORCH_SCATTER:
            logger.error("Could not find torch_scatter and/or torch_sparse package!")
        self.depth = depth
        self.device = resolve_device()
        self.attribute_encoder = tokenized_frame_encoder_resolver.make(
            attribute_encoder, attribute_encoder_kwargs
        )

    def _forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        edge_index_with_loops, edge_weights = _gcn_norm(edge_index, num_nodes=len(x))
        return sparse_matmul(
            SparseTensor.from_edge_index(edge_index_with_loops, edge_attr=edge_weights),
            x,
        )

    def _encode_rel(
        self,
        rel_triples_left: np.ndarray,
        rel_triples_right: np.ndarray,
        ent_features: NamedVector,
    ) -> GeneralVector:
        full_graph = np.concatenate([rel_triples_left, rel_triples_right])
        edge_index = torch.from_numpy(full_graph[:, [0, 2]]).t()
        x = ent_features.vectors
        for _ in range(self.depth):
            x = self._forward(x, edge_index)
        return x
