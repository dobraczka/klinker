import logging
import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
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


def _add_remaining_self_loops(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_attr: Optional[torch.Tensor] = None,
    fill_value: Optional[Union[float, torch.Tensor, str]] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    # adapted from https://github.com/pyg-team/pytorch_geometric/blob/2463371cf290a106e057c0c1f24f7a5a38318328/torch_geometric/utils/loop.py#L218
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
    fill_value=2.0,
    add_self_loops=True,
    flow="source_to_target",
    dtype=None,
):
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


class BasicMessagePassing:
    def __init__(
        self,
        edge_weight: float = 1.0,
        self_loop_weight: float = 2.0,
        aggr: str = "add",
    ):
        self.edge_weight = edge_weight
        self.self_loop_weight = self_loop_weight
        self.aggr = aggr

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        edge_index_with_loops, edge_weights = _gcn_norm(
            edge_index,
            num_nodes=len(x),
            edge_weight=torch.tensor([self.edge_weight] * len(edge_index[0])),
            fill_value=self.self_loop_weight,
        )
        return sparse_matmul(
            SparseTensor.from_edge_index(edge_index_with_loops, edge_attr=edge_weights),
            x,
            reduce=self.aggr,
        )


def _glorot(value: torch.Tensor):
    # see https://github.com/pyg-team/pytorch_geometric/blob/3e55a4c263f04ed6676618226f9a0aaf406d99b9/torch_geometric/nn/inits.py#L30
    stdv = math.sqrt(6.0 / (value.size(-2) + value.size(-1)))
    value.data.uniform_(-stdv, stdv)


class FrozenGCNConv(BasicMessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = False,
        edge_weight: float = 1.0,
        self_loop_weight: float = 2.0,
        aggr: str = "add",
    ):
        super().__init__(
            edge_weight=edge_weight, self_loop_weight=self_loop_weight, aggr=aggr
        )
        self.lin = nn.Linear(in_channels, out_channels, bias=bias)
        for param in self.lin.parameters():
            param.requires_grad = False
        # Use glorot initialization
        _glorot(self.lin.weight)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.lin(x)
        return super().forward(x, edge_index)


class GCNFrameEncoder(RelationFrameEncoder):
    """Use untrained GCN for aggregating neighboring embeddings with self.

    Args:
        depth: How many hops of neighbors should be incorporated
        edge_weight: Weighting of non-self-loops
        self_loop_weight: Weighting of self-loops
        layer_dims: Dimensionality of layers if used
        bias: Whether to use bias in layers
        use_weight_layers: Whether to use randomly initialized layers in aggregation
        aggr: Which aggregation to use. Can be :obj:`"sum"`, :obj:`"mean"`, :obj:`"min"` or :obj:`"max"`
        attribute_encoder: HintOrType[TokenizedFrameEncoder]: Base encoder class
        attribute_encoder_kwargs: OptionalKwargs: Keyword arguments for initializing encoder
    """

    def __init__(
        self,
        depth: int = 2,
        edge_weight: float = 1.0,
        self_loop_weight: float = 2.0,
        layer_dims: int = 300,
        bias: bool = False,
        use_weight_layers: bool = True,
        aggr: str = "sum",
        attribute_encoder: HintOrType[TokenizedFrameEncoder] = None,
        attribute_encoder_kwargs: OptionalKwargs = None,
    ):
        if not TORCH_SCATTER:
            logger.error("Could not find torch_scatter and/or torch_sparse package!")
        self.depth = depth
        self.edge_weight = edge_weight
        self.self_loop_weight = self_loop_weight
        self.device = resolve_device()
        self.attribute_encoder = tokenized_frame_encoder_resolver.make(
            attribute_encoder, attribute_encoder_kwargs
        )
        layers: List[BasicMessagePassing]
        if use_weight_layers:
            layers = [
                FrozenGCNConv(
                    in_channels=layer_dims,
                    out_channels=layer_dims,
                    edge_weight=edge_weight,
                    self_loop_weight=self_loop_weight,
                    aggr=aggr,
                )
                for _ in range(self.depth)
            ]
        else:
            layers = [
                BasicMessagePassing(
                    edge_weight=edge_weight,
                    self_loop_weight=self_loop_weight,
                    aggr=aggr,
                )
                for _ in range(self.depth)
            ]
        self.layers = layers

    def _encode_rel(
        self,
        rel_triples_left: np.ndarray,
        rel_triples_right: np.ndarray,
        ent_features: NamedVector,
    ) -> GeneralVector:
        full_graph = np.concatenate([rel_triples_left, rel_triples_right])
        edge_index = torch.from_numpy(full_graph[:, [0, 2]]).t()
        x = ent_features.vectors
        for layer in self.layers:
            x = layer.forward(x, edge_index)
        return x
