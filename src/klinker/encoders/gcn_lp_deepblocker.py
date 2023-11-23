import logging

from class_resolver import HintOrType, OptionalKwargs

from .base import TokenizedFrameEncoder
from .deepblocker import DeepBlockerFrameEncoder, deep_blocker_encoder_resolver
from .gcn import GCNFrameEncoder
from .light_ea import LightEAFrameEncoder

logger = logging.getLogger(__name__)


class GCNDeepBlockerFrameEncoder(GCNFrameEncoder):
    def __init__(
        self,
        depth: int = 2,
        edge_weight: float = 1.0,
        self_loop_weight: float = 2.0,
        embedding_dimension: int = 300,
        hidden_dimension: int = 75,
        bias: bool = False,
        use_weight_layers: bool = True,
        aggr: str = "sum",
        inner_encoder: HintOrType[TokenizedFrameEncoder] = None,
        inner_encoder_kwargs: OptionalKwargs = None,
        deepblocker_encoder: HintOrType[DeepBlockerFrameEncoder] = None,
        deepblocker_encoder_kwargs: OptionalKwargs = None,
    ):
        if deepblocker_encoder_kwargs is None:
            deepblocker_encoder_kwargs = {}
        if (
            "frame_encoder" in deepblocker_encoder_kwargs
            or "frame_encoder_kwargs" in deepblocker_encoder_kwargs
        ):
            logger.warn("Cannot use frame_encoder args for DeepBlocker here. Ignoring!")
        assert deepblocker_encoder_kwargs  # for mypy
        deepblocker_encoder_kwargs["frame_encoder"] = inner_encoder
        deepblocker_encoder_kwargs["frame_encoder_kwargs"] = inner_encoder_kwargs
        deepblocker_encoder_kwargs["hidden_dimensions"] = (
            embedding_dimension,
            hidden_dimension,
        )
        db_enc = deep_blocker_encoder_resolver.make(
            deepblocker_encoder, deepblocker_encoder_kwargs
        )
        super().__init__(
            depth=depth,
            edge_weight=edge_weight,
            self_loop_weight=self_loop_weight,
            layer_dims=hidden_dimension,
            bias=bias,
            use_weight_layers=use_weight_layers,
            aggr=aggr,
            attribute_encoder=db_enc,
        )


class LightEADeepBlockerFrameEncoder(LightEAFrameEncoder):
    def __init__(
        self,
        depth: int = 2,
        mini_dim: int = 16,
        embedding_dimension: int = 300,
        hidden_dimension: int = 75,
        inner_encoder: HintOrType[TokenizedFrameEncoder] = None,
        inner_encoder_kwargs: OptionalKwargs = None,
        deepblocker_encoder: HintOrType[DeepBlockerFrameEncoder] = None,
        deepblocker_encoder_kwargs: OptionalKwargs = None,
    ):
        if deepblocker_encoder_kwargs is None:
            deepblocker_encoder_kwargs = {}
        if (
            "frame_encoder" in deepblocker_encoder_kwargs
            or "frame_encoder_kwargs" in deepblocker_encoder_kwargs
        ):
            logger.warn("Cannot use frame_encoder args for DeepBlocker here. Ignoring!")
        assert deepblocker_encoder_kwargs  # for mypy
        deepblocker_encoder_kwargs["frame_encoder"] = inner_encoder
        deepblocker_encoder_kwargs["frame_encoder_kwargs"] = inner_encoder_kwargs
        deepblocker_encoder_kwargs["hidden_dimensions"] = (
            embedding_dimension,
            hidden_dimension,
        )
        db_enc = deep_blocker_encoder_resolver.make(
            deepblocker_encoder, deepblocker_encoder_kwargs
        )
        super().__init__(
            depth=depth,
            mini_dim=mini_dim,
            ent_dim=hidden_dimension,
            attribute_encoder=db_enc,
        )
