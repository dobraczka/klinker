from class_resolver import ClassResolver

from .base import FrameEncoder, TokenizedFrameEncoder, RelationFrameEncoder
from .deepblocker import (
    AutoEncoderDeepBlockerFrameEncoder,
    CrossTupleTrainingDeepBlockerFrameEncoder,
    HybridDeepBlockerFrameEncoder,
)
from .light_ea import LightEAFrameEncoder
from .pretrained import (
    AverageEmbeddingTokenizedFrameEncoder,
    SIFEmbeddingTokenizedFrameEncoder,
    TransformerTokenizedFrameEncoder,
)

frame_encoder_resolver = ClassResolver(
    [
        TransformerTokenizedFrameEncoder,
        AverageEmbeddingTokenizedFrameEncoder,
        SIFEmbeddingTokenizedFrameEncoder,
        AutoEncoderDeepBlockerFrameEncoder,
        CrossTupleTrainingDeepBlockerFrameEncoder,
        HybridDeepBlockerFrameEncoder,
        LightEAFrameEncoder,
    ],
    base=FrameEncoder,
    default=SIFEmbeddingTokenizedFrameEncoder,
)

__all__ = [
    # abstract classes
    "FrameEncoder",
    "TokenizedFrameEncoder",
    "RelationFrameEncoder",
    # concrete implementations
    "TransformerTokenizedFrameEncoder",
    "AverageEmbeddingTokenizedFrameEncoder",
    "SIFEmbeddingTokenizedFrameEncoder",
    "AutoEncoderDeepBlockerFrameEncoder",
    "CrossTupleTrainingDeepBlockerFrameEncoder",
    "HybridDeepBlockerFrameEncoder",
    "LightEAFrameEncoder",
    # resolver
    "frame_encoder_resolver",
]
