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
    SentenceTransformerTokenizedFrameEncoder,
)
from .gcn import GCNFrameEncoder

frame_encoder_resolver = ClassResolver(
    [
        TransformerTokenizedFrameEncoder,
        AverageEmbeddingTokenizedFrameEncoder,
        SIFEmbeddingTokenizedFrameEncoder,
        AutoEncoderDeepBlockerFrameEncoder,
        CrossTupleTrainingDeepBlockerFrameEncoder,
        HybridDeepBlockerFrameEncoder,
        LightEAFrameEncoder,
        GCNFrameEncoder
    ],
    base=FrameEncoder,
    default=SIFEmbeddingTokenizedFrameEncoder,
)

__all__ = [
    # abstract
    "FrameEncoder",
    "TokenizedFrameEncoder",
    "RelationFrameEncoder",
    # concrete
    "TransformerTokenizedFrameEncoder",
    "SentenceTransformerTokenizedFrameEncoder",
    "AverageEmbeddingTokenizedFrameEncoder",
    "SIFEmbeddingTokenizedFrameEncoder",
    "AutoEncoderDeepBlockerFrameEncoder",
    "CrossTupleTrainingDeepBlockerFrameEncoder",
    "HybridDeepBlockerFrameEncoder",
    "LightEAFrameEncoder",
    "GCNFrameEncoder",
    # resolver
    "frame_encoder_resolver",
]
