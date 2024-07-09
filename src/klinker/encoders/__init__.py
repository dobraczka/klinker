from class_resolver import ClassResolver

from .base import FrameEncoder, RelationFrameEncoder, TokenizedFrameEncoder
from .deepblocker import (
    AutoEncoderDeepBlockerFrameEncoder,
    CrossTupleTrainingDeepBlockerFrameEncoder,
    HybridDeepBlockerFrameEncoder,
)
from .gcn import GCNFrameEncoder
from .gcn_lp_deepblocker import (
    GCNDeepBlockerFrameEncoder,
    LightEADeepBlockerFrameEncoder,
)
from .light_ea import LightEAFrameEncoder
from .pretrained import (
    AverageEmbeddingTokenizedFrameEncoder,
    SentenceTransformerTokenizedFrameEncoder,
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
        GCNFrameEncoder,
        SentenceTransformerTokenizedFrameEncoder,
        GCNDeepBlockerFrameEncoder,
        LightEADeepBlockerFrameEncoder,
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
    "GCNDeepBlockerFrameEncoder",
    "LightEADeepBlockerFrameEncoder",
    # resolver
    "frame_encoder_resolver",
]
