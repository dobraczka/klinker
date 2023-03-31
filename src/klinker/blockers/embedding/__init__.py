from class_resolver import ClassResolver

from .blockbuilder import (
    EmbeddingBlockBuilder,
    KiezEmbeddingBlockBuilder,
    NearestNeighborEmbeddingBlockBuilder,
)
from .encoder import FrameEncoder, TransformerTextFrameEncoder

__all__ = [
    "EmbeddingBlockBuilder",
    "NearestNeighborEmbeddingBlockBuilder",
    "KiezEmbeddingBlockBuilder",
    "FrameEncoder",
    "TransformerTextFrameEncoder",
]

block_builder_resolver = ClassResolver(
    [KiezEmbeddingBlockBuilder],
    base=EmbeddingBlockBuilder,
    default=KiezEmbeddingBlockBuilder,
)

frame_encoder_resolver = ClassResolver(
    [TransformerTextFrameEncoder],
    base=FrameEncoder,
    default=TransformerTextFrameEncoder,
)
