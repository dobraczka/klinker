import pathlib
from typing import Optional, Union

from class_resolver import HintOrType, OptionalKwargs

from klinker.encoders.deepblocker import (
    DeepBlockerFrameEncoder,
    deep_blocker_encoder_resolver,
)

from .blockbuilder import EmbeddingBlockBuilder
from .blocker import EmbeddingBlocker


class DeepBlocker(EmbeddingBlocker):
    """Base class for DeepBlocker strategies.

    Args:
        frame_encoder: DeepBlockerFrameEncoder: DeepBlocker strategy.
        frame_encoder_kwargs: keyword arguments for initialisation of encoder
        embedding_block_builder_kwargs: keyword arguments for initalising blockbuilder.
        save: If true saves the embeddings before using blockbuilding.
        save_dir: Directory where to save the embeddings.
        force: If true, recalculate the embeddings and overwrite existing. Else use precalculated if present.

    Attributes:
        frame_encoder: DeepBlocker Encoder class to use for embedding the datasets.
        embedding_block_builder: Block building class to create blocks from embeddings.
        save: If true saves the embeddings before using blockbuilding.
        save_dir: Directory where to save the embeddings.
        force: If true, recalculate the embeddings and overwrite existing. Else use precalculated if present.


    Examples:

        >>> # doctest: +SKIP
        >>> from sylloge import MovieGraphBenchmark
        >>> from klinker.data import KlinkerDataset
        >>> ds = KlinkerDataset.from_sylloge(MovieGraphBenchmark(),clean=True)
        >>> from klinker.blockers import DeepBlocker
        >>> blocker = DeepBlocker(frame_encoder="autoencoder")
        >>> blocks = blocker.assign(left=ds.left, right=ds.right)

    Quote: Reference
        Thirumuruganathan et. al. 'Deep Learning for Blocking in Entity Matching: A Design Space Exploration', VLDB 2021, <http://vldb.org/pvldb/vol14/p2459-thirumuruganathan.pdf>
    """

    def __init__(
        self,
        frame_encoder: HintOrType[DeepBlockerFrameEncoder] = None,
        frame_encoder_kwargs: OptionalKwargs = None,
        embedding_block_builder: HintOrType[EmbeddingBlockBuilder] = None,
        embedding_block_builder_kwargs: OptionalKwargs = None,
        save: bool = True,
        save_dir: Optional[Union[str, pathlib.Path]] = None,
        force: bool = False,
    ):
        frame_encoder = deep_blocker_encoder_resolver.make(
            frame_encoder, frame_encoder_kwargs
        )
        super().__init__(
            frame_encoder=frame_encoder,
            embedding_block_builder=embedding_block_builder,
            embedding_block_builder_kwargs=embedding_block_builder_kwargs,
            save=save,
            save_dir=save_dir,
            force=force,
        )
