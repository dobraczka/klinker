import logging
import os
import pathlib
import time
import warnings
from typing import Literal, Optional, Tuple, Union, get_args

from class_resolver import HintOrType, OptionalKwargs

from klinker.data import (
    KlinkerBlockManager,
    KlinkerFrame,
    NamedVector,
)
from klinker.encoders import FrameEncoder, frame_encoder_resolver
from klinker.typing import SeriesType

from ...data import generic_upgrade_from_series
from ..base import SchemaAgnosticBlocker
from .blockbuilder import EmbeddingBlockBuilder, block_builder_resolver

ENC_PREFIX = Literal["left_", "right_"]
ENC_SUFFIX = "_enc.pkl"

logger = logging.getLogger("klinker")


class EmbeddingBlocker(SchemaAgnosticBlocker):
    """Base class for embedding-based blocking approaches.

    Args:
    ----
        frame_encoder: Encoder class to use for embedding the datasets.
        frame_encoder_kwargs: keyword arguments for initialising encoder class.
        embedding_block_builder: Block building class to create blocks from embeddings.
        embedding_block_builder_kwargs: keyword arguments for initalising blockbuilder.
        save: If true saves the embeddings before using blockbuilding.
        save_dir: Directory where to save the embeddings.
        force: If true, recalculate the embeddings and overwrite existing. Else use precalculated if present.

    Attributes:
    ----------
        frame_encoder: Encoder class to use for embedding the datasets.
        embedding_block_builder: Block building class to create blocks from embeddings.
        save: If true saves the embeddings before using blockbuilding.
        save_dir: Directory where to save the embeddings.
        force: If true, recalculate the embeddings and overwrite existing. Else use precalculated if present.
    """

    def __init__(
        self,
        frame_encoder: HintOrType[FrameEncoder] = None,
        frame_encoder_kwargs: OptionalKwargs = None,
        embedding_block_builder: HintOrType[EmbeddingBlockBuilder] = None,
        embedding_block_builder_kwargs: OptionalKwargs = None,
        save: bool = True,
        save_dir: Optional[Union[str, pathlib.Path]] = None,
        force: bool = False,
    ):
        self.frame_encoder = frame_encoder_resolver.make(
            frame_encoder, frame_encoder_kwargs
        )
        self.embedding_block_builder = block_builder_resolver.make(
            embedding_block_builder, embedding_block_builder_kwargs
        )
        self.save = save
        self.save_dir = save_dir
        self.force = force

    def _handle_encode(
        self,
        left: SeriesType,
        right: SeriesType,
        left_rel: Optional[KlinkerFrame] = None,
        right_rel: Optional[KlinkerFrame] = None,
    ) -> Tuple[NamedVector, NamedVector]:
        print("self.save=%s" % (self.save))
        left_emb = None
        right_emb = None
        # handle save dir
        if self.save:
            if self.save_dir is None:
                save_dir = pathlib.Path(".").joinpath(
                    f"{left.table_name}_{right.table_name}_{self.frame_encoder.__class__.__name__}"
                )
                self.save_dir = save_dir
            # check if loadable
            if os.path.exists(self.save_dir):
                left_path, left_name = self._encoding_path_and_table_name_from_dir(
                    "left_", left.table_name
                )
                right_path, right_name = self._encoding_path_and_table_name_from_dir(
                    "right_", right.table_name
                )
                if left_path is not None and right_path is not None:
                    if self.force:
                        warnings.warn(
                            f"{self.save_dir} exists. Overwriting! This behaviour can be changed by setting `force=False`"
                        )
                        os.makedirs(self.save_dir, exist_ok=True)
                    else:
                        logger.info(
                            f"Loading existing encodings from {left_path} and {right_path}. To recalculate set `force=True`"
                        )
                        left_emb, right_emb = self.from_encoded(
                            left_path=left_path,
                            left_name=left_name,
                            right_path=right_path,
                            right_name=right_name,
                        )
        if left_emb is None and right_emb is None:
            # else encode
            left_emb, right_emb = self.frame_encoder.encode(
                left=left,
                right=right,
                left_rel=left_rel,
                right_rel=right_rel,
            )
            if self.save:
                assert self.save_dir  # for mypy
                assert left.table_name
                assert right.table_name
                EmbeddingBlocker.save_encoded(
                    self.save_dir,
                    (left_emb, right_emb),
                    (left.table_name, right.table_name),
                )
        assert left_emb
        assert right_emb
        return left_emb, right_emb

    def _assign(
        self,
        left: SeriesType,
        right: SeriesType,
        left_rel: Optional[KlinkerFrame] = None,
        right_rel: Optional[KlinkerFrame] = None,
    ) -> KlinkerBlockManager:
        """Args:
        ----
          left: SeriesType:
          right: SeriesType:
          left_rel: Optional[KlinkerFrame]:  (Default value = None)
          right_rel: Optional[KlinkerFrame]:  (Default value = None)

        Returns
        -------

        """
        left = generic_upgrade_from_series(left, reset_index=False)
        right = generic_upgrade_from_series(right, reset_index=False)
        left_emb, right_emb = self._handle_encode(left, right, left_rel, right_rel)
        assert left.table_name
        assert right.table_name
        return self.embedding_block_builder.build_blocks(
            left=left_emb,
            right=right_emb,
            left_name=left.table_name,
            right_name=right.table_name,
        )

    @staticmethod
    def save_encoded(
        save_dir: Union[str, pathlib.Path],
        encodings: Tuple[NamedVector, NamedVector],
        table_names: Tuple[str, str],
    ):
        """Save embeddings.

        Args:
        ----
          save_dir: Union[str, pathlib.Path]: Directory to save into.
          encodings: Tuple[NamedVector, NamedVector]: Tuple of named embeddings.
          table_names: Tuple[str, str]: Name of left/right dataset.

        """
        if isinstance(save_dir, str):
            save_dir = pathlib.Path(save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for enc, table_name, left_right in zip(
            encodings, table_names, get_args(ENC_PREFIX)
        ):
            path = save_dir.joinpath(f"{left_right}{table_name}{ENC_SUFFIX}")
            logger.info(f"Saved encoding in {path}")
            enc.to_pickle(path)

    def _encoding_path_and_table_name_from_dir(
        self, left_or_right: ENC_PREFIX, table_name: Optional[str] = None
    ) -> Tuple[Optional[pathlib.Path], Optional[str]]:
        assert self.save_dir  # for mypy
        if isinstance(self.save_dir, str):
            self.save_dir = pathlib.Path(self.save_dir)

        if table_name is not None:
            possible_path = self.save_dir.joinpath(
                f"{left_or_right}{table_name}{ENC_SUFFIX}"
            )
            if os.path.exists(possible_path):
                return possible_path, table_name
            return None, None

        enc_path_list = list(self.save_dir.glob(f"{left_or_right}*{ENC_SUFFIX}"))
        if len(enc_path_list) > 1:
            warnings.warn(
                f"Found multiple encodings {enc_path_list} will choose the first"
            )
        elif len(enc_path_list) == 0:
            raise FileNotFoundError(
                f"Expected to find encoding pickle in {self.save_dir} for {left_or_right} side!"
            )

        enc_path = enc_path_list[0]
        table_name = (
            str(enc_path.name).replace(f"{left_or_right}", "").replace(ENC_SUFFIX, "")
        )
        return enc_path, table_name

    def from_encoded(
        self,
        left_path=None,
        right_path=None,
        left_name=None,
        right_name=None,
    ) -> Tuple[NamedVector, NamedVector]:
        start = time.time()
        if self.save_dir is None:
            raise ValueError("Cannot run `from_encoded` if `self.save_dir` is None!")
        if left_path is None:
            left_path, left_name = self._encoding_path_and_table_name_from_dir("left_")
            right_path, right_name = self._encoding_path_and_table_name_from_dir(
                "right_"
            )

        left_enc = NamedVector.from_pickle(left_path)
        right_enc = NamedVector.from_pickle(right_path)
        end = time.time()
        self._loading_time = end - start
        return left_enc, right_enc
