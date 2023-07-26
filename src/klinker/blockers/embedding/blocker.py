import logging
import os
import pathlib
import warnings
from typing import Literal, Optional, Tuple, Union, get_args

import pandas as pd
from class_resolver import HintOrType, OptionalKwargs

from klinker.data import (
    KlinkerBlockManager,
    KlinkerFrame,
    KlinkerPandasFrame,
    NamedVector,
)

from .blockbuilder import EmbeddingBlockBuilder, block_builder_resolver
from ..base import SchemaAgnosticBlocker
from ...encoders import FrameEncoder, frame_encoder_resolver

ENC_PREFIX = Literal["left_", "right_"]
ENC_SUFFIX = "_enc.pkl"

logger = logging.getLogger("klinker")


class EmbeddingBlocker(SchemaAgnosticBlocker):
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

    def _check_string_ids(self, id_col: pd.Series):
        if not id_col.apply(lambda x: isinstance(x, str)).all():
            raise ValueError("Ids must be string!")

    def _check_ids(self, left: KlinkerFrame, right: KlinkerFrame):
        left_ids_col = left[left.id_col]
        right_ids_col = right[right.id_col]
        self._check_string_ids(left_ids_col)
        self._check_string_ids(right_ids_col)
        left_ids = set(left_ids_col)
        right_ids = set(right_ids_col)
        intersected_ids = left_ids.intersection(right_ids)
        if len(intersected_ids) > 0:
            warnings.warn(
                f"Left and right ids are not disjunct! This may be unintentional and lead to problems. Found {len(intersected_ids)} common ids across {len(left)} left ids and {len(right)} right ids."
            )

    def _assign(
        self,
        left: KlinkerFrame,
        right: KlinkerFrame,
        left_rel: Optional[pd.DataFrame] = None,
        right_rel: Optional[pd.DataFrame] = None,
    ) -> KlinkerBlockManager:
        assert left.table_name is not None
        assert right.table_name is not None
        if isinstance(left, KlinkerPandasFrame):
            self._check_ids(left, right)

        # handle save dir
        if self.save:
            if self.save_dir is None:
                save_dir = pathlib.Path(".").joinpath(
                    "{left.table_name}_{right.table_name}_{self.frame_encoder.__class__.__name__}"
                )
                self.save_dir = save_dir
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
                        return self.from_encoded(
                            left_path=left_path,
                            left_name=left_name,
                            right_path=right_path,
                            right_name=right_name,
                        )

        left_reduced = left.set_index(left.id_col)[left.non_id_columns]
        right_reduced = right.set_index(right.id_col)[right.non_id_columns]
        # TODO fix typing issue
        left_emb, right_emb = self.frame_encoder.encode(
            left=left_reduced,
            right=right_reduced,
            left_rel=left_rel,
            right_rel=right_rel,
        )  # type: ignore
        if self.save:
            assert self.save_dir  # for mypy
            EmbeddingBlocker.save_encoded(
                self.save_dir,
                (left_emb, right_emb),
                (left.table_name, right.table_name),
            )
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
    ) -> KlinkerBlockManager:
        if self.save_dir is None:
            raise ValueError("Cannot run `from_encoded` if `self.save_dir` is None!")
        if left_path is None:
            left_path, left_name = self._encoding_path_and_table_name_from_dir("left_")
            right_path, right_name = self._encoding_path_and_table_name_from_dir(
                "right_"
            )

        left_enc = NamedVector.from_pickle(left_path)
        right_enc = NamedVector.from_pickle(right_path)
        return self.embedding_block_builder.build_blocks(
            left=left_enc,
            right=right_enc,
            left_name=left_name,
            right_name=right_name,
        )
