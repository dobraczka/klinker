import abc
from typing import List, Optional, Tuple, Union

import pandas as pd

from klinker.data import KlinkerFrame, KlinkerTripleFrame
from klinker.typing import DualColumnSpecifier, SingleOrDualColumnSpecifier


def transform_triple_frames_if_needed(kf: KlinkerFrame) -> KlinkerFrame:
    if isinstance(kf, KlinkerTripleFrame):
        return kf.concat_values()
    return kf


def postprocess(blocks: pd.DataFrame) -> pd.DataFrame:
    # remove blocks with only one entry
    max_number_nans = len(blocks.columns) - 1
    blocks = blocks[~(blocks.isnull().sum(axis=1) == max_number_nans)]
    return blocks.applymap(lambda x: [x] if not isinstance(x, list) else x)


class Blocker(abc.ABC):
    @abc.abstractmethod
    def _assign(self, left: KlinkerFrame, right: KlinkerFrame) -> pd.DataFrame:
        raise NotImplementedError

    def assign(self, left: KlinkerFrame, right: KlinkerFrame) -> pd.DataFrame:
        res = self._assign(left=left, right=right)
        return postprocess(res)


class SchemaAgnosticBlocker(Blocker):
    _actual_wanted_cols: DualColumnSpecifier

    def __init__(
        self,
        wanted_cols: Optional[SingleOrDualColumnSpecifier] = None,
    ) -> None:
        self.wanted_cols = wanted_cols

    def _invalid_cols(
        self,
        kf: KlinkerFrame,
        cols: Optional[SingleOrDualColumnSpecifier] = None,
    ):
        if cols is None:
            cols = self.wanted_cols
        if isinstance(cols, str):
            if cols not in kf.cols:
                return True
        elif isinstance(cols, List):
            if not set(cols) <= set(kf.columns):
                return True
        return False

    def _get_legit_wanted_cols(
        self, left: KlinkerFrame, right: KlinkerFrame
    ) -> DualColumnSpecifier:
        error_msg = f"Wanted column(s) {self.wanted_cols} must be in both tables!"
        if self.wanted_cols is None:
            return (left.non_id_columns, right.non_id_columns)
        elif isinstance(self.wanted_cols, str) or isinstance(self.wanted_cols, List):
            if self._invalid_cols(left.columns) or self._invalid_cols(right.columns):
                raise ValueError(error_msg)
            return (self.wanted_cols, self.wanted_cols)
        elif isinstance(self.wanted_cols, tuple):
            if len(self.wanted_cols) != 2:
                raise ValueError(
                    f"Wanted cols tuple has to have exactly two entries, but was {self.wanted_cols}!"
                )
            if self._invalid_cols(
                left.columns, self.wanted_cols[0]
            ) or self._invalid_cols(right.columns, self.wanted_cols[1]):
                raise ValueError(error_msg)
            # TODO don't know how to make mypy find out the type correctly here
            return self.wanted_cols  # type: ignore
        else:
            raise ValueError(
                f"Unknown format for wanted_cols: {type(self.wanted_cols)}"
            )
        return self.wanted_cols

    def assign(self, left: KlinkerFrame, right: KlinkerFrame) -> pd.DataFrame:
        left = transform_triple_frames_if_needed(left)
        right = transform_triple_frames_if_needed(right)
        self._actual_wanted_cols = self._get_legit_wanted_cols(left=left, right=right)
        return super().assign(left=left, right=right)
