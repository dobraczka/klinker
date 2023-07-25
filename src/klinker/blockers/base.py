import abc
from typing import Optional

import pandas as pd

from ..data import KlinkerBlockManager, KlinkerFrame, KlinkerTriplePandasFrame


def transform_triple_frames_if_needed(kf: KlinkerFrame) -> Optional[KlinkerFrame]:
    if isinstance(kf, KlinkerTriplePandasFrame):
        return kf.concat_values()
    return None


class Blocker(abc.ABC):
    @abc.abstractmethod
    def _assign(
        self,
        left: KlinkerFrame,
        right: KlinkerFrame,
        left_rel: Optional[pd.DataFrame] = None,
        right_rel: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        # TODO we don't need the _assign method anymore the assign method can be abstract
        raise NotImplementedError

    def assign(
        self,
        left: KlinkerFrame,
        right: KlinkerFrame,
        left_rel: Optional[pd.DataFrame] = None,
        right_rel: Optional[pd.DataFrame] = None,
    ) -> KlinkerBlockManager:
        res = self._assign(
            left=left, right=right, left_rel=left_rel, right_rel=right_rel
        )
        return res


class SchemaAgnosticBlocker(Blocker):
    def assign(
        self,
        left: KlinkerFrame,
        right: KlinkerFrame,
        left_rel: Optional[pd.DataFrame] = None,
        right_rel: Optional[pd.DataFrame] = None,
    ) -> KlinkerBlockManager:
        left_reduced, right_reduced = left.concat_values(), right.concat_values()
        return super().assign(
            left=left_reduced,
            right=right_reduced,
            left_rel=left_rel,
            right_rel=right_rel,
        )
