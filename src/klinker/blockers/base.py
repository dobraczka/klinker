import abc
from typing import Optional

import pandas as pd

from ..data import KlinkerBlockManager, KlinkerFrame, KlinkerTriplePandasFrame

class Blocker(abc.ABC):
    @abc.abstractmethod
    def assign(
        self,
        left: KlinkerFrame,
        right: KlinkerFrame,
        left_rel: Optional[KlinkerFrame] = None,
        right_rel: Optional[KlinkerFrame] = None,
    ) -> KlinkerBlockManager:
        pass


class SchemaAgnosticBlocker(Blocker):
    @abc.abstractmethod
    def _assign(
        self,
        left: KlinkerFrame,
        right: KlinkerFrame,
        left_rel: Optional[KlinkerFrame] = None,
        right_rel: Optional[KlinkerFrame] = None,
    ) -> KlinkerBlockManager:
        pass

    def assign(
        self,
        left: KlinkerFrame,
        right: KlinkerFrame,
        left_rel: Optional[pd.DataFrame] = None,
        right_rel: Optional[pd.DataFrame] = None,
    ) -> KlinkerBlockManager:
        left_reduced, right_reduced = left.concat_values(), right.concat_values()
        return self._assign(
            left=left_reduced,
            right=right_reduced,
            left_rel=left_rel,
            right_rel=right_rel,
        )
