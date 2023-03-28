import abc
from typing import Iterable, List, Union, Optional

import pandas as pd

from klinker.data import KlinkerFrame
from klinker.blockers.base import Blocker


class AttributeEmbeddingBlocker(Blocker):
    def __init__(
        self,
        blocking_key: Optional[Union[str, List[str]]] = None,
    ):
        self.blocking_key = blocking_key

    def _assign(self, left: KlinkerFrame, right: KlinkerFrame) -> pd.DataFrame:
        raise NotImplementedError
