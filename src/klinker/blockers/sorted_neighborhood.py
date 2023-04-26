from typing import Dict, List, Union, Optional

import pandas as pd

from .standard import StandardBlocker
from ..data import KlinkerFrame


class SortedNeighborhoodBlocker(StandardBlocker):
    def __init__(self, blocking_key: Union[str, List[str]], window_size: int = 3):
        self.blocking_key = blocking_key
        self.window_size = window_size

    def _assign(
        self,
        left: KlinkerFrame,
        right: KlinkerFrame,
        left_rel: Optional[pd.DataFrame] = None,
        right_rel: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        name_id_tuple_col = "name_id_tuple"
        tables = [left, right]
        for tab in tables:
            tab[name_id_tuple_col] = tab[tab.id_col].apply(
                lambda x, name: (name, x), name=tab.name
            )
        conc = pd.concat(tables)

        res: Dict = {tab.name: {} for tab in tables}
        for w_id, window in enumerate(
            conc.sort_values(by=self.blocking_key)[name_id_tuple_col].rolling(
                window=self.window_size
            )
        ):
            if len(window) == self.window_size:
                for entry in window:
                    entry_ds_name = entry[0]
                    entry_id = entry[1]
                    if w_id in res[entry_ds_name]:
                        res[entry_ds_name][w_id].append(entry_id)
                    else:
                        res[entry_ds_name][w_id] = [entry_id]
        return pd.DataFrame(res)
