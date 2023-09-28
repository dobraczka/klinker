from typing import Dict, List, Optional, Tuple, Union

import dask.dataframe as dd
import pandas as pd

from .standard import StandardBlocker
from ..data import KlinkerBlockManager, KlinkerFrame, KlinkerPandasFrame


class SortedNeighborhoodBlocker(StandardBlocker):
    """Uses sorted neighborhood blocking.


    Examples:

        >>> # doctest: +SKIP
        >>> from sylloge import MovieGraphBenchmark
        >>> from klinker.data import KlinkerDataset
        >>> ds = KlinkerDataset.from_sylloge(MovieGraphBenchmark(),clean=True)
        >>> from klinker.blockers import SortedNeighborhoodBlocker
        >>> blocker = SortedNeighborhoodBlocker(blocking_key="tail")
        >>> blocks = blocker.assign(left=ds.left, right=ds.right)

    """

    def __init__(self, blocking_key: Union[str, Tuple[str, str]], window_size: int = 3):
        self.blocking_key = blocking_key
        self.window_size = window_size

    def assign(
        self,
        left: KlinkerFrame,
        right: KlinkerFrame,
        left_rel: Optional[KlinkerFrame] = None,
        right_rel: Optional[KlinkerFrame] = None,
    ) -> KlinkerBlockManager:
        """Assign entity ids to blocks.

        Note:
            not implemented for Dask.

        Args:
          left: KlinkerFrame: Contains entity attribute information of left dataset.
          right: KlinkerFrame: Contains entity attribute information of right dataset.
          left_rel: Optional[KlinkerFrame]:  (Default value = None) Contains relational information of left dataset.
          right_rel: Optional[KlinkerFrame]:  (Default value = None) Contains relational information of left dataset.

        Returns:
            KlinkerBlockManager: instance holding the resulting blocks.

        Raises:
            NotImplementedError: if frames are using dask.

        """
        if not isinstance(left, KlinkerPandasFrame):
            raise NotImplementedError("Not implemented for Dask!")
        name_id_tuple_col = "name_id_tuple"
        tables = [left, right]
        for tab in tables:
            tab[name_id_tuple_col] = tab[tab.id_col].apply(
                lambda x, name: (name, x), name=tab.table_name
            )
        conc = pd.concat(tables)

        res: Dict = {tab.table_name: {} for tab in tables}
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
                        res[entry_ds_name][w_id].add(entry_id)
                    else:
                        res[entry_ds_name][w_id] = {entry_id}
        return KlinkerBlockManager.from_pandas(
            pd.DataFrame(res).dropna().applymap(list)
        )
