from dataclasses import dataclass
from typing import Optional, Union

import pandas as pd
from sylloge.base import EADataset

from .enhanced_df import (
    KlinkerDaskFrame,
    KlinkerFrame,
    KlinkerPandasFrame,
    KlinkerTripleDaskFrame,
    KlinkerTriplePandasFrame,
)
from ..typing import Side, Tuple
from ..utils import tokenize_row


@dataclass
class KlinkerDataset:
    """Helper class to hold info of benchmark datasets."""
    left: KlinkerFrame
    right: KlinkerFrame
    gold: pd.DataFrame
    left_rel: Optional[pd.DataFrame] = None
    right_rel: Optional[pd.DataFrame] = None


    @classmethod
    def from_sylloge(cls, dataset: EADataset, clean: bool = False) -> "KlinkerDataset":
        """Create a klinker dataset from sylloge dataset.

        Args:
          dataset: EADataset: Sylloge dataset.
          clean: bool: Clean attribute information.

        Returns:
            klinker dataset

        Example:
            >>> from sylloge import MovieGraphBenchmark
            >>> ds = KlinkerDataset.from_sylloge(MovieGraphBenchmark())
        """
        left: Union[KlinkerDaskFrame, KlinkerPandasFrame]
        right: Union[KlinkerDaskFrame, KlinkerPandasFrame]
        if dataset.backend == "pandas":
            left = KlinkerTriplePandasFrame.from_df(
                dataset.attr_triples_left, table_name="left", id_col="head"
            )
            right = KlinkerTriplePandasFrame.from_df(
                dataset.attr_triples_right, table_name="right", id_col="head"
            )
        elif dataset.backend == "dask":
            left = KlinkerTripleDaskFrame.from_dask_dataframe(
                dataset.attr_triples_left, table_name="left", id_col="head"
            )
            right = KlinkerTripleDaskFrame.from_dask_dataframe(
                dataset.attr_triples_right, table_name="right", id_col="head"
            )
        else:
            raise ValueError(f"Unknown dataset backend {dataset.backend}")

        if clean:
            # remove datatype
            left["tail"] = left["tail"].map(lambda x: str(x).split("^^")[0])
            right["tail"] = right["tail"].map(lambda x: str(x).split("^^")[0])

        return cls(
            left=left,
            right=right,
            left_rel=dataset.rel_triples_left,
            right_rel=dataset.rel_triples_right,
            gold=dataset.ent_links,
        )

    def _sample_side(
        self, sample: pd.DataFrame, side: Side
    ) -> Tuple[KlinkerFrame, Optional[pd.DataFrame]]:
        if side == "left":
            rel_df = self.left_rel
            attr_df = self.left
            sample_col = sample.columns[0]
        else:
            rel_df = self.right_rel
            attr_df = self.right
            sample_col = sample.columns[1]
        sampled_attr_df = attr_df[attr_df[attr_df.id_col].isin(sample[sample_col])]
        if rel_df is None:
            return sampled_attr_df, None
        return (
            sampled_attr_df,
            rel_df[
                rel_df["head"].isin(sample[sample_col])
                | rel_df["tail"].isin(sample[sample_col])
            ],
        )

    def sample(self, size: int) -> "KlinkerDataset":
        """Get a sample of the dataset.

        Note:
            Currently this only takes the first n entities of the gold standard.

        Args:
          size: int: size of the sample

        Returns:
            sampled klinker dataset

        Example:
            >>> from sylloge import MovieGraphBenchmark
            >>> ds = KlinkerDataset.from_sylloge(MovieGraphBenchmark())
            >>> sampled = ds.sample(10)
        """
        # TODO actually sample
        sample_ent_links = self.gold.iloc[:size]
        sample_left, sample_left_rel = self._sample_side(sample_ent_links, "left")
        sample_right, sample_right_rel = self._sample_side(sample_ent_links, "right")
        return KlinkerDataset(
            left=sample_left,
            right=sample_right,
            left_rel=sample_left_rel,
            right_rel=sample_right_rel,
            gold=sample_ent_links,
        )
