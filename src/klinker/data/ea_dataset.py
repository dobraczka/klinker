from dataclasses import dataclass
from eche import PrefixedClusterHelper
from typing import Optional, Union, Tuple

import pandas as pd
from sylloge.base import MultiSourceEADataset

from ..typing import Side
from .enhanced_df import (
    KlinkerDaskFrame,
    KlinkerFrame,
    KlinkerPandasFrame,
    KlinkerTripleDaskFrame,
    KlinkerTriplePandasFrame,
)


@dataclass
class KlinkerDataset:
    """Helper class to hold info of benchmark datasets."""

    left: KlinkerFrame
    right: KlinkerFrame
    gold: pd.DataFrame
    left_rel: Optional[pd.DataFrame] = None
    right_rel: Optional[pd.DataFrame] = None

    @classmethod
    def from_sylloge(
        cls,
        dataset: MultiSourceEADataset,
        clean: bool = False,
        repartition: Optional[int] = None,
    ) -> "KlinkerDataset":
        """Create a klinker dataset from sylloge dataset.

        Args:
        ----
          dataset: EADataset: Sylloge dataset.
          clean: bool: Clean attribute information.

        Returns:
        -------
            klinker dataset

        Examples:
        --------
            >>> # doctest: +SKIP
            >>> from sylloge import MovieGraphBenchmark
            >>> ds = KlinkerDataset.from_sylloge(MovieGraphBenchmark())

        """
        left: Union[KlinkerDaskFrame, KlinkerPandasFrame]
        right: Union[KlinkerDaskFrame, KlinkerPandasFrame]
        ds_names = dataset.dataset_names

        attr_left = dataset.attr_triples[0]
        attr_right = dataset.attr_triples[1]
        left_rel = dataset.rel_triples[0]
        right_rel = dataset.rel_triples[1]
        if dataset.backend == "pandas":
            left = KlinkerTriplePandasFrame.from_df(
                attr_left, table_name=ds_names[0], id_col="head"
            )
            right = KlinkerTriplePandasFrame.from_df(
                attr_right, table_name=ds_names[1], id_col="head"
            )
        elif dataset.backend == "dask":
            if repartition:
                attr_left, attr_right, left_rel, right_rel = [
                    frame.repartition(npartitions=repartition)
                    for frame in [
                        dataset.attr_triples[0],
                        dataset.attr_triples[1],
                        left_rel,
                        right_rel,
                    ]
                ]
            left = KlinkerTripleDaskFrame.from_dask_dataframe(
                attr_left, table_name=ds_names[0], id_col="head"
            )
            right = KlinkerTripleDaskFrame.from_dask_dataframe(
                attr_right, table_name=ds_names[1], id_col="head"
            )
        else:
            raise ValueError(f"Unknown dataset backend {dataset.backend}")

        if clean:
            # remove datatype
            left["tail"] = left["tail"].map(lambda x: str(x).split("^^")[0])
            right["tail"] = right["tail"].map(lambda x: str(x).split("^^")[0])

        if isinstance(dataset.ent_links, PrefixedClusterHelper):
            ent_links = pd.DataFrame(
                dataset.ent_links.all_pairs_no_intra(), columns=dataset.dataset_names
            )
        else:
            ent_links = dataset.ent_links.rename(
                columns={
                    "left": dataset.dataset_names[0],
                    "right": dataset.dataset_names[1],
                }
            )
        return cls(
            left=left,
            right=right,
            left_rel=left_rel,
            right_rel=right_rel,
            gold=ent_links,
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

    def sample(self, frac: float) -> "KlinkerDataset":
        """Get a sample of the dataset.

        Note:
        ----
            Currently this only takes the first n entities of the gold standard.

        Args:
        ----
          frac: percentage of whole
        Returns:
        -------
            sampled klinker dataset

        Examples:
        --------
            >>> # doctest: +SKIP
            >>> from sylloge import MovieGraphBenchmark
            >>> ds = KlinkerDataset.from_sylloge(MovieGraphBenchmark())
            >>> sampled = ds.sample(0.2)

        """
        # TODO actually sample
        sample_ent_links = self.gold.sample(frac=frac)
        sample_left, sample_left_rel = self._sample_side(sample_ent_links, "left")
        sample_right, sample_right_rel = self._sample_side(sample_ent_links, "right")
        return KlinkerDataset(
            left=sample_left,
            right=sample_right,
            left_rel=sample_left_rel,
            right_rel=sample_right_rel,
            gold=sample_ent_links,
        )
