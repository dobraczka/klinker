from dataclasses import dataclass
from eche import PrefixedClusterHelper
from typing import Optional, Tuple

import pandas as pd
from sylloge.base import MultiSourceEADataset

from ..typing import Side, FrameType


@dataclass
class KlinkerDataset:
    """Helper class to hold info of benchmark datasets."""

    left: FrameType
    right: FrameType
    gold: pd.DataFrame
    left_rel: Optional[FrameType] = None
    right_rel: Optional[FrameType] = None
    left_id_col: str = "head"
    right_id_col: str = "head"
    left_table_name: str = "left"
    right_table_name: str = "right"

    @classmethod
    def from_sylloge(
        cls,
        dataset: MultiSourceEADataset,
        clean: bool = False,
        partition_size: Optional[str] = None,
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
        if dataset.backend == "dask" and partition_size:
            attr_left, attr_right, left_rel, right_rel = [
                frame.repartition(partition_size=partition_size)
                for frame in [
                    dataset.attr_triples[0],
                    dataset.attr_triples[1],
                    dataset.rel_triples[0],
                    dataset.rel_triples[1],
                ]
            ]
        left = dataset.attr_triples[0]
        right = dataset.attr_triples[1]
        left_rel = dataset.rel_triples[0]
        right_rel = dataset.rel_triples[1]

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
            left_id_col="head",
            right_id_col="head",
            left_table_name=dataset.dataset_names[0],
            right_table_name=dataset.dataset_names[0],
        )

    def _sample_side(
        self, sample: FrameType, side: Side, id_col: str = "head"
    ) -> Tuple[FrameType, Optional[pd.DataFrame]]:
        if side == "left":
            rel_df = self.left_rel
            attr_df = self.left
            sample_col = sample.columns[0]
        else:
            rel_df = self.right_rel
            attr_df = self.right
            sample_col = sample.columns[1]
        sampled_attr_df = attr_df[attr_df[id_col].isin(sample[sample_col])]
        if rel_df is None:
            return sampled_attr_df, None
        return (
            sampled_attr_df,
            rel_df[
                rel_df["head"].isin(sample[sample_col])
                | rel_df["tail"].isin(sample[sample_col])
            ],
        )

    def sample(self, frac: float, id_col: str = "head") -> "KlinkerDataset":
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
        sample_left, sample_left_rel = self._sample_side(
            sample_ent_links, "left", id_col
        )
        sample_right, sample_right_rel = self._sample_side(
            sample_ent_links, "right", id_col
        )
        return KlinkerDataset(
            left=sample_left,
            right=sample_right,
            left_rel=sample_left_rel,
            right_rel=sample_right_rel,
            gold=sample_ent_links,
            left_id_col=self.left_id_col,
            right_id_col=self.right_id_col,
            left_table_name=self.left_table_name,
            right_table_name=self.right_table_name,
        )
