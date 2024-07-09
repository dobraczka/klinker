import itertools
import pathlib
import pickle
from typing import (
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pyarrow as pa
from deprecated import deprecated


EntityIdTypeVar = TypeVar("EntityIdTypeVar", str, int)
BlockIdTypeVar = TypeVar("BlockIdTypeVar", str, int)


class KlinkerBlockManager:
    """Class for handling of blocks.

    Args:
    ----
        blocks: dataframe with blocks.

    Examples:
    --------
        >>> from klinker import KlinkerBlockManager
        >>> kbm = KlinkerBlockManager.from_dict({ "block1": [[1,3,4],[3,4,5]], "block2": [[3,4,5],[5,6]]}, dataset_names=("A","B"))
        >>> kbm.blocks.compute()
                        A          B
        block1  [1, 3, 4]  [3, 4, 5]
        block2  [3, 4, 5]     [5, 6]
        >>> kbm["block1"].compute()
                        A          B
        block1  [1, 3, 4]  [3, 4, 5]
        >>> len(kbm)
        2
        >>> set(kbm.all_pairs())
        {(4, 4), (5, 5), (3, 4), (1, 5), (4, 3), (4, 6), (1, 4), (4, 5), (3, 3), (5, 6), (3, 6), (1, 3), (3, 5)}
        >>> kbm.block_sizes
        block1    6
        block2    5
        Name: block_sizes, dtype: int64
        >>> kbm.mean_block_size
        5.5
        >>> kbm.to_dict()
        {'block1': ([1, 3, 4], [3, 4, 5]), 'block2': ([3, 4, 5], [5, 6])}

        ```
    """

    def __init__(self, blocks: dd.DataFrame):
        self.blocks = blocks
        self._grouped: Optional[Tuple] = None

    def __getitem__(self, key):
        return self.blocks.loc[key]

    def __len__(self) -> int:
        return len(self.blocks)

    def __repr__(self) -> str:
        return f"KlinkerBlockManager(blocks=\n{self.blocks.__repr__()})"

    def to_dict(self) -> Dict[Union[str, int], Tuple[Union[str, int], Union[str, int]]]:
        """Return blocks as dict.

        Returns
        -------
          The dict has block names as keys and a tuple of sets of entity ids.
        """
        return (
            self.blocks.apply(tuple, axis=1, meta=pd.Series([], dtype=object))
            .compute()
            .to_dict()
        )

    def find_blocks(self, entity_id: Union[str, int], column_id: int) -> np.ndarray:
        """Find blocks where entity id belongs to.

        Args:
        ----
          entity_id: Union[str, int]: Entity id.
          column_id: int: Whether entity belongs to left (0) or right (1) dataset.

        Returns:
        -------
            Blocks where entity id belongs to.
        """
        if self._grouped is None:
            grouped = []
            for column_name in self.blocks.columns:
                cur_ex = self.blocks[column_name].explode()
                grouped.append(cur_ex.to_frame().groupby(by=column_name))
            self._grouped = tuple(grouped)
        assert self._grouped  # for mypy
        return self._grouped[column_id].get_group(entity_id).index.values.compute()

    def entity_pairs(
        self, entity_id: Union[str, int], column_id: int
    ) -> Generator[Tuple[Union[int, str], ...], None, None]:
        """Get all pairs where this entity shows up.

        Args:
        ----
          entity_id: Union[str, int]: Entity id.
          column_id: int: Whether entity belongs to left (0) or right (1) dataset.

        Returns:
        -------
            Generator for these pairs.
        """
        cur_blocks = self.find_blocks(entity_id, column_id)
        other_column = 0 if column_id == 1 else 1
        other_column_name = self.blocks.columns[other_column]
        return (
            pair
            for blk_name in cur_blocks
            for _, blk in self.blocks.loc[blk_name][other_column_name].compute().items()
            for pair in itertools.product({entity_id}, blk)
        )

    def all_pairs(self) -> Generator[Tuple[Union[int, str], ...], None, None]:
        """Get all pairs.

        Returns
        -------
            Generator that creates all pairs, from blocks (including duplicates).
        """
        for block_tuple in self.blocks.itertuples(index=False, name=None):
            yield from itertools.product(*block_tuple)

    def inner_block_assignments(self) -> dd.DataFrame:
        return self.blocks.applymap(len)

    def block_assignments(self) -> dd.DataFrame:
        return self.inner_block_assignments().sum(axis=1)

    def block_comparisons(self) -> dd.DataFrame:
        ibs = self.inner_block_assignments()
        return ibs[self.blocks.columns[0]] * ibs[self.blocks.columns[1]]

    def individual_blocking_cardinality_per_ds(
        self, dataset_lens: List[int]
    ) -> Tuple[float, float]:
        return tuple(
            self.blocks[col].apply(len, meta=(col, "int64")).sum().compute() / ds_len
            for col, ds_len in zip(self.blocks.columns, dataset_lens)
        )

    def overall_blocking_cardinality(self, dataset_lens: List[int]) -> float:
        return self.block_assignments().sum().compute() / sum(dataset_lens)

    def comparisons_cardinality(self) -> float:
        block_sizes = self.blocks.applymap(len)
        sum_of_block_sizes = block_sizes.sum().sum().compute()
        aggregate_cardinality = (
            (block_sizes[self.blocks.columns[0]] * block_sizes[self.blocks.columns[1]])
            .sum()
            .compute()
        )
        return sum_of_block_sizes / aggregate_cardinality

    def _iterative_get_purge_threshold(self) -> int:
        block_stats = self.inner_block_assignments()
        # add block comparisons to block_stats
        block_stats["ind_block_card"] = (
            block_stats[block_stats.columns[0]] * block_stats[block_stats.columns[1]]
        )
        block_stats = block_stats.reset_index().set_index("ind_block_card")
        block_assignments = 0
        total_comparisons = 0
        last_i_cardinality = 1
        stats = []
        for (
            block_card,
            _,
            left_num_assign,
            right_num_assign,
        ) in block_stats.compute().itertuples(name=None):
            if last_i_cardinality < block_card:
                stats.append(
                    {
                        "i_cardinality": last_i_cardinality,
                        "cc": block_assignments / total_comparisons,
                    }
                )
                last_i_cardinality = block_card
            block_assignments += left_num_assign + right_num_assign
            total_comparisons += block_card

        stats.append(
            {
                "i_cardinality": last_i_cardinality,
                "cc": block_assignments / total_comparisons,
            }
        )
        max_i_cardinality = last_i_cardinality
        for i, cur_stats in enumerate(stats):
            if cur_stats["cc"] == stats[i - 1]["cc"]:
                max_i_cardinality = cur_stats["i_cardinality"]  # type: ignore[assignment]
                break
        return max_i_cardinality

    def _get_purge_threshold(self, round_cc: int) -> int:
        left_col, right_col = self.blocks.columns
        block_stats = self.inner_block_assignments()
        # add block comparisons to block_stats
        block_stats["ind_block_card"] = (
            block_stats[block_stats.columns[0]] * block_stats[block_stats.columns[1]]
        )
        block_stats["block_assignments"] = (
            block_stats[left_col] + block_stats[right_col]
        )
        block_stats["block_card"] = block_stats[left_col] * block_stats[right_col]
        bs = block_stats.reset_index().set_index("ind_block_card").compute()

        bs = bs[~bs.index.duplicated(keep="first")]
        bs["i_card"] = bs["block_card"].cumsum()
        bs["cc"] = bs["block_assignments"].cumsum() / bs["block_card"].cumsum()
        find_mask = bs["cc"].round(round_cc).duplicated(keep="first")
        if find_mask.any():
            return bs[find_mask].head(1)["i_card"].iloc[0]
        return bs.iloc[-1]["i_card"].iloc[0]

    def purge(self, round_cc=int) -> "KlinkerBlockManager":
        purge_threshold = self._get_purge_threshold(round_cc)
        left_col, right_col = self.blocks.columns
        block_stats = self.inner_block_assignments()
        # add block comparisons to block_stats
        block_stats["ind_block_card"] = (
            block_stats[block_stats.columns[0]] * block_stats[block_stats.columns[1]]
        )
        bs = block_stats.compute()
        blocks = self.blocks.loc[bs[bs["ind_block_card"] <= purge_threshold].index]
        return KlinkerBlockManager(blocks)

    @classmethod
    def combine(
        cls, this: "KlinkerBlockManager", other: "KlinkerBlockManager"
    ) -> "KlinkerBlockManager":
        """Combine blocks.

        Args:
        ----
          this: one block manager to combine
          other: other block manager to combine

        Returns:
        -------
          Combined KlinkerBlockManager

        Examples:
        --------
            >>> from klinker import KlinkerBlockManager
            >>> kbm = KlinkerBlockManager.from_dict({"block1": [[1,3,4],[3,4,5]], "block2": [[3,4,5],[5,6]]}, dataset_names=("A","B"))
            >>> kbm2 = KlinkerBlockManager.from_dict({"block3": [[7,4],[12,8]]}, dataset_names=("A","B"))
            >>> kbm_merged = KlinkerBlockManager.combine(kbm, kbm2)
            >>> kbm_merged.blocks.compute()
                            A          B
            block1  [1, 3, 4]  [3, 4, 5]
            block2  [3, 4, 5]     [5, 6]
            block3     [7, 4]    [12, 8]

        """

        def _merge_blocks(
            row: pd.Series, output_names: Sequence[str], left_right_names: Sequence[str]
        ):
            nonnull = row[~row.isnull()]
            if len(nonnull) == 2:  # no block overlap
                nonnull.index = output_names
                return nonnull
            else:
                A_left = set(nonnull[left_right_names[0]])
                A_right = set(nonnull[left_right_names[2]])
                B_left = set(nonnull[left_right_names[1]])
                B_right = set(nonnull[left_right_names[3]])
                A = list(A_left.union(A_right))
                B = list(B_left.union(B_right))
                return pd.Series([A, B], index=output_names, name=nonnull.name)

        if list(this.blocks.columns) != list(other.blocks.columns):
            raise ValueError("Cannot combine blocks from different datasets!")

        output_names = this.blocks.columns
        left_suffix = "left"
        right_suffix = "right"
        left_right_names = [
            col + suffix
            for col_names, suffix in zip(
                [this.blocks.columns, other.blocks.columns], [left_suffix, right_suffix]
            )
            for col in col_names
        ]
        joined = this.blocks.join(
            other.blocks, how="outer", lsuffix="left", rsuffix="right"
        )

        meta = pd.DataFrame([], columns=output_names)
        return cls(
            joined.apply(
                _merge_blocks,
                output_names=output_names,
                left_right_names=left_right_names,
                axis=1,
                meta=meta,
            )
        )

    def to_parquet(self, path: Union[str, pathlib.Path], **kwargs):
        """Write blocks as parquet file(s).

        Args:
        ----
          path: Union[str, pathlib.Path]: Where to write.
          **kwargs: passed to the parquet function
        """
        if "schema" not in kwargs:
            left, right = self.blocks.columns[:2]
            block_type = pa.list_(pa.string())
            schema = {
                left: block_type,
                right: block_type,
            }
        else:
            schema = kwargs.pop["schema"]  # type: ignore
        try:
            self.blocks.to_parquet(path, schema=schema, **kwargs)
        except ValueError:
            # If index is incorrectly assumed by dask to be string
            # and it turns out to be int64 an error would be thrown
            # This is kind of a dirty hack
            schema["__null_dask_index__"] = pa.int64()
            self.blocks.to_parquet(path, schema=schema, **kwargs)

    @staticmethod
    def read_parquet(
        path: Union[str, pathlib.Path],
        calculate_divisions: bool = True,
        **kwargs,
    ) -> "KlinkerBlockManager":
        """Read blocks from parquet.

        Args:
        ----
          path: Union[str, pathlib.Path]: Path where blocks are stored.
          calculate_divisions: bool: Calculate index divisions.
          **kwargs: Passed to `dd.read_parquet` function.

        Returns:
        -------
            Blocks as KlinkerBlockManager
        """
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        if path.joinpath("nn_blocks").exists():
            return CompositeWithNNBasedKlinkerBlockManager.read_parquet(
                path, calculate_divisions=calculate_divisions, **kwargs
            )
        blocks = dd.read_parquet(
            path=path,
            calculate_divisions=calculate_divisions,
            **kwargs,
        )
        if len(blocks.columns) > 2:
            return NNBasedKlinkerBlockManager(blocks)
        # for the rare case, that NN was <=2
        if isinstance(
            blocks[blocks.columns[0]].head(1, npartitions=-1).values[0], (str, int)
        ):
            return NNBasedKlinkerBlockManager(blocks)
        return KlinkerBlockManager(blocks)

    @classmethod
    def from_pandas(
        cls, df: pd.DataFrame, npartitions: int = 1, **kwargs
    ) -> "KlinkerBlockManager":
        """Create from pandas.

        Args:
        ----
          df: pd.DataFrame: DataFrame
          npartitions: int:  Partitions for dask
          **kwargs: Passed to `dd.from_pandas`

        Returns:
        -------
            Blocks as KlinkerBlockManager

        Examples:
        --------
            >>> import pandas as pd
            >>> from klinker import KlinkerBlockManager
            >>> pd_blocks = pd.DataFrame({'A': {'block1': [1, 3, 4], 'block2': [3, 4, 5]}, 'B': {'block1': [3, 4, 5], 'block2': [5, 6]}})
            >>> kbm = KlinkerBlockManager.from_pandas(pd_blocks)

        """
        return cls(dd.from_pandas(df, npartitions=npartitions, **kwargs))

    @classmethod
    def from_dict(
        cls,
        block_dict: Dict[
            BlockIdTypeVar, Tuple[List[EntityIdTypeVar], List[EntityIdTypeVar]]
        ],
        dataset_names: Tuple[str, str] = ("left", "right"),
        npartitions: int = 1,
        **kwargs,
    ) -> "KlinkerBlockManager":
        """Args:
        ----
          block_dict: Dictionary with block information.
          dataset_names: Tuple[str, str]: Tuple of dataset names.
          npartitions: int: Partitions used for dask.
          **kwargs: Passed to `dd.from_dict`.

        Returns
        -------
            Blocks as KlinkerBlockManager

        Examples
        --------
            >>> from klinker import KlinkerBlockManager
            >>> kbm = KlinkerBlockManager.from_dict({"block1": [[1,3,4],[3,4,5]], "block2": [[3,4,5],[5,6]]}, dataset_names=("A","B"))

        """
        return cls(
            dd.from_dict(
                block_dict,
                orient="index",
                columns=dataset_names,
                npartitions=npartitions,
                **kwargs,
            )
        )

    @classmethod
    @deprecated(reason="Please use parquet files")
    def read_pickle(cls, path) -> "KlinkerBlockManager":
        with open(path, "rb") as in_file:
            res = pickle.load(in_file)
            if isinstance(res, dict):
                return cls.from_dict(res)
            elif isinstance(res, pd.DataFrame):
                return cls.from_pandas(res)
            elif hasattr(res, "blocks") and isinstance(res.blocks, dict):
                return cls.from_dict(
                    {
                        bk: (list(left_v), list(right_v))
                        for bk, (left_v, right_v) in res.blocks.items()
                    }
                )  # type: ignore
            else:
                raise ValueError(f"Unknown pickled object of type {type(res)}")


class NNBasedKlinkerBlockManager(KlinkerBlockManager):
    def to_dict(self) -> Dict[Union[str, int], Tuple[Union[str, int], Union[str, int]]]:
        raise NotImplementedError

    def find_blocks(self, entity_id: Union[str, int], column_id: int) -> np.ndarray:
        raise NotImplementedError

    @classmethod
    def from_dict(
        cls,
        block_dict: Dict[
            BlockIdTypeVar, Tuple[List[EntityIdTypeVar], List[EntityIdTypeVar]]
        ],
        dataset_names: Tuple[str, str] = ("left", "right"),
        npartitions: int = 1,
        **kwargs,
    ) -> "KlinkerBlockManager":
        raise NotImplementedError

    def to_parquet(self, path: Union[str, pathlib.Path], **kwargs):
        self.blocks.to_parquet(path, **kwargs)

    def entity_pairs(
        self, entity_id: Union[str, int], column_id: int
    ) -> Generator[Tuple[Union[int, str], ...], None, None]:
        raise NotImplementedError

    def all_pairs(self) -> Generator[Tuple[Union[int, str], ...], None, None]:
        """Get all pairs.

        Returns
        -------
            Generator that creates all pairs, from blocks (including duplicates).
        """
        for row in self.blocks.itertuples(name=None):
            # first entry in itertuples here is index
            for pair in itertools.product([row[0]], row[1:]):
                if pair[1] is None:
                    continue
                yield pair

    @property
    def block_sizes(self) -> pd.DataFrame:
        """Sizes of blocks."""
        return (
            self.blocks.apply(
                np.count_nonzero, axis=1, meta=pd.Series([], dtype="int64")
            ).compute()
            + 1
        )

    @classmethod
    def combine(
        cls, this: "KlinkerBlockManager", other: "KlinkerBlockManager"
    ) -> "NNBasedKlinkerBlockManager":
        len_this = len(this.blocks.columns)
        len_other = len(other.blocks.columns)
        # we need string columns for saving to parquet
        new_cols = list(map(str, range(len_this + len_other)))
        cc = this.blocks.join(other.blocks, lsuffix="l", rsuffix="r", how="outer")
        cc.columns = new_cols
        return cls(cc)

    def _create_stat_df(self, name: str, offset: int) -> dd.DataFrame:
        nn_num = len(self.blocks.columns)
        # to keep index
        empty_df = self.blocks[[]]
        empty_df[name] = offset + nn_num
        return empty_df

    def block_assignments(self) -> dd.DataFrame:
        return self._create_stat_df("block_assignments", 1)

    def block_comparisons(self) -> dd.DataFrame:
        return self._create_stat_df("block_comparisons", 0)

    def individual_blocking_cardinality_per_ds(
        self, dataset_lens: List[int]
    ) -> Tuple[float, float]:
        left = 1.0
        right = len(self.blocks.columns) / dataset_lens[1]
        return left, right

    def overall_blocking_cardinality(self, dataset_lens: List[int]) -> float:
        numerator = dataset_lens[0] + len(self.blocks.columns)
        return numerator / sum(dataset_lens)

    def comparisons_cardinality(self) -> float:
        nn_num = len(self.blocks.columns)
        sum_of_block_sizes = 1 + nn_num
        aggregate_cardinality = 1 * nn_num
        return sum_of_block_sizes / aggregate_cardinality


class CompositeWithNNBasedKlinkerBlockManager(KlinkerBlockManager):
    def __init__(
        self, blocks: KlinkerBlockManager, nn_blocks: NNBasedKlinkerBlockManager
    ):
        self._blocks = blocks
        self._nn_blocks = nn_blocks

    def __repr__(self) -> str:
        return f"KlinkerBlockManager(blocks=\n{self._blocks.__repr__()}\nnn_blocks=\n{self._nn_blocks.__repr__()})"

    def to_dict(self):
        raise NotImplementedError

    def find_blocks(self, entity_id: Union[str, int], column_id: int):
        raise NotImplementedError

    def entity_pairs(self, entity_id: Union[str, int], column_id: int):
        raise NotImplementedError

    def all_pairs(self):
        for pair in itertools.chain(
            self._blocks.all_pairs(), self._nn_blocks.all_pairs()
        ):
            yield pair

    def inner_block_assignments(self):
        raise NotImplementedError

    def block_assignments(self):
        raise NotImplementedError

    def block_comparisons(self):
        raise NotImplementedError

    def individual_blocking_cardinality_per_ds(self, dataset_lens: List[int]):
        raise NotImplementedError

    def overall_blocking_cardinality(self, dataset_lens: List[int]):
        raise NotImplementedError

    def comparisons_cardinality(self):
        raise NotImplementedError

    def purge(self, round_cc=int):
        raise NotImplementedError

    def to_parquet(self, path: Union[str, pathlib.Path], **kwargs):
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        self._blocks.to_parquet(path.joinpath("blocks"), **kwargs)
        self._nn_blocks.to_parquet(path.joinpath("nn_blocks"), **kwargs)

    @staticmethod
    def read_parquet(
        path: Union[str, pathlib.Path],
        calculate_divisions: bool = True,
        **kwargs,
    ) -> "KlinkerBlockManager":
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        blocks = KlinkerBlockManager.read_parquet(
            path.joinpath("blocks"), calculate_divisions=calculate_divisions, **kwargs
        )
        nn_blocks = NNBasedKlinkerBlockManager.read_parquet(
            path.joinpath("nn_blocks"),
            calculate_divisions=calculate_divisions,
            **kwargs,
        )
        assert isinstance(nn_blocks, NNBasedKlinkerBlockManager)
        return CompositeWithNNBasedKlinkerBlockManager(
            blocks=blocks, nn_blocks=nn_blocks
        )

    @classmethod
    def from_pandas(
        cls, df: pd.DataFrame, npartitions: int = 1, **kwargs
    ) -> "KlinkerBlockManager":
        raise NotImplementedError

    @classmethod
    def from_dict(
        cls,
        block_dict: Dict[
            BlockIdTypeVar, Tuple[List[EntityIdTypeVar], List[EntityIdTypeVar]]
        ],
        dataset_names: Tuple[str, str] = ("left", "right"),
        npartitions: int = 1,
        **kwargs,
    ) -> "KlinkerBlockManager":
        raise NotImplementedError

    def read_pickle(cls, path):
        raise NotImplementedError


# TODO maybe just use composite to enable non-destructive combination
def combine_blocks(
    this: "KlinkerBlockManager", other: "KlinkerBlockManager"
) -> "KlinkerBlockManager":
    if isinstance(this, NNBasedKlinkerBlockManager) and isinstance(
        other, NNBasedKlinkerBlockManager
    ):
        return NNBasedKlinkerBlockManager.combine(this, other)
    if isinstance(this, KlinkerBlockManager) and isinstance(
        other, NNBasedKlinkerBlockManager
    ):
        return CompositeWithNNBasedKlinkerBlockManager(this, other)
    if isinstance(other, KlinkerBlockManager) and isinstance(
        this, NNBasedKlinkerBlockManager
    ):
        return CompositeWithNNBasedKlinkerBlockManager(other, this)
    return KlinkerBlockManager.combine(this, other)


if __name__ == "__main__":
    from klinker.data import KlinkerDataset
    from sylloge import OpenEA
    from klinker.eval import Evaluation

    ds = KlinkerDataset.from_sylloge(OpenEA())
    blocks = KlinkerBlockManager.read_parquet(
        "experiment_artifacts/openea_d_w_15k_v1/SimpleRelationalTokenBlocker/01af7d8c9ee424b0ca69e99ebcdc3645efd53aa8_blocks.parquet"
    )
    # print("No purge:")
    # print(Evaluation.from_dataset(blocks, ds).to_dict())
    print("purge 6:")
    print(Evaluation.from_dataset(blocks.purge(6), ds).to_dict())
    print("purge 5:")
    print(Evaluation.from_dataset(blocks.purge(5), ds).to_dict())
    print("purge 4:")
    print(Evaluation.from_dataset(blocks.purge(4), ds).to_dict())
    print("purge 3:")
    print(Evaluation.from_dataset(blocks.purge(3), ds).to_dict())
    print("purge 2:")
    print(Evaluation.from_dataset(blocks.purge(2), ds).to_dict())
