import itertools
import math
import pathlib
import pickle
from typing import (
    Dict,
    Generator,
    ItemsView,
    KeysView,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
    ValuesView,
)

import dask.bag as db
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pyarrow as pa
from dask.base import tokenize
from deprecated import deprecated
from tlz import partition_all

from ..typing import Frame

EntityIdTypeVar = TypeVar("EntityIdTypeVar", str, int)
BlockIdTypeVar = TypeVar("BlockIdTypeVar", str, int)


@deprecated(reason="Please use KlinkerBlockManager")
class OldKlinkerBlockManager:
    def __init__(
        self,
        blocks: Dict[Union[str, int], Tuple[Set[int], ...]],
        dataset_names: Tuple[str, ...],
        id_mappings: Optional[Tuple[Dict[int, str], ...]] = None,
    ):
        self._assert_tuple_len(blocks, dataset_names, id_mappings)
        self.blocks = blocks
        self.dataset_names = dataset_names
        self._assert_id_mappings_consistency(blocks, id_mappings)
        self.id_mappings = id_mappings

    @staticmethod
    def _assert_tuple_len(
        blocks: Dict[Union[str, int], Tuple[Set[int], ...]],
        dataset_names: Tuple[str, ...],
        id_mappings: Optional[Tuple[Dict[int, str], ...]] = None,
    ):
        wrong_tuple_len_err = "All tuples must have same length!"
        if not len(next(iter(blocks.values()))) == len(dataset_names):
            raise ValueError(wrong_tuple_len_err)
        if id_mappings is not None and len(dataset_names) != len(id_mappings):
            raise ValueError(wrong_tuple_len_err)

    @staticmethod
    def _assert_id_mappings_consistency(
        blocks: Dict[Union[str, int], Tuple[Set[int], ...]],
        id_mappings: Optional[Tuple[Dict[int, str], ...]] = None,
    ):
        if id_mappings:
            unique_ids: Tuple[Set[int], ...] = tuple(
                set() for i in range(len(id_mappings))
            )
            # check if all entities have a mapping
            for block_tuple in blocks.values():
                for i in range(len(id_mappings)):
                    for ent_id in block_tuple[i]:
                        unique_ids[i].add(ent_id)
                        if ent_id not in id_mappings[i]:
                            raise ValueError(
                                "If id_mappings are supplied, they have to map all ids"
                            )

    def tuple_id_from_dataset_name(self, dataset_name: str) -> int:
        tuple_id = -1
        for possible_tuple_id, ds_name in enumerate(self.dataset_names):
            if ds_name == dataset_name:
                tuple_id = possible_tuple_id
                break
        if tuple_id == -1:
            raise ValueError(f"Unknown dataset_name {dataset_name}")
        return tuple_id

    @property
    def block_sizes(self) -> pd.DataFrame:
        return pd.Series(
            {
                block_name: (sum(len(block_side) for block_side in block))
                for block_name, block in self.blocks.items()
            }
        )

    @property
    def mean_block_size(self) -> float:
        return self.block_sizes.mean()

    def get_ids(self, dataset_name: str) -> Set[Union[int, str]]:
        tuple_id = self.tuple_id_from_dataset_name(dataset_name)
        return set(itertools.chain(*(block[tuple_id] for block in self.values())))

    def entity_pairs(
        self, entity_id: Union[str, int], dataset_name: str
    ) -> Generator[Tuple[Union[int, str], ...], None, None]:
        cur_blocks = self.find_blocks(entity_id, dataset_name)
        return (
            pair
            for _, blk in cur_blocks.items()
            for pair in itertools.product({entity_id}, blk[1])
        )

    def to_pairs(
        self, replace_id_mappings=False, remove_duplicates=True
    ) -> Generator[Tuple[Union[int, str], ...], None, None]:
        def _handle_pair(replace_id_mappings, pair, id_mappings):
            if replace_id_mappings and id_mappings is not None:
                yield tuple(id_mappings[i][pair[i]] for i in range(len(pair)))
            else:
                yield pair

        if not remove_duplicates:
            for block_name, block in self.blocks.items():
                for pair in itertools.product(*block):
                    return _handle_pair(
                        replace_id_mappings=replace_id_mappings,
                        pair=pair,
                        id_mappings=self.id_mappings,
                    )
        else:
            left_ds_name = self.dataset_names[0]
            left_ids = self.get_ids(left_ds_name)
            for cur_id in left_ids:
                self.find_blocks(cur_id, left_ds_name)
                for pair in set(self.entity_pairs(cur_id, left_ds_name)):
                    return _handle_pair(
                        replace_id_mappings=replace_id_mappings,
                        pair=pair,
                        id_mappings=self.id_mappings,
                    )
        # empty generator
        yield from ()

    def count_duplicate_pairs_and_total(self) -> Tuple[int, int]:
        duplicates = 0
        total = 0
        left_ds_name = self.dataset_names[0]
        left_ids = self.get_ids(left_ds_name)
        for cur_id in left_ids:
            self.find_blocks(cur_id, left_ds_name)
            cur_pairs = list(self.entity_pairs(cur_id, left_ds_name))
            total += len(cur_pairs)
            duplicates += len(cur_pairs) - len(set(cur_pairs))
        return duplicates, total

    @classmethod
    def combine(
        cls, this: "OldKlinkerBlockManager", other: "OldKlinkerBlockManager"
    ) -> "OldKlinkerBlockManager":
        if this.dataset_names != other.dataset_names:
            raise ValueError("Cannot combine blocks from different datasets!")
        if (this.id_mappings is None) != (other.id_mappings is None):
            raise ValueError("Cannot combine blocks where only one has id_mappings!")
        new_blocks = this.blocks.copy()
        for block_name, other_block_tuple in other.blocks.items():
            if block_name in this.blocks:
                block_tuple = new_blocks[block_name]
                new_blocks[block_name] = tuple(
                    block_tuple[i].union(other_block_tuple[i])
                    for i in range(len(block_tuple))
                )
            else:
                new_blocks[block_name] = other.blocks[block_name]
        new_id_mappings = None
        if this.id_mappings and other.id_mappings:
            new_id_mappings = tuple(
                # prefer mappings from this rather than other
                # by putting this after other
                {**other.id_mappings[i], **this.id_mappings[i]}
                for i in range(len(this.id_mappings))
            )
        return cls(
            blocks=new_blocks,
            dataset_names=this.dataset_names,
            id_mappings=new_id_mappings,
        )

    def __getitem__(self, key):
        if isinstance(key, list):
            return OldKlinkerBlockManager(
                {k: self.blocks[k] for k in key},
                dataset_names=self.dataset_names,
                id_mappings=self.id_mappings,
            )
        return self.blocks[key]

    def __setitem__(self, key, value):
        self.blocks[key] = value

    def copy(self) -> "OldKlinkerBlockManager":
        return OldKlinkerBlockManager(
            self.blocks.copy(),
            self.dataset_names,
            (self.id_mappings[0].copy(), self.id_mappings[1].copy())
            if self.id_mappings
            else None,
        )

    def items(self) -> ItemsView[Union[str, int], Tuple[Set[int], ...]]:
        return self.blocks.items()

    def values(self) -> ValuesView[Tuple[Set[int], ...]]:
        return self.blocks.values()

    def keys(self) -> KeysView[Union[str, int]]:
        return self.blocks.keys()

    def __iter__(self) -> ItemsView[Union[str, int], Tuple[Set[int], ...]]:
        return self.items()

    def __contains__(self, value: Union[str, int]) -> bool:
        return value in self.blocks

    def __eq__(self, other) -> bool:
        if type(self) == type(other):
            if (
                self.blocks == other.blocks
                and self.dataset_names == other.dataset_names
            ):
                if self.id_mappings and other.id_mappings:
                    if self.id_mappings == other.id_mappings:
                        return True
                else:
                    return True
        return False

    def find_blocks(
        self, entity_id: Union[str, int], dataset_name: str
    ) -> Dict[Union[str, int], Tuple[Set[int], ...]]:
        tuple_id = self.tuple_id_from_dataset_name(dataset_name)
        return dict(
            (block_name, block)
            for block_name, block in self.blocks.items()
            if entity_id in block[tuple_id]
        )

    def __repr__(self) -> str:
        repr_str = "\n".join(
            [f"{key} | {block_tuple}" for key, block_tuple in self.blocks.items()]
        )
        repr_str += f"\ndataset_names = {self.dataset_names}, "
        repr_str += f"id_mappings = yes" if self.id_mappings else f"id_mappings = no"
        return repr_str

    def __len__(self) -> int:
        return len(self.blocks)

    @classmethod
    def from_pandas(
        cls,
        pd_blocks: Frame,
        id_mappings: Optional[Tuple[Dict[int, str], ...]] = None,
    ) -> "OldKlinkerBlockManager":
        def _ensure_set(value) -> Set:
            """

            Args:
              value:

            Returns:

            """
            if isinstance(value, set):
                return value
            elif isinstance(value, str) or isinstance(value, int):
                return {value}
            else:
                return set(value)

        if isinstance(pd_blocks, dd.DataFrame):
            pd_blocks = pd_blocks.compute()
        # remove blocks with only one entry
        max_number_nans = len(pd_blocks.columns) - 1
        pd_blocks = pd_blocks[~(pd_blocks.isnull().sum(axis=1) == max_number_nans)]
        pd_blocks = pd_blocks.applymap(_ensure_set)
        return cls(
            blocks=pd_blocks.agg(tuple, axis=1).to_dict(),
            dataset_names=tuple(pd_blocks.columns),
            id_mappings=id_mappings,
        )

    def to_pickle(self, path):
        with open(path, "wb") as out_file:
            pickle.dump(self, out_file)

    @staticmethod
    def read_pickle(path) -> "OldKlinkerBlockManager":
        with open(path, "rb") as in_file:
            return pickle.load(in_file)

    def to_bag(
        self, partition_size: Optional[int] = None, npartitions: Optional[int] = None
    ) -> db.Bag:
        if npartitions and not partition_size:
            partition_size = int(math.ceil(len(self) / npartitions))
        if npartitions is None and partition_size is None:
            if len(self) < 100:
                partition_size = 1
            else:
                partition_size = int(len(self) / 100)

        parts = list(partition_all(partition_size, self.blocks))
        name = "from_sequence-" + tokenize(self, partition_size)
        if len(parts) > 0:
            d = {(name, i): self[list(part)] for i, part in enumerate(parts)}
        else:
            d = {(name, 0): self[[]]}

        return db.Bag(d, name, len(d))


class KlinkerBlockManager:
    """Class for handling of blocks.

    Args:
        blocks: dataframe with blocks.

    Example:
        >>> from klinker import KlinkerBlockManager
        >>> kbm = KlinkerBlockManager.from_dict({"block1": [[1,3,4],[3,4,5]], "block2": [[3,4,5],[5,6]]}, dataset_names=("A","B"))
        >>> kbm.blocks.compute()
                        A          B
        block1  [1, 3, 4]  [3, 4, 5]
        block2  [3, 4, 5]     [5, 6]
        >>> kbm["block1"].compute()
                        A          B
        block1  [1, 3, 4]  [3, 4, 5]
        >>> len(kbm)
        2
        >>> >>> set(kbm.all_pairs())
        {(4, 4), (5, 5), (3, 4), (1, 5), (4, 3), (4, 6), (1, 4), (4, 5), (3, 3), (5, 6), (3, 6), (1, 3), (3, 5)}
        >>> kbm.block_sizes
        block1    6
        block2    5
        Name: block_sizes, dtype: int64
        >>> kbm.mean_block_size
        5.5
        >>> kbm.to_dict()
        {'block1': ([1, 3, 4], [3, 4, 5]), 'block2': ([3, 4, 5], [5, 6])}
    """

    def __init__(self, blocks: dd.DataFrame):
        self.blocks = blocks
        grouped = []
        for column_name in self.blocks.columns:
            cur_ex = self.blocks[column_name].explode()
            grouped.append(cur_ex.to_frame().groupby(by=column_name))
        self._grouped = tuple(grouped)

    def __getitem__(self, key):
        return self.blocks.loc[key]

    def __eq__(self, other) -> bool:
        if not len(self) == len(other):
            return False
        for blk_name in self.blocks.index:
            if not self[blk_name].compute().equals(other[blk_name].compute()):
                return False
        return True

    def __len__(self) -> int:
        return len(self.blocks)

    def __repr__(self) -> str:
        return f"KlinkerBlockManager(blocks=\n{self.blocks.__repr__()})"

    def to_dict(self) -> Dict[Union[str, int], Tuple[Union[str, int], Union[str, int]]]:
        """Return blocks as dict.

        Returns:
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
          entity_id: Union[str, int]: Entity id.
          column_id: int: Whether entity belongs to left (0) or right (1) dataset.

        Returns:
            Blocks where entity id belongs to.
        """
        return self._grouped[column_id].get_group(entity_id).index.values.compute()

    def entity_pairs(
        self, entity_id: Union[str, int], column_id: int
    ) -> Generator[Tuple[Union[int, str], ...], None, None]:
        """Get all pairs where this entity shows up.

        Args:
          entity_id: Union[str, int]: Entity id.
          column_id: int: Whether entity belongs to left (0) or right (1) dataset.

        Returns:
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
        """Get all pairs

        Returns:
            Generator that creates all pairs, from blocks (including duplicates).
        """
        for block_tuple in self.blocks.itertuples(index=False, name=None):
            for pair in itertools.product(*block_tuple):
                yield pair

    @property
    def block_sizes(self) -> pd.DataFrame:
        """Sizes of blocks"""
        meta = pd.Series([], dtype="int64", name="block_sizes")
        return self.blocks.apply(
            lambda x: sum(len(v) for v in x), axis=1, meta=meta
        ).compute()

    @property
    def mean_block_size(self) -> float:
        """Mean size of all blocks."""
        return self.block_sizes.mean()

    @classmethod
    def combine(
        cls, this: "KlinkerBlockManager", other: "KlinkerBlockManager"
    ) -> "KlinkerBlockManager":
        """Combine blocks.

        Args:
          this: one block manager to combine
          other: other block manager to combine

        Returns:
          Combined KlinkerBlockManager

        Example:
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
          path: Union[str, pathlib.Path]: Where to write.
          **kwargs: passed to the parquet function
        """
        if not "schema" in kwargs:
            left, right = self.blocks.columns[:2]
            block_type = pa.list_(pa.string())
            if not self.blocks.index.name:
                schema = {
                    left: block_type,
                    right: block_type,
                }
            else:
                index_type = (
                    pa.string() if self.blocks.index.dtype == "O" else pa.int64()
                )
                schema = {
                    self.blocks.index.name: index_type,
                    left: block_type,
                    right: block_type,
                }
            schema = pa.schema(schema)
        else:
            schema = kwargs.pop["schema"]  # type: ignore
        self.blocks.to_parquet(path, schema=schema, **kwargs)

    @classmethod
    def read_parquet(
        cls,
        path: Union[str, pathlib.Path],
        calculate_divisions: bool = True,
        **kwargs,
    ) -> "KlinkerBlockManager":
        """Read blocks from parquet.

        Args:
          path: Union[str, pathlib.Path]: Path where blocks are stored.
          calculate_divisions: bool: Calculate index divisions.
          **kwargs: Passed to `dd.read_parquet` function.

        Returns:
            Blocks as KlinkerBlockManager
        """
        return cls(
            dd.read_parquet(
                path=path,
                calculate_divisions=calculate_divisions,
                **kwargs,
            )
        )

    @classmethod
    def from_pandas(
        cls, df: pd.DataFrame, npartitions: int = 1, **kwargs
    ) -> "KlinkerBlockManager":
        """Create from pandas.

        Args:
          df: pd.DataFrame: DataFrame
          npartitions: int:  Partitions for dask
          **kwargs: Passed to `dd.from_pandas`

        Returns:
            Blocks as KlinkerBlockManager

        Example:
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
        """

        Args:
          block_dict: Dictionary with block information.
          dataset_names: Tuple[str, str]: Tuple of dataset names.
          npartitions: int: Partitions used for dask.
          **kwargs: Passed to `dd.from_dict`.

        Returns:
            Blocks as KlinkerBlockManager

        Example:
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
            elif isinstance(res, OldKlinkerBlockManager):
                return cls.from_dict(res.blocks)
            else:
                raise ValueError(f"Unknown pickled object of type {type(res)}")
