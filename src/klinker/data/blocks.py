import itertools
from typing import (
    Dict,
    Generator,
    ItemsView,
    KeysView,
    Optional,
    Set,
    Tuple,
    Union,
    ValuesView,
)

import dask.dataframe as dd
import pandas as pd


class KlinkerBlockManager:
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
                        if not ent_id in id_mappings[i]:
                            raise ValueError(
                                "If id_mappings are supplied, they have to map all ids"
                            )

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

    def to_pairs(
        self, replace_id_mappings=False
    ) -> Generator[Tuple[Union[int, str], ...], None, None]:
        for block_name, block in self.blocks.items():
            for pair in itertools.product(*block):
                if replace_id_mappings and self.id_mappings is not None:
                    yield tuple(self.id_mappings[i][pair[i]] for i in range(len(pair)))
                else:
                    yield pair

    @classmethod
    def combine(
        cls, this: "KlinkerBlockManager", other: "KlinkerBlockManager"
    ) -> "KlinkerBlockManager":
        """Combine blocks and id_mappings with preferenc of `this` id_mappings.

        :param this: one block manager to combine
        :param other: other block manager to combine
        :return: Combined KlinkerBlockManager
        """
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
        return self.blocks[key]

    def __setitem__(self, key, value):
        self.blocks[key] = value

    def copy(self) -> "KlinkerBlockManager":
        return KlinkerBlockManager(
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
        pd_blocks: Union[pd.DataFrame, dd.DataFrame],
        id_mappings: Optional[Tuple[Dict[int, str], ...]] = None,
    ) -> "KlinkerBlockManager":
        def _ensure_set(value) -> Set:
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
