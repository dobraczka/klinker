from typing import Dict

import pandas as pd

from . import KlinkerBlockManager, KlinkerDataset


def harmonic_mean(a: float, b: float) -> float:
    if a + b == 0:
        return 0
    return 2 * ((a * b) / (a + b))


class Evaluation:
    def __init__(
        self,
        blocks: KlinkerBlockManager,
        gold: pd.DataFrame,
        left_data_len: int,
        right_data_len: int,
    ):
        self._check_consistency(blocks, gold)

        left_col = gold.columns[0]
        right_col = gold.columns[1]

        self.block_pair_set = set(blocks.to_pairs())
        self.gold_pair_set = set(zip(gold[left_col], gold[right_col]))

        self.comp_without_blocking = left_data_len * right_data_len
        self.comp_with_blocking = len(self.block_pair_set)
        self._calc_tp_fp_fn()
        self.mean_block_size = blocks.mean_block_size

    def _calc_tp_fp_fn(self):
        self.true_positive = len(self.block_pair_set.intersection(self.gold_pair_set))
        self.false_negative = len(self.gold_pair_set - self.block_pair_set)
        self.false_positive = len(self.block_pair_set - self.gold_pair_set)

    def _check_consistency(self, blocks: KlinkerBlockManager, gold: pd.DataFrame):
        if not len(gold.columns) == 2:
            raise ValueError("Only binary matching supported!")
        if not set(blocks.dataset_names) == set(gold.columns):
            raise ValueError(
                "Blocks and gold standard frame need to have the same columns!"
            )

    @classmethod
    def from_dataset(
        cls, blocks: KlinkerBlockManager, dataset: KlinkerDataset
    ) -> "Evaluation":
        return cls(
            blocks=blocks,
            gold=dataset.gold,
            left_data_len=len(dataset.left),
            right_data_len=len(dataset.right),
        )

    @property
    def recall(self) -> float:
        return self.true_positive / (self.true_positive + self.false_negative)

    @property
    def precision(self) -> float:
        return self.true_positive / (self.true_positive + self.false_positive)

    @property
    def f_measure(self) -> float:
        rec = self.recall
        prec = self.precision
        return harmonic_mean(a=rec, b=prec)

    @property
    def reduction_ratio(self) -> float:
        return 1 - (self.comp_with_blocking / self.comp_without_blocking)

    @property
    def h3r(self) -> float:
        rr = self.reduction_ratio
        rec = self.recall
        return harmonic_mean(a=rr, b=rec)

    def __repr__(self) -> str:
        return f"Evaluation: {self.to_dict()}"

    def to_dict(self) -> Dict[str, float]:
        return {
            "recall": self.recall,
            "precision": self.precision,
            "f_measure": self.f_measure,
            "reduction_ratio": self.reduction_ratio,
            "h3r": self.h3r,
            "mean_block_size": self.mean_block_size,
        }
