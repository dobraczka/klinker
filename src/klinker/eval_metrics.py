from collections import OrderedDict
from itertools import chain
from typing import Dict, Optional, Set, Tuple, Union

import pandas as pd
from tqdm import tqdm

from . import KlinkerBlockManager, KlinkerDataset


def harmonic_mean(a: float, b: float) -> float:
    if a + b == 0:
        return 0
    return 2 * ((a * b) / (a + b))


def calc_tp_fp_comp_with_blocking(
    blocks: KlinkerBlockManager, gold_pair_set: Set[Tuple[str, str]], save_pairs: bool
):
    tp = 0
    fp = 0
    pair_number = 0
    tp_set: Set[Tuple[Union[int, str], ...]] = set()
    for pair_number, pair in enumerate(blocks.to_pairs(), start=1):
        if pair in gold_pair_set:
            tp += 1
            if save_pairs:
                tp_set.add(pair)
        else:
            fp += 1
    comp_with_blocking = pair_number
    return tp, fp, comp_with_blocking, tp_set


def sum_tuple(x):
    return tuple(
        map(
            lambda x: sum(x) if isinstance(x[0], int) else chain.from_iterable(x),
            zip(*x),
        )
    )


class Evaluation:
    def __init__(
        self,
        blocks: KlinkerBlockManager,
        gold: pd.DataFrame,
        left_data_len: int,
        right_data_len: int,
        partition_size: Optional[int] = None,
        save_pairs: bool = False,
    ):
        self._check_consistency(blocks, gold)

        left_col = gold.columns[0]
        right_col = gold.columns[1]

        self.gold_pair_set = set(zip(gold[left_col], gold[right_col]))
        self._calc_tp_fp_fn(blocks)

        self.comp_without_blocking = left_data_len * right_data_len
        # self.true_positive, self.false_positive, self.comp_with_blocking, tp_it = (
        #     blocks.to_bag(partition_size=partition_size)
        #     .map_partitions(
        #         calc_tp_fp_comp_with_blocking, self.gold_pair_set, save_pairs
        #     )
        #     .reduction(lambda x: x, sum_tuple)
        #     .compute()
        # )
        # self.tp_set = set(tp_it)
        # self.false_negative = len(self.gold_pair_set) - self.true_positive
        self.mean_block_size = blocks.mean_block_size

    def _calc_tp_fp_fn(self, blocks: KlinkerBlockManager):
        tp_pairs = set()
        fp = 0
        for pair_number, pair in enumerate(blocks.all_pairs(), start=1):
            if pair in self.gold_pair_set:
                tp_pairs.add(pair)
            else:
                fp += 1
        tp = len(tp_pairs)
        self.false_negative = len(self.gold_pair_set) - tp
        self.true_positive = tp
        self.false_positive = fp
        self.comp_with_blocking = pair_number

    def _check_consistency(self, blocks: KlinkerBlockManager, gold: pd.DataFrame):
        if not len(gold.columns) == 2:
            raise ValueError("Only binary matching supported!")
        if not set(blocks.blocks.columns) == set(gold.columns):
            raise ValueError(
                "Blocks and gold standard frame need to have the same columns!"
            )

    @classmethod
    def from_dataset(
        cls,
        blocks: KlinkerBlockManager,
        dataset: KlinkerDataset,
        partition_size: Optional[int] = None,
        save_pairs: bool = False,
    ) -> "Evaluation":
        return cls(
            blocks=blocks,
            gold=dataset.gold,
            left_data_len=len(dataset.left),
            right_data_len=len(dataset.right),
            partition_size=partition_size,
            save_pairs=save_pairs,
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


def dice(a: Set, b: Set) -> float:
    return (2 * len(a.intersection(b))) / (len(a) + len(b))


def compare_blocks_from_eval(
    blocks_a: KlinkerBlockManager,
    blocks_b: KlinkerBlockManager,
    eval_a: Evaluation,
    eval_b: Evaluation,
    dataset: KlinkerDataset,
    improvement_metric: str = "h3r",
) -> Dict:
    def percent_improvement(new: float, old: float):
        return (new - old) / old

    blocks_both = KlinkerBlockManager.combine(blocks_a, blocks_b)
    dice_tp = dice(eval_a.tp_set, eval_b.tp_set)

    eval_both = Evaluation.from_dataset(
        blocks=blocks_both, dataset=dataset, save_pairs=True, partition_size=500
    )
    eval_both_metric = eval_both.to_dict()[improvement_metric]
    improvement_a = percent_improvement(
        eval_both_metric, eval_a.to_dict()[improvement_metric]
    )
    improvement_b = percent_improvement(
        eval_both_metric, eval_b.to_dict()[improvement_metric]
    )
    return {
        "eval_a": eval_a,
        "eval_b": eval_b,
        "dice_tp": dice_tp,
        "eval_both": eval_both,
        "improvement_a": improvement_a,
        "improvement_b": improvement_b,
    }


def compare_blocks(
    blocks_a: KlinkerBlockManager,
    blocks_b: KlinkerBlockManager,
    dataset: KlinkerDataset,
    improvement_metric: str = "h3r",
) -> Dict:
    eval_a = Evaluation.from_dataset(
        blocks=blocks_a, dataset=dataset, save_pairs=True, partition_size=500
    )
    eval_b = Evaluation.from_dataset(
        blocks=blocks_b, dataset=dataset, save_pairs=True, partition_size=500
    )
    return compare_blocks_from_eval(
        blocks_a=blocks_a,
        blocks_b=blocks_b,
        eval_a=eval_a,
        eval_b=eval_b,
        dataset=dataset,
        improvement_metric=improvement_metric,
    )


def multiple_block_comparison(
    blocks: Dict[str, KlinkerBlockManager],
    dataset: KlinkerDataset,
    improvement_metric: str = "h3r",
) -> pd.DataFrame:
    blocks_with_eval = OrderedDict(
        {
            name: (
                blk,
                Evaluation.from_dataset(
                    blocks=blk, dataset=dataset, save_pairs=True, partition_size=500
                ),
            )
            for name, blk in blocks.items()
        }
    )
    result = []
    seen_pairs = set()
    for (b_a_name, (blocks_a, eval_a)) in blocks_with_eval.items():
        for (b_b_name, (blocks_b, eval_b)) in blocks_with_eval.items():
            if (
                b_a_name != b_b_name
                and (b_a_name, b_b_name) not in seen_pairs
                and (
                    b_b_name,
                    b_a_name,
                )
                not in seen_pairs
            ):
                print(b_a_name, b_b_name)
                comparison = compare_blocks_from_eval(
                    blocks_a, blocks_b, eval_a, eval_b, dataset, "h3r"
                )
                result.append(
                    [
                        b_a_name,
                        b_b_name,
                        comparison["improvement_a"],
                        comparison["dice_tp"],
                    ]
                )
                result.append(
                    [
                        b_b_name,
                        b_a_name,
                        comparison["improvement_b"],
                        comparison["dice_tp"],
                    ]
                )
    result_df = pd.DataFrame(
        result, columns=["base", "other", "improvement", "dice_tp"]
    )
    seen_pairs.add((b_a_name, b_b_name))
    return result_df
