from collections import OrderedDict
from typing import Any, Dict, List, Set, Tuple, Union, Optional
import dask.dataframe as dd

import numpy as np
import pandas as pd

from . import KlinkerBlockManager, KlinkerDataset, NNBasedKlinkerBlockManager


def harmonic_mean(a: float, b: float) -> float:
    """Calculate harmonic mean between a and b."""
    if a + b == 0:
        return 0
    return 2 * ((a * b) / (a + b))


class MinimalEvaluation:
    @staticmethod
    def _check_consistency(blocks: KlinkerBlockManager, gold: pd.DataFrame):
        if isinstance(blocks, NNBasedKlinkerBlockManager):
            return
        if not len(gold.columns) == 2:
            raise ValueError("Only binary matching supported!")
        if not set(blocks.blocks.columns) == set(gold.columns):
            raise ValueError(
                "Blocks and gold standard frame need to have the same columns!"
            )

    def __init__(self, blocks: KlinkerBlockManager, dataset: KlinkerDataset):
        gold = dataset.gold
        MinimalEvaluation._check_consistency(blocks, gold)
        block_df = blocks.blocks
        left_col, right_col = block_df.columns

        # mean pairs per block, other calc in KlinkerBlockManager is wrong!
        # mean_block_size = block_pairs.groupby(block_pairs.index.name).size().mean().compute()
        # pairs_count = block_df.apply(
        #     lambda x, col1, col2: sum(1 for _ in itertools.product(x[col1], x[col2])),
        #     axis=1,
        #     col1=left_col,
        #     col2=right_col,
        #     meta=(None, "int64"),
        # ).sum().compute()
        gold_count = len(gold)

        left = block_df[left_col].explode().to_frame()
        right = block_df[right_col].explode().to_frame()
        pairs_count = len(left.join(right).drop_duplicates())

        gold_dd = dd.from_pandas(gold, npartitions=1)
        actual_suffix = "_actual"
        gold_suffix = "_gold"
        bla = (
            left.join(right)
            .drop_duplicates()
            .merge(
                gold_dd,
                left_on=left_col,
                right_on=left_col,
                suffixes=[actual_suffix, gold_suffix],
            )
        )
        tp = (
            (bla[right_col + actual_suffix] == bla[right_col + gold_suffix])
            .sum()
            .compute()
        )
        print("tp=%s" % (tp))
        print("pairs_count=%s" % (pairs_count))
        print("gold_count=%s" % (gold_count))

        # block_pairs = left.join(right, how="inner").drop_duplicates()
        # tp = block_pairs.merge(
        #     dd.from_pandas(gold, npartitions=1),
        #     left_on=list(block_df.columns),
        #     right_on=list(gold.columns),
        #     how="inner",
        # ).index.size.compute()

        fp = pairs_count - tp
        fn = gold_count - tp
        self.recall = tp / (tp + fn)
        self.precision = tp / (tp + fp)
        self.f_measure = harmonic_mean(self.recall, self.precision)

    def to_dict(self) -> Dict[str, float]:
        return {
            "pairs_quality": self.precision,
            "precision": self.precision,
            "recall": self.recall,
            "pairs_completeness": self.recall,
            "f_measure": self.f_measure,
        }

    def __repr__(self) -> str:
        return f"MinimalEvaluation: {self.to_dict()}"


class Evaluation:
    """Class used for evaluation."""

    def __init__(
        self,
        *,
        true_positive_set: Set[Tuple[Any, Any]],
        gold_pair_set: Set[Tuple[Any, Any]],
        false_positive: int,
        comp_with_blocking: int,
        comp_without_blocking: int,
        mean_block_size: float,
        false_positive_set: Optional[Set[Tuple[Any, Any]]] = None,
    ):
        self.gold_pair_set = gold_pair_set
        self.comp_without_blocking = comp_without_blocking
        self.mean_block_size = mean_block_size
        self.tp_set = true_positive_set
        self.fn_set = self.gold_pair_set - self.tp_set  # type: ignore
        self.false_negative = len(self.fn_set)
        self.true_positive = len(self.tp_set)
        self.false_positive = false_positive
        self.comp_with_blocking = comp_with_blocking
        self.false_positive_set = false_positive_set

    @staticmethod
    def _check_consistency(blocks: KlinkerBlockManager, gold: pd.DataFrame):
        if isinstance(blocks, NNBasedKlinkerBlockManager):
            return
        if not len(gold.columns) == 2:
            raise ValueError("Only binary matching supported!")
        if not set(blocks.blocks.columns) == set(gold.columns):
            raise ValueError(
                "Blocks and gold standard frame need to have the same columns!"
            )

    @classmethod
    def from_blocks_and_gold(
        cls,
        blocks: KlinkerBlockManager,
        gold: pd.DataFrame,
        left_data_len: int,
        right_data_len: int,
        keep_false_positive_set: bool = False,
    ):
        Evaluation._check_consistency(blocks, gold)

        left_col = gold.columns[0]
        right_col = gold.columns[1]

        gold_pair_set = set(zip(gold[left_col], gold[right_col]))
        tp_pairs: Set[Tuple[Any, Any]] = set()
        fp = 0
        fp_set: Optional[Set[Tuple[Any, Any]]] = (
            set() if keep_false_positive_set else None
        )
        _pair_number = 0  # in case no pairs exist
        for _pair_number, pair in enumerate(blocks.all_pairs(), start=1):
            if pair in gold_pair_set:
                left, right = pair  # for mypy
                tp_pairs.add((left, right))
            else:
                fp += 1
                if keep_false_positive_set:
                    assert fp_set
                    fp_set.add((left, right))
        comp_without_blocking = left_data_len * right_data_len
        return cls(
            true_positive_set=tp_pairs,
            gold_pair_set=gold_pair_set,
            false_positive=fp,
            comp_with_blocking=_pair_number,
            comp_without_blocking=comp_without_blocking,
            mean_block_size=blocks.mean_block_size,
            false_positive_set=fp_set,
        )

    @classmethod
    def from_dataset(
        cls,
        blocks: KlinkerBlockManager,
        dataset: KlinkerDataset,
        keep_false_positive_set: bool = False,
    ) -> "Evaluation":
        """Helper function to initialise evaluation with dataset.

        Args:
        ----
          blocks: KlinkerBlockManager: Calculated blocks
          dataset: KlinkerDataset: Dataset that was used for blocking
          keep_false_positive_set: Whether to keep false positive

        Returns:
        -------
            eval instance

        Examples:
        --------
            >>> # doctest: +SKIP
            >>> from sylloge import MovieGraphBenchmark
            >>> from klinker.data import KlinkerDataset
            >>> ds = KlinkerDataset.from_sylloge(MovieGraphBenchmark(),clean=True)
            >>> from klinker.blockers import TokenBlocker
            >>> blocks = TokenBlocker().assign(left=ds.left, right=ds.right)
            >>> from klinker.eval import Evaluation
            >>> ev = Evaluation.from_dataset(blocks, ds)
            >>> ev.to_dict()
            {'recall': 0.993933265925177, 'precision': 0.002804877004859314, 'f_measure': 0.005593967847488974, 'reduction_ratio': 0.9985747694185365, 'h3r': 0.9962486115318822, 'mean_block_size': 10.160596863935256}

        """
        return cls.from_blocks_and_gold(
            blocks=blocks,
            gold=dataset.gold,
            left_data_len=len(dataset.left),
            right_data_len=len(dataset.right),
            keep_false_positive_set=keep_false_positive_set,
        )

    @classmethod
    def from_joined_evals(cls, eval_a: "Evaluation", eval_b: "Evaluation"):
        if (
            eval_a.gold_pair_set != eval_b.gold_pair_set
            or eval_a.comp_without_blocking != eval_b.comp_without_blocking
        ):
            raise ValueError("Can only join on identical datasets!")
        joined_tp_set = eval_a.tp_set.union(eval_b.tp_set)
        joined_comp_with_blocking = (
            eval_a.comp_with_blocking + eval_b.comp_with_blocking
        )
        joined_fp = eval_a.false_positive + eval_b.false_positive
        return cls(
            true_positive_set=joined_tp_set,
            gold_pair_set=eval_a.gold_pair_set,
            false_positive=joined_fp,
            comp_with_blocking=joined_comp_with_blocking,
            comp_without_blocking=eval_a.comp_without_blocking,
            mean_block_size=np.nan,
        )

    @property
    def pairs_completeness(self) -> float:
        return self.true_positive / len(self.gold_pair_set)

    @property
    def recall(self) -> float:
        denom = self.true_positive + self.false_negative
        if denom == 0:
            return 0
        return self.true_positive / denom

    @property
    def precision(self) -> float:
        denom = self.true_positive + self.false_positive
        if denom == 0:
            return 0
        return self.true_positive / denom

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
            "pairs_completeness": self.pairs_completeness,
        }


def dice(a: Set, b: Set) -> float:
    """Calculate Soerensen-Dice Coefficient."""
    return (2 * len(a.intersection(b))) / (len(a) + len(b))


def compare_blocks_from_eval(
    blocks_a: KlinkerBlockManager,
    blocks_b: KlinkerBlockManager,
    eval_a: Evaluation,
    eval_b: Evaluation,
    dataset: KlinkerDataset,
    improvement_metrics: Union[str, List[str]] = "h3r",
) -> Dict:
    """Compare similarity between blocks using calculated eval.

    Args:
    ----
      blocks_a: KlinkerBlockManager: one blocking result
      blocks_b: KlinkerBlockManager: other blocking result
      eval_a: Evaluation: eval of a
      eval_b: Evaluation: eval of b
      dataset: KlinkerDataset: dataset from which blocks where calculated
      improvement_metric: Union[str, List[str]]: Metric(s) used for calculating improvement

    Returns:
    -------
        Dictionary with improvement metrics.
    """
    if isinstance(improvement_metrics, str):
        improvement_metrics = [improvement_metrics]

    def percent_improvement(new: float, old: float):
        return (new - old) / old

    dice_tp = dice(eval_a.tp_set, eval_b.tp_set)
    eval_both = Evaluation.from_joined_evals(eval_a, eval_b)
    result_dict = {
        "eval_a": eval_a,
        "eval_b": eval_b,
        "dice_tp": dice_tp,
        "eval_both": eval_both,
    }

    for im in improvement_metrics:
        eval_both_metric = eval_both.to_dict()[im]
        result_dict[f"improvement_a_{im}"] = percent_improvement(
            eval_both_metric, eval_a.to_dict()[im]
        )
        result_dict[f"improvement_b_{im}"] = percent_improvement(
            eval_both_metric, eval_b.to_dict()[im]
        )
    return result_dict


def compare_blocks(
    blocks_a: KlinkerBlockManager,
    blocks_b: KlinkerBlockManager,
    dataset: KlinkerDataset,
    improvement_metrics: Union[str, List[str]] = "h3r",
) -> Dict:
    """Compare similarity between blocks using calculated eval.

    Args:
    ----
      blocks_a: KlinkerBlockManager: one blocking result
      blocks_b: KlinkerBlockManager: other blocking result
      dataset: KlinkerDataset: dataset from which blocks where calculated
      improvement_metric: Union[str, List[str]]: Metric(s) used for calculating improvement

    Returns:
    -------
        Dictionary with improvement metrics.
    """
    eval_a = Evaluation.from_dataset(blocks=blocks_a, dataset=dataset)
    eval_b = Evaluation.from_dataset(blocks=blocks_b, dataset=dataset)
    return compare_blocks_from_eval(
        blocks_a=blocks_a,
        blocks_b=blocks_b,
        eval_a=eval_a,
        eval_b=eval_b,
        dataset=dataset,
        improvement_metrics=improvement_metrics,
    )


def multiple_block_comparison_from_eval(
    blocks_with_eval: Dict[str, Tuple[KlinkerBlockManager, Evaluation]],
    dataset: KlinkerDataset,
    improvement_metrics: Union[str, List[str]] = "h3r",
) -> pd.DataFrame:
    """Compare multiple blocking strategies.

    Args:
    ----
      blocks_with_eval: Dict[str, Tuple[KlinkerBlockManager, Evaluation]]: Blocking results and Evaluations
      dataset: KlinkerDataset: Dataset that was used for blocking
      improvement_metric: Union[str, List[str]]: Metric(s) used for calculating improvement

    Returns:
    -------
        DataFrame with improvement values.
    """
    result = []
    seen_pairs = set()
    for b_a_name, (blocks_a, eval_a) in blocks_with_eval.items():
        for b_b_name, (blocks_b, eval_b) in blocks_with_eval.items():
            if (
                b_a_name != b_b_name
                and (b_a_name, b_b_name) not in seen_pairs
                and (
                    b_b_name,
                    b_a_name,
                )
                not in seen_pairs
            ):
                comparison = compare_blocks_from_eval(
                    blocks_a,
                    blocks_b,
                    eval_a,
                    eval_b,
                    dataset,
                    improvement_metrics=improvement_metrics,
                )
                comparison_a = [
                    comparison[f"improvement_a_{im}"] for im in improvement_metrics
                ]
                comparison_b = [
                    comparison[f"improvement_b_{im}"] for im in improvement_metrics
                ]
                result.append(
                    [
                        b_a_name,
                        b_b_name,
                        *comparison_a,
                        comparison["dice_tp"],
                    ]
                )
                result.append(
                    [
                        b_b_name,
                        b_a_name,
                        *comparison_b,
                        comparison["dice_tp"],
                    ]
                )
                seen_pairs.add((b_a_name, b_b_name))
    im_cols = [f"improvement_{im}" for im in improvement_metrics]
    return pd.DataFrame(result, columns=["base", "other", *im_cols, "dice_tp"])


def multiple_block_comparison(
    blocks: Dict[str, KlinkerBlockManager],
    dataset: KlinkerDataset,
    improvement_metrics: Union[str, List[str]] = "h3r",
) -> pd.DataFrame:
    """Compare multiple blocking strategies.

    Args:
    ----
      blocks: Dict[str, KlinkerBlockManager]: Blocking results
      dataset: KlinkerDataset: Dataset that was used for blocking
      improvement_metric: Union[str, List[str]]: Metric(s) used for calculating improvement

    Returns:
    -------
        DataFrame with improvement values.
    """
    blocks_with_eval = OrderedDict(
        {
            name: (
                blk,
                Evaluation.from_dataset(blocks=blk, dataset=dataset),
            )
            for name, blk in blocks.items()
        }
    )
    return multiple_block_comparison_from_eval(
        blocks_with_eval, dataset, improvement_metrics=improvement_metrics
    )


if __name__ == "__main__":
    from klinker.data import KlinkerDataset
    from sylloge import OpenEA

    ds = KlinkerDataset.from_sylloge(OpenEA())
    blocks = KlinkerBlockManager.read_parquet(
        "experiment_artifacts/openea_d_w_15k_v1/TokenBlocker/b9869e9da6c61d3a99db6fb217db7256837380e6_blocks.parquet",
        partition_size="100MB",
    )
    print(Evaluation.from_dataset(blocks, ds).to_dict())
    print(MinimalEvaluation(blocks, ds).to_dict())
