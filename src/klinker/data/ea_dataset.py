from dataclasses import dataclass
from typing import Optional

import pandas as pd
from sylloge.base import EADataset

from .enhanced_df import KlinkerFrame, KlinkerTripleFrame
from ..typing import Side, Tuple
from ..utils import tokenize_row


@dataclass
class KlinkerDataset:
    left: KlinkerFrame
    right: KlinkerFrame
    gold: pd.DataFrame
    left_rel: Optional[pd.DataFrame] = None
    right_rel: Optional[pd.DataFrame] = None

    def tokenized_dice_coefficient(self, min_token_length: int = 3) -> pd.DataFrame:
        def tokenize_values(kf: KlinkerFrame, min_token_length: int) -> pd.Series:
            tok_col = "merged"
            conc_kf = kf.concat_values(new_column_name=tok_col)
            conc_kf[tok_col] = conc_kf[conc_kf.non_id_columns].apply(
                tokenize_row, axis=1, min_token_length=min_token_length
            )
            return conc_kf

        def tok_statistics(x: pd.Series):
            left_set = set(x["merged_left"])
            right_set = set(x["merged_right"])
            len_int = len(left_set.intersection(right_set))
            len_left = len(left_set)
            len_right = len(right_set)
            x["len_intersection"] = len_int
            x["len_left"] = len_left
            x["len_right"] = len_right
            x["dice_coefficient"] = (2 * len_int) / (len_left + len_right)
            return x

        left_tok = tokenize_values(self.left, min_token_length)
        right_tok = tokenize_values(self.right, min_token_length)

        gold_merged = left_tok.merge(
            self.gold, how="inner", left_on="id", right_on="left"
        ).merge(
            right_tok,
            how="inner",
            left_on="right",
            right_on="id",
            suffixes=["_left", "_right"],
        )
        gold_merged = gold_merged.apply(tok_statistics, axis=1)
        return gold_merged[
            ["len_intersection", "len_left", "len_right", "dice_coefficient"]
        ]

    @classmethod
    def from_sylloge(cls, dataset: EADataset, clean: bool = False) -> "KlinkerDataset":
        left = KlinkerTripleFrame.from_df(
            dataset.attr_triples_left, name="left", id_col="head"
        )
        right = KlinkerTripleFrame.from_df(
            dataset.attr_triples_right, name="right", id_col="head"
        )
        if clean:
            # remove datatype
            left["tail"] = left["tail"].str.split(pat=r"\^\^", expand=True)[0]
            right["tail"] = right["tail"].str.split(pat=r"\^\^", expand=True)[0]

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
