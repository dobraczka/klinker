from dataclasses import dataclass

import pandas as pd
from sylloge.base import EADataset

from .enhanced_df import KlinkerFrame, KlinkerTripleFrame
from ..utils import tokenize_row


@dataclass
class KlinkerDataset:
    left: KlinkerFrame
    right: KlinkerFrame
    gold: pd.DataFrame

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
        return gold_merged[["len_intersection", "len_left", "len_right","dice_coefficient"]]

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
            left["tail"] = left["tail"].str.split(pat="\^\^", expand=True)[0]
            right["tail"] = right["tail"].str.split(pat="\^\^", expand=True)[0]
        return cls(left=left, right=right, gold=dataset.ent_links)


if __name__ == "__main__":
    from sylloge import OpenEA

    kd = KlinkerDataset.from_sylloge(OpenEA(), True)
    tdc = kd.tokenized_dice_coefficient()
    import ipdb  # noqa: autoimport

    ipdb.set_trace()  # BREAKPOINT
