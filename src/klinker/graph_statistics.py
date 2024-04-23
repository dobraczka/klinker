from klinker import KlinkerDataset
from strawman import dummy_triples
import dask.dataframe as dd
from klinker.blockers.relation_aware import reverse_rel
from klinker import KlinkerBlockManager
from .utils import concat_frames
import pandas as pd
from typing import Tuple


def get_entities(attr_triples: dd.DataFrame, rel_triples: dd.DataFrame):
    return set(attr_triples[attr_triples.columns[0]]).union(
        set(rel_triples[rel_triples.columns[0]]).union(
            set(rel_triples[rel_triples.columns[2]])
        )
    )


def get_subj_or_obj(triple_inst, idx: int):
    return triple_inst.apply(
        lambda x, idx: len(set(so[idx] for so in x)), idx=idx, meta=(None, "int64")
    )


def _create_so_set(x):
    instances = set(x[[x.columns[0], x.columns[2]]].itertuples(name=None, index=False))
    return instances


def create_instances(triples: dd.DataFrame):
    return triples.groupby(triples.columns[1]).apply(
        _create_so_set, meta=pd.Series(dtype="object")
    )


def sup_disc_f1(support, discriminability):
    return support.combine(
        discriminability, lambda supp, disc: 2 * ((supp * disc) / (supp + disc))
    )


def relation_importance(rel, num_entities):
    rel_inst = create_instances(rel)
    rel_inst_number = rel_inst.apply(len, meta=(None, "int64"))
    objects = get_subj_or_obj(rel_inst, idx=1)
    rel_support = rel_inst_number / (num_entities**2)
    rel_disc = objects / rel_inst_number
    return sup_disc_f1(rel_support, rel_disc)


def name_property(attr, num_entities):
    attr_inst = create_instances(attr)
    attr_inst_number = attr_inst.apply(len, meta=(None, "int64"))
    subjects = get_subj_or_obj(attr_inst, idx=0)
    objects = get_subj_or_obj(attr_inst, idx=1)
    attr_support = subjects / num_entities
    attr_disc = objects / attr_inst_number
    return sup_disc_f1(attr_support, attr_disc)


def filter_for_name_and_concat(group, name_importance, n):
    eid, rel_col_name, val_col_name = group.columns
    return " ".join(
        set(
            group[
                group[rel_col_name].isin(
                    name_importance.loc[group[rel_col_name]]
                    .sort_values(ascending=False)
                    .head(n)
                    .index
                )
            ][val_col_name]
        )
    )


def filter_for_name(group, name_importance, n):
    eid, rel_col_name, val_col_name = group.columns
    return group[
        group[rel_col_name].isin(
            name_importance.loc[group[rel_col_name]]
            .sort_values(ascending=False)
            .head(n)
            .index
        )
    ]


def filtered_concated(ds: KlinkerDataset, n: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    res = []
    for attr, rel in [(ds.left, ds.left_rel), (ds.right, ds.right_rel)]:
        entities = get_entities(attr, rel)
        num_entities = len(entities)
        names = name_property(attr, num_entities).compute()
        res.append(
            attr.groupby(attr.columns[0]).apply(
                filter_for_name_and_concat,
                name_importance=names,
                n=n,
                meta=pd.Series(dtype="object"),
            )
        )
    return tuple(res)


def filtered_names(ds: KlinkerDataset, n: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    res = []
    for attr, rel in [(ds.left, ds.left_rel), (ds.right, ds.right_rel)]:
        entities = get_entities(attr, rel)
        num_entities = len(entities)
        names = name_property(attr, num_entities).compute()
        res.append(
            attr.groupby(attr.columns[0]).apply(
                filter_for_name,
                name_importance=names,
                n=n,
                meta=pd.DataFrame(dtype="object", columns=["head", "relation", "tail"]),
            )
        )
    return tuple(res)


def _importance_filter(group, rei, neigh_n):
    rel_col = group.columns[1]
    rel_mask = rei.loc[group[rel_col]].sort_values(ascending=False).head(neigh_n).index
    return group[group[rel_col].isin(rel_mask)]


def filtered_concat_neighbor(
    conc_attr_frame, rel_frame, num_entities, neigh_n: int
) -> pd.DataFrame:
    rel_frame = rel_frame[
        rel_frame["relation"] != "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
    ]
    rev_rel_frame = reverse_rel(rel_frame, inverse_prefix="")
    with_inv = concat_frames([rel_frame, rev_rel_frame])

    rei = relation_importance(rel_frame, num_entities).compute()

    with_inv_filtered = with_inv.groupby("head").apply(
        _importance_filter,
        rei=rei,
        neigh_n=neigh_n,
        meta=pd.DataFrame(columns=["head", "relation", "tail"]),
    )
    res = with_inv_filtered.merge(
        conc_attr_frame, how="left", left_index=True, right_index=True
    ).dropna()

    return (
        res.reset_index(drop=True)
        .groupby(res.columns[0])
        .apply(
            lambda x: " ".join(set(x[x.columns[-1]].values)),
            meta=pd.Series(dtype="object", index=pd.Index([], dtype=str, name="head")),
        )
    )


def name_blocker(ds: KlinkerDataset, n: int):
    left_c, right_c = filtered_names(ds, n=n)
    vc = left_c["tail"].value_counts()
    left_names_cand = vc[vc == 1]
    vc2 = right_c["tail"].value_counts()
    right_names_cand = vc2[vc2 == 1]
    left_cand = left_c[left_c["tail"].isin(left_names_cand.index.compute())]
    right_cand = right_c[right_c["tail"].isin(right_names_cand.index.compute())]
    res = left_cand.merge(right_cand, left_on="tail", right_on="tail", how="inner")
    return KlinkerBlockManager.from_dict(
        {
            i: ([lb], [rb])
            for (i, lb, rb) in res[["head_x", "head_y"]].itertuples(name=None)
        },
        dataset_names=[ds.left.table_name, ds.right.table_name],
    )


def example():
    left_dummy = dummy_triples(100, relation_triples=False)
    right_dummy = dummy_triples(100, relation_triples=False)
    right_dummy[right_dummy["head"] == "e1"] = left_dummy[left_dummy["head"] == "e1"]
    right_dummy[right_dummy["head"] == "e2"] = left_dummy[left_dummy["head"] == "e2"]
    attr_left = KlinkerDaskFrame.from_dask_dataframe(
        dd.from_pandas(left_dummy, npartitions=1),
        table_name="A",
        id_col="head",
    )
    rel_left = KlinkerDaskFrame.from_dask_dataframe(
        dd.from_pandas(dummy_triples(100, relation_triples=True), npartitions=1),
        table_name="A",
        id_col="head",
    )
    attr_right = KlinkerDaskFrame.from_dask_dataframe(
        dd.from_pandas(right_dummy, npartitions=1),
        table_name="B",
        id_col="head",
    )
    rel_right = KlinkerDaskFrame.from_dask_dataframe(
        dd.from_pandas(dummy_triples(100, relation_triples=True), npartitions=1),
        table_name="B",
        id_col="head",
    )

    return KlinkerDataset(
        left=attr_left,
        right=attr_right,
        gold=pd.DataFrame([["e1", "e1"], ["e2", "e2"]], columns=["A", "B"]),
        left_rel=rel_left,
        right_rel=rel_right,
    )


def filtered_concat_neigbor_blocker(ds: KlinkerDataset):
    left_num_entities = len(get_entities(ds.left, ds.left_rel))
    right_num_entities = len(get_entities(ds.left, ds.left_rel))
    left_c, right_c = filtered_concated(ds, n=2)

    left_filtered = filtered_concat_neighbor(
        pd.DataFrame(left_c, columns=["value"]), ds.left_rel, left_num_entities, 5
    )
    right_filtered = filtered_concat_neighbor(
        pd.DataFrame(right_c, columns=["value"]), ds.right_rel, right_num_entities, 5
    )
    left_filtered.name = ds.left.table_name
    right_filtered.name = ds.right.table_name

    return TokenBlocker()._assign(left_filtered, right_filtered)


if __name__ == "__main__":
    from klinker.data import KlinkerDataset, KlinkerDaskFrame
    from klinker.blockers import TokenBlocker
    from klinker.eval import Evaluation
    from sylloge import OAEI

    ds = KlinkerDataset.from_sylloge(OAEI(), clean=True)
    # ds = KlinkerDataset.from_sylloge(OpenEA(backend="dask"), clean=True)
    blocks = name_blocker(ds, n=5)
    print(len(blocks))
    ev = Evaluation.from_dataset(blocks, ds)
    print(ev)
    print(ev.comp_with_blocking)
