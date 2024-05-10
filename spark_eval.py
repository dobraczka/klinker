import click
from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F


def calc_metrics(pairs: DataFrame, gold: DataFrame):
    join_cond = (F.col("left") == F.col("gl")) & (F.col("right") == F.col("gr"))
    pairs_count = pairs.count()
    gold_count = gold.count()
    tp = pairs.join(
        gold.selectExpr("left as gl", "right as gr"), join_cond, how="inner"
    ).count()
    fp = pairs_count - tp
    fn = gold_count - tp
    print({"fp": fp, "fn": fn, "tp": tp})
    print(tp + fn)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    return {"recall": recall, "precision": precision}


@click.command()
@click.argument("gold_path", type=str)
@click.argument("blocks_path", type=str)
def evaluate(gold_path: str, blocks_path: str):
    spark = SparkSession.builder.getOrCreate()
    gold = spark.read.parquet(gold_path)
    blocks = spark.read.parquet(blocks_path)
    names = blocks.schema.names[:2]
    print(names)
    pairs = (
        blocks.select(F.explode(names[0]).alias("left"), names[1])
        .select("left", F.explode(names[1]).alias("right"))
        .distinct()
    )
    print(calc_metrics(pairs, gold))


if __name__ == "__main__":
    evaluate()
