import click
from glob import glob
from sylloge import OpenEA, OAEI
from klinker.data import KlinkerDataset
from klinker.eval import MinimalEvaluation
from klinker import KlinkerBlockManager
from experiment import prepare_dask_slurm_cluster
import logging

logger = logging.getLogger("KlinkerLaterEval")


def canonical_name_to_ds(canonical_name: str):
    if "openea" in canonical_name:
        gp1, gp2, size, version = canonical_name.replace("openea_", "").split("_")
        return OpenEA(
            graph_pair=f"{gp1.upper()}_{gp2.upper()}",
            size=size.upper(),
            version=version.upper(),
        )
    if "oaei" in canonical_name:
        task1, task2 = canonical_name.replace("oaei_", "").split("_")
        return OAEI(task=f"{task1.upper()}-{task2.upper()}")
    raise ValueError(f"Unknown canonical_name: {canonical_name}")


@click.command()
@click.argument("run-id", type=str)
@click.option("--base-path", type=str, default="experiment_artifacts")
@click.option("--use-cluster", type=bool, default=False)
@click.option("--num-cores", type=int, default=1)
@click.option("--memory", type=str, default="8GB")
@click.option("--walltime", type=str, default="01:00:00")
@click.option("--num-clusters", type=int, default=2)
@click.option("--local-directory", type=str, default=None)
@click.option("--partition-size", type=str, default="100MB")
def run_later_eval(
    run_id: str,
    base_path: str,
    use_cluster: bool,
    num_cores: int,
    memory: str,
    walltime: str,
    num_clusters: int,
    local_directory: str,
    partition_size: str,
):
    possible_files = list(glob(f"{base_path}/*/*/{run_id}_blocks.parquet"))
    if len(possible_files) > 1:
        raise ValueError(f"Found multiple candidates {possible_files}")
    block_file = possible_files[0]
    _, ds_name, blocker_name, _ = block_file.split("/")
    logger.info(ds_name)
    logger.info(blocker_name)
    if use_cluster:
        prepare_dask_slurm_cluster(
            num_cores=num_cores,
            memory=memory,
            walltime=walltime,
            num_clusters=num_clusters,
            local_directory=local_directory,
        )
    ds = KlinkerDataset.from_sylloge(canonical_name_to_ds(ds_name))
    blocks = KlinkerBlockManager.read_parquet(block_file, partition_size=partition_size)
    ev_res = MinimalEvaluation(blocks=blocks, dataset=ds).to_dict()
    for m_name, m_val in ev_res.items():
        logger.info(f"{m_name}:{m_val}")


if __name__ == "__main__":
    run_later_eval()
