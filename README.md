<p align="center">
<img src="https://github.com/dobraczka/klinker/raw/main/docs/assets/logo.png" alt="klinker logo", width=200/>
</p>
<h2 align="center"> klinker</h2>

<!-- <p align="center"> -->
<!-- <a href="https://github.com/dobraczka/klinker/actions/workflows/main.yml"><img alt="Actions Status" src="https://github.com/dobraczka/klinker/actions/workflows/main.yml/badge.svg?branch=main"></a> -->
<!-- <a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a> -->
<!-- </p> -->

Installation
============
Clone the repo and change into the directory:

```bash
git clone https://github.com/dobraczka/klinker.git
cd klinker
```

For usage with GPU create a [micromamba](https://mamba.readthedocs.io/en/latest/micromamba-installation.html) environment:

```bash
micromamba env create -n klinker-conda --file=klinker-conda.yaml
```

Activate it and install the remaining dependencies:
```
mamba activate klinker-conda
pip install -e .
```

Alternatively if you don't intend to utilize a GPU you can install it in a virtual environment:
```
python -m venv klinker-env
source klinker-env/bin/activate
pip install -e .[all]
```

or via [poetry](https://python-poetry.org/docs/):
```
poetry install
```

Usage
=====
Load a dataset:
```python
from sylloge import MovieGraphBenchmark
from klinker.data import KlinkerDataset

ds = KlinkerDataset.from_sylloge(MovieGraphBenchmark(graph_pair="tmdb-tvdb"))
```

Create blocks and write to parquet:

```python
from klinker.blockers import SimpleRelationalTokenBlocker

blocker = SimpleRelationalTokenBlocker()
blocks = blocker.assign(left=ds.left, right=ds.right, left_rel=ds.left_rel, right_rel=ds.right_rel)
blocks.to_parquet("tmdb-tvdb-tokenblocked")
```

Read blocks from parquet and evaluate:
```python
from klinker import KlinkerBlockManager
from klinker.eval_metrics import Evaluation

kbm = KlinkerBlockManager.read_parqet("tmdb-tvdb-tokenblocked")
ev = Evaluation.from_dataset(blocks=kbm, dataset=ds)
```

Reproduce Experiments
=====================

The `experiment.py` has commands for datasets and blockers. You can use `python experiment.py --help` to show the available commands. Subcommands can also offer help e.g. `python experiment.py gcn-blocker --help`.

You have to use a dataset command before a blocker command.

For example if you used micromamba for installation:
```bash
micromamba run -n klinker-conda python experiment.py movie-graph-benchmark-dataset --graph-pair "tmdb-tvdb" relational-token-blocker
```

This would be similar to the steps described in the above usage section.
