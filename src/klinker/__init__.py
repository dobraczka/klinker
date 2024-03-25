from importlib.metadata import version  # pragma: no cover

from .data import (
    KlinkerBlockManager,
    KlinkerDaskFrame,
    KlinkerDataset,
    KlinkerFrame,
    NNBasedKlinkerBlockManager,
)

__all__ = [
    "KlinkerFrame",
    "KlinkerPandasFrame",
    "KlinkerDaskFrame",
    "KlinkerPandasTripleFrame",
    "KlinkerDataset",
    "KlinkerBlockManager",
    "NNBasedKlinkerBlockManager",
]

__version__ = version(__package__)
