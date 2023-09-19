from importlib.metadata import version  # pragma: no cover

from .data import (
    KlinkerBlockManager,
    KlinkerDaskFrame,
    KlinkerDataset,
    KlinkerFrame,
    KlinkerTriplePandasFrame,
)

__all__ = [
    "KlinkerFrame",
    "KlinkerPandasFrame",
    "KlinkerDaskFrame",
    "KlinkerPandasTripleFrame",
    "KlinkerDataset",
    "KlinkerBlockManager",
]

__version__ = version(__package__)
