from importlib.metadata import version  # pragma: no cover

from .data import (
    KlinkerBlockManager,
    KlinkerDataset,
    NNBasedKlinkerBlockManager,
    CompositeWithNNBasedKlinkerBlockManager,
)

__all__ = [
    "KlinkerDataset",
    "KlinkerBlockManager",
    "NNBasedKlinkerBlockManager",
    "CompositeWithNNBasedKlinkerBlockManager",
]

__version__ = version(__package__)
