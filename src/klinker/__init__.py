from importlib.metadata import version  # pragma: no cover
from .data import KlinkerFrame, KlinkerTripleFrame, KlinkerDataset

__all__ = ["KlinkerFrame", "KlinkerTripleFrame", "KlinkerDataset"]

__version__ = version(__package__)
