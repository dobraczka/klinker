from importlib.metadata import version  # pragma: no cover

from .data import KlinkerDataset, KlinkerFrame, KlinkerTripleFrame

__all__ = ["KlinkerFrame", "KlinkerTripleFrame", "KlinkerDataset"]

__version__ = version(__package__)
