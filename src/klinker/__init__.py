from importlib.metadata import version  # pragma: no cover

from .data import KlinkerDataset, KlinkerFrame, KlinkerTripleFrame, KlinkerBlockManager

__all__ = ["KlinkerFrame", "KlinkerTripleFrame", "KlinkerDataset", "KlinkerBlockManager"]

__version__ = version(__package__)
