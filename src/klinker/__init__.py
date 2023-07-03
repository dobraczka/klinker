from importlib.metadata import version  # pragma: no cover

from .data import KlinkerDataset, KlinkerFrame, KlinkerTriplePandasFrame, KlinkerBlockManager, KlinkerDaskFrame

__all__ = ["KlinkerFrame", "KlinkerPandasFrame", "KlinkerDaskFrame", "KlinkerPandasTripleFrame", "KlinkerDataset", "KlinkerBlockManager"]

__version__ = version(__package__)
