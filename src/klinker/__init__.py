from importlib.metadata import version  # pragma: no cover
from klinker.data import KlinkerFrame, KlinkerTripleFrame

__all__ = ["KlinkerFrame", "KlinkerTripleFrame"]

__version__ = version(__package__)
