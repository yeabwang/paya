"""Body-signal decisions with choice-keyed probabilities."""

from .model import PayaModel
from .inputs import UnsupportedSignalError
from .provenance import PACKAGE_VERSION as __version__

__all__ = ["PayaModel", "UnsupportedSignalError"]
