"""One shared non-autoregressive body-signal decision model."""

from .model import PayaModel
from .inputs import UnsupportedSignalError

__all__ = ["PayaModel", "UnsupportedSignalError"]
