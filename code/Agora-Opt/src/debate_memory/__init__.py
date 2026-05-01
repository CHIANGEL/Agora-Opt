"""Debate-with-memory v2 core package."""

from importlib import metadata

try:
    __version__ = metadata.version("debate-memory")
except metadata.PackageNotFoundError:  # pragma: no cover - local usage
    __version__ = "0.0.0"

__all__ = ["__version__"]

