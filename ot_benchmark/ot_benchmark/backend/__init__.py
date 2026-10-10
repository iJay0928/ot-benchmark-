"""GPU acceleration backend abstraction layer for OT Benchmark."""

from .base import AbstractBackend, get_backend, list_available_backends
from .numpy_backend import NumPyBackend
from .torch_backend import TorchBackend

__all__ = [
    "AbstractBackend",
    "NumPyBackend",
    "TorchBackend",
    "get_backend",
    "list_available_backends",
]
