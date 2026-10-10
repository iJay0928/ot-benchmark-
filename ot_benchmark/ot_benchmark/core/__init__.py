"""Core infrastructure for OT Benchmark system."""

from .exceptions import (
    OTBenchmarkError,
    BackendNotAvailableError,
    AlgorithmFailedError,
    ConfigValidationError,
    DataGenerationError,
)
from .logging_config import setup_logging, get_logger
from .types import (
    Distribution,
    CostMatrix,
    SparseMatrix,
    AlgorithmFunc,
    BackendProtocol,
)

__all__ = [
    "OTBenchmarkError",
    "BackendNotAvailableError",
    "AlgorithmFailedError",
    "ConfigValidationError",
    "DataGenerationError",
    "setup_logging",
    "get_logger",
    "Distribution",
    "CostMatrix",
    "SparseMatrix",
    "AlgorithmFunc",
    "BackendProtocol",
]
