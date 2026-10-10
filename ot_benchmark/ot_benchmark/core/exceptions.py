"""Project-specific exception hierarchy."""


class OTBenchmarkError(Exception):
    """Base exception for all OT Benchmark errors."""
    pass


class BackendNotAvailableError(OTBenchmarkError):
    """Raised when a requested compute backend is not available."""

    def __init__(self, backend_name: str, reason: str = ""):
        msg = f"Backend '{backend_name}' is not available"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)
        self.backend_name = backend_name


class AlgorithmFailedError(OTBenchmarkError):
    """Raised when an OT algorithm fails during execution."""

    def __init__(self, algorithm_name: str, dataset_name: str, reason: str = ""):
        msg = f"Algorithm '{algorithm_name}' failed on dataset '{dataset_name}'"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)
        self.algorithm_name = algorithm_name
        self.dataset_name = dataset_name


class ConfigValidationError(OTBenchmarkError):
    """Raised when configuration is invalid."""
    pass


class DataGenerationError(OTBenchmarkError):
    """Raised when synthetic data generation fails."""
    pass


class CloudConnectionError(OTBenchmarkError):
    """Raised when cloud service connection fails."""
    pass


class SparseFormatError(OTBenchmarkError):
    """Raised when sparse matrix operations fail."""
    pass
