"""Abstract backend interface for OT computations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class AbstractBackend(ABC):
    """Abstract base class for compute backends.

    Each backend provides a uniform interface for array operations
    needed by OT algorithms: norms, exponentials, matrix products,
    reductions, and conversion to/from NumPy.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend name."""
        ...

    @staticmethod
    @abstractmethod
    def available() -> bool:
        """Check whether this backend can be used in the current environment."""
        ...

    @abstractmethod
    def norm(self, x: Any) -> Any:
        """Compute Frobenius / L2 norm."""
        ...

    @abstractmethod
    def exp(self, x: Any) -> Any:
        """Element-wise exponential."""
        ...

    @abstractmethod
    def dot(self, a: Any, b: Any) -> Any:
        """Matrix or matrix-vector product."""
        ...

    @abstractmethod
    def sum(self, x: Any, axis: int | None = None) -> Any:
        """Summation along an axis."""
        ...

    @abstractmethod
    def max(self, x: Any, axis: int | None = None) -> Any:
        """Maximum along an axis."""
        ...

    @abstractmethod
    def log(self, x: Any) -> Any:
        """Element-wise natural logarithm."""
        ...

    @abstractmethod
    def abs(self, x: Any) -> Any:
        """Element-wise absolute value."""
        ...

    @abstractmethod
    def from_numpy(self, arr: np.ndarray) -> Any:
        """Convert a NumPy array to the backend's native format.

        On GPU backends this copies data to device memory.
        """
        ...

    @abstractmethod
    def to_numpy(self, arr: Any) -> np.ndarray:
        """Convert a backend array back to a NumPy array.

        On GPU backends this copies data to host memory.
        """
        ...

    @abstractmethod
    def zeros_like(self, x: Any) -> Any:
        """Create an array of zeros with the same shape as x."""
        ...

    @abstractmethod
    def ones(self, n: int) -> Any:
        """Create a 1-D array of ones of length n."""
        ...

    def __repr__(self) -> str:
        return f"<{self.name} Backend>"


def _detect_backend(preferred: str = "auto") -> AbstractBackend:
    """Auto-detect the best available backend.

    Priority: CuPy > PyTorch-CUDA > PyTorch-CPU > NumPy.
    If preferred is a specific name, try that first and fall back.
    """
    from .numpy_backend import NumPyBackend

    if preferred == "numpy":
        return NumPyBackend()

    if preferred in ("auto", "cupy"):
        try:
            from .cupy_backend import CuPyBackend
            if CuPyBackend.available():
                return CuPyBackend()
        except Exception:
            pass

    if preferred in ("auto", "torch"):
        try:
            from .torch_backend import TorchBackend
            be = TorchBackend()
            if be.available():
                return be
        except ImportError:
            pass

    if preferred not in ("auto", "numpy"):
        import warnings
        warnings.warn(f"Backend '{preferred}' not available, falling back to NumPy")

    return NumPyBackend()


_backend_instance: AbstractBackend | None = None


def get_backend(name: str = "auto") -> AbstractBackend:
    """Get or create a backend instance (singleton per session).

    Args:
        name: One of 'auto', 'numpy', 'torch', 'cupy'.
              'auto' selects the best available.

    Returns:
        An AbstractBackend instance.
    """
    global _backend_instance
    if _backend_instance is None or name != "auto":
        _backend_instance = _detect_backend(name)
    return _backend_instance


def list_available_backends() -> list[str]:
    """List all backends that are importable and usable."""
    available: list[str] = ["numpy"]
    try:
        from .torch_backend import TorchBackend
        if TorchBackend.available():
            available.append("torch")
    except ImportError:
        pass
    try:
        from .cupy_backend import CuPyBackend
        if CuPyBackend.available():
            available.append("cupy")
    except ImportError:
        pass
    return available
