"""CuPy GPU backend — high-performance NumPy-compatible GPU arrays.

Note: CuPy has limited Windows support. On Windows, prefer TorchBackend.
"""

from __future__ import annotations

from typing import Any
import numpy as np

from .base import AbstractBackend


class CuPyBackend(AbstractBackend):
    """Backend using CuPy for NVIDIA GPU acceleration.

    Provides a nearly drop-in NumPy replacement on CUDA devices.
    """

    def __init__(self, device_id: int = 0):
        import cupy as cp
        self._xp = cp
        self._device_id = device_id
        cp.cuda.Device(device_id).use()

    @property
    def name(self) -> str:
        return f"CuPy (GPU:{self._device_id})"

    @staticmethod
    def available() -> bool:
        try:
            import cupy as _cp
            _cp.cuda.runtime.getDeviceCount()
            return _cp.cuda.runtime.getDeviceCount() > 0
        except Exception:
            return False

    def norm(self, x: Any) -> Any:
        return self._xp.linalg.norm(x)

    def exp(self, x: Any) -> Any:
        return self._xp.exp(x)

    def dot(self, a: Any, b: Any) -> Any:
        return self._xp.dot(a, b)

    def sum(self, x: Any, axis: int | None = None) -> Any:
        return self._xp.sum(x, axis=axis)

    def max(self, x: Any, axis: int | None = None) -> Any:
        return self._xp.max(x, axis=axis)

    def log(self, x: Any) -> Any:
        return self._xp.log(x)

    def abs(self, x: Any) -> Any:
        return self._xp.abs(x)

    def from_numpy(self, arr: np.ndarray) -> Any:
        return self._xp.asarray(arr, dtype=self._xp.float64)

    def to_numpy(self, arr: Any) -> np.ndarray:
        return self._xp.asnumpy(arr)

    def zeros_like(self, x: Any) -> Any:
        return self._xp.zeros_like(x)

    def ones(self, n: int) -> Any:
        return self._xp.ones(n, dtype=self._xp.float64)

    def empty_cache(self) -> None:
        """Release all unused GPU memory blocks."""
        mempool = self._xp.get_default_memory_pool()
        mempool.free_all_blocks()

    def memory_allocated_mb(self) -> float:
        """Return currently allocated GPU memory in MB."""
        mempool = self._xp.get_default_memory_pool()
        return float(mempool.used_bytes() / (1024 * 1024))
