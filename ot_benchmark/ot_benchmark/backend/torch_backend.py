"""PyTorch GPU/CPU backend — primary GPU option with good Windows support."""

from __future__ import annotations

from typing import Any
import numpy as np

from .base import AbstractBackend


class TorchBackend(AbstractBackend):
    """Backend using PyTorch tensors.

    Supports CUDA (NVIDIA GPU), MPS (Apple Silicon), and CPU.
    This is the recommended GPU backend due to excellent Windows support.
    """

    def __init__(self, device: str = "auto"):
        import torch
        self._torch = torch
        if device == "auto":
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self._device = torch.device("mps")
            else:
                self._device = torch.device("cpu")
        else:
            self._device = torch.device(device)

    @property
    def name(self) -> str:
        return f"PyTorch ({self._device.type.upper()})"

    @property
    def device(self) -> Any:
        return self._device

    @staticmethod
    def available() -> bool:
        try:
            import torch
            return True
        except ImportError:
            return False

    @property
    def is_gpu(self) -> bool:
        return self._device.type in ("cuda", "mps")

    def norm(self, x: Any) -> Any:
        return self._torch.linalg.norm(x)

    def exp(self, x: Any) -> Any:
        return self._torch.exp(x)

    def dot(self, a: Any, b: Any) -> Any:
        return self._torch.matmul(a, b)

    def sum(self, x: Any, axis: int | None = None) -> Any:
        if axis is None:
            return self._torch.sum(x)
        return self._torch.sum(x, dim=axis)

    def max(self, x: Any, axis: int | None = None) -> Any:
        if axis is None:
            return self._torch.max(x)
        return self._torch.max(x, dim=axis).values

    def log(self, x: Any) -> Any:
        return self._torch.log(x)

    def abs(self, x: Any) -> Any:
        return self._torch.abs(x)

    def from_numpy(self, arr: np.ndarray) -> Any:
        return self._torch.from_numpy(arr.astype(np.float64)).to(self._device)

    def to_numpy(self, arr: Any) -> np.ndarray:
        return arr.detach().cpu().numpy()

    def zeros_like(self, x: Any) -> Any:
        return self._torch.zeros_like(x)

    def ones(self, n: int) -> Any:
        return self._torch.ones(n, dtype=self._torch.float64, device=self._device)

    def empty_cache(self) -> None:
        """Release unused GPU memory."""
        if self._device.type == "cuda":
            self._torch.cuda.empty_cache()
        elif self._device.type == "mps":
            self._torch.mps.empty_cache()

    def memory_allocated_mb(self) -> float:
        """Return currently allocated GPU memory in MB."""
        if self._device.type == "cuda":
            return float(self._torch.cuda.memory_allocated() / (1024 * 1024))
        return 0.0
