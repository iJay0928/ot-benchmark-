"""NumPy CPU backend — always available as fallback."""

from __future__ import annotations

from typing import Any
import numpy as np

from .base import AbstractBackend


class NumPyBackend(AbstractBackend):
    """CPU backend using NumPy. Guaranteed to always be available."""

    @property
    def name(self) -> str:
        return "NumPy (CPU)"

    @staticmethod
    def available() -> bool:
        return True

    def norm(self, x: np.ndarray) -> float:
        return float(np.linalg.norm(x))

    def exp(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x)

    def dot(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.dot(a, b)

    def sum(self, x: np.ndarray, axis: int | None = None) -> np.ndarray:
        return np.sum(x, axis=axis)

    def max(self, x: np.ndarray, axis: int | None = None) -> float | np.ndarray:
        return np.max(x, axis=axis)

    def log(self, x: np.ndarray) -> np.ndarray:
        return np.log(x)

    def abs(self, x: np.ndarray) -> np.ndarray:
        return np.abs(x)

    def from_numpy(self, arr: np.ndarray) -> np.ndarray:
        return arr

    def to_numpy(self, arr: np.ndarray) -> np.ndarray:
        return np.asarray(arr)

    def zeros_like(self, x: np.ndarray) -> np.ndarray:
        return np.zeros_like(x)

    def ones(self, n: int) -> np.ndarray:
        return np.ones(n)
