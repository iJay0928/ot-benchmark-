"""Shared type aliases and protocols for the OT Benchmark system."""

from __future__ import annotations

from typing import Protocol, Any, TYPE_CHECKING
import numpy as np
from scipy.sparse import csr_matrix

Distribution = np.ndarray
"""1-D probability distribution vector (n_samples,)."""

CostMatrix = np.ndarray
"""2-D dense cost matrix (n_samples, n_samples)."""

SparseMatrix = csr_matrix
"""CSR sparse matrix."""


class AlgorithmFunc(Protocol):
    """Protocol for algorithm functions."""

    def __call__(
        self,
        source: Distribution,
        target: Distribution,
        cost_matrix: CostMatrix,
        **params: Any,
    ) -> float:
        ...


class BackendProtocol(Protocol):
    """Protocol for compute backend objects."""

    @property
    def name(self) -> str: ...

    @staticmethod
    def available() -> bool: ...

    def norm(self, x: Any) -> Any: ...
    def exp(self, x: Any) -> Any: ...
    def dot(self, a: Any, b: Any) -> Any: ...
    def sum(self, x: Any, axis: int | None = None) -> Any: ...
    def max(self, x: Any, axis: int | None = None) -> Any: ...
    def log(self, x: Any) -> Any: ...
    def abs(self, x: Any) -> Any: ...
    def from_numpy(self, arr: np.ndarray) -> Any: ...
    def to_numpy(self, arr: Any) -> np.ndarray: ...
