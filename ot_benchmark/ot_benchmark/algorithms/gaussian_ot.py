"""Gaussian Optimal Transport — Bures-Wasserstein closed form.

For distributions approximated as Gaussians, the Wasserstein-2 distance has
a closed-form solution: W_2^2 = ||m0 - m1||^2 + Bures(Sigma_0, Sigma_1).
"""

import numpy as np


def _fit_gaussian_params(x: np.ndarray, positions: np.ndarray | None = None) -> tuple:
    """Fit Gaussian mean and covariance to a discrete distribution.

    Treats the distribution as weighted samples at positions [0, 1, ..., n-1]
    and computes weighted empirical mean and covariance.

    Returns:
        (mean, covariance) where covariance is 1x1 for 1-D positions.
    """
    if positions is None:
        positions = np.arange(len(x), dtype=np.float64)

    x = x / x.sum()
    mean = np.sum(x[:, None] * positions, axis=0)
    centered = positions - mean
    cov = (centered * x[:, None]).T @ centered
    cov += np.eye(cov.shape[0]) * 1e-12
    return mean, cov


def _sqrt_psd(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2.0)
    values = np.maximum(values, 0.0)
    return (vectors * np.sqrt(values)) @ vectors.T


def _bures_distance(mean0, cov0, mean1, cov1) -> float:
    """Compute Bures-Wasserstein distance for 1-D Gaussians.

    Bures(Sigma_0, Sigma_1) = Tr(Sigma_0) + Tr(Sigma_1)
                              - 2 * Tr((Sigma_0^{1/2} Sigma_1 Sigma_0^{1/2})^{1/2})
    """
    mean_diff_sq = float(np.sum((mean0 - mean1) ** 2))
    sqrt_cov0 = _sqrt_psd(cov0)
    middle = _sqrt_psd(sqrt_cov0 @ cov1 @ sqrt_cov0)
    bures = float(np.trace(cov0 + cov1 - 2.0 * middle))
    return max(0.0, mean_diff_sq + bures)


def run_gaussian_ot(
    source: np.ndarray,
    target: np.ndarray,
    cost_matrix: np.ndarray,  # unused, kept for interface consistency
    support_points: np.ndarray | None = None,
) -> float:
    """Compute Gaussian OT (Bures-Wasserstein) distance.

    Treats the source and target distributions as weighted samples
    and fits Gaussian parameters. Uses the closed-form Bures metric.

    This is O(n) for fitting + O(1) for the distance — vastly faster
    than full OT for distributions that are approximately unimodal.

    Args:
        source: Source distribution.
        target: Target distribution.
        cost_matrix: Ignored.
    """
    n = len(source)
    if support_points is None:
        positions = np.linspace(0.0, 1.0, n).reshape(-1, 1)
    else:
        positions = np.asarray(support_points, dtype=np.float64)
        if positions.ndim == 1:
            positions = positions.reshape(-1, 1)

    mean_s, cov_s = _fit_gaussian_params(source, positions)
    mean_t, cov_t = _fit_gaussian_params(target, positions)

    return _bures_distance(mean_s, cov_s, mean_t, cov_t)
