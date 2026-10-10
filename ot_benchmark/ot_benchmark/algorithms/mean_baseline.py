"""Mean Baseline — 简单基线算法，用于校准其他算法的提升幅度."""

import numpy as np


def run_mean_baseline(
    source: np.ndarray,
    target: np.ndarray,
    cost_matrix: np.ndarray,
) -> float:
    """用源分布和目标分布的加权均值作为运输代价下界参考.

    返回: Wasserstein 距离上界 (Baseline 参考值)
    """
    a = source.astype(np.float64).copy()
    b = target.astype(np.float64).copy()
    M = cost_matrix.astype(np.float64).copy()

    a = a / a.sum()
    b = b / b.sum()

    # 直接用代价矩阵的最大值 × 分布差异作为 naive 上界
    mean_cost = np.dot(a, M @ b)
    return float(mean_cost)
