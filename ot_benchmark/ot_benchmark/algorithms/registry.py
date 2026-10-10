"""Algorithm registry with enriched metadata."""

from __future__ import annotations

from typing import Any, Callable

try:
    from ..core.types import AlgorithmFunc
except ImportError:
    from core.types import AlgorithmFunc


# Algorithm metadata structure
# Each entry: {name: {func, description, max_dim, supports_sparse, supports_gpu,
#                    complexity_class, paper_reference, hyperparameters}}
ALGORITHMS: dict[str, dict[str, Any]] = {}


def register_algorithm(
    name: str,
    func: AlgorithmFunc,
    description: str = "",
    max_dim: int | float = float("inf"),
    supports_sparse: bool = False,
    supports_gpu: bool = True,
    complexity_class: str = "",
    paper_reference: str = "",
    hyperparameters: list[dict[str, Any]] | None = None,
    uses_support_points: bool = False,
    uses_sparse_cost: bool = False,
    ranking_group: str = "balanced",
    max_active_dim: int | float = float("inf"),
) -> None:
    """Register an algorithm in the global registry.

    Args:
        name: Unique algorithm name.
        func: Callable (source, target, cost_matrix, **params) -> float.
        description: Human-readable description.
        max_dim: Maximum supported dimension.
        supports_sparse: Whether the algorithm can use CSR cost matrices.
        supports_gpu: Whether the algorithm can run on GPU.
        complexity_class: e.g. 'O(n^2)', 'O(n^3)', 'O(nnz)'.
        paper_reference: Citation or DOI for algorithm provenance.
        hyperparameters: List of {name, default, type, description} dicts.
    """
    ALGORITHMS[name] = {
        "func": func,
        "description": description,
        "max_dim": max_dim,
        "supports_sparse": supports_sparse,
        "supports_gpu": supports_gpu,
        "complexity_class": complexity_class,
        "paper_reference": paper_reference,
        "hyperparameters": hyperparameters or [],
        "uses_support_points": uses_support_points,
        "uses_sparse_cost": uses_sparse_cost,
        "ranking_group": ranking_group,
        "max_active_dim": max_active_dim,
    }


def get_algorithm(name: str) -> dict[str, Any]:
    """Get an algorithm's metadata by name."""
    if name not in ALGORITHMS:
        raise KeyError(f"Algorithm '{name}' not registered. Available: {list(ALGORITHMS.keys())}")
    return ALGORITHMS[name]


def list_algorithms() -> list[str]:
    """Return a sorted list of registered algorithm names."""
    return sorted(ALGORITHMS.keys())


# ---- Register all algorithms ----


def _register_defaults():
    """Register all built-in algorithms. Called at module import time."""
    if ALGORITHMS:
        return

    from .emd import run_emd_exact
    from .sinkhorn import run_sinkhorn
    from .greenkhorn import run_greenkhorn
    from .sliced_ot import run_sliced_ot
    from .unbalanced_ot import run_unbalanced_ot
    from .lowrank_ot import run_lowrank_ot
    from .mean_baseline import run_mean_baseline

    register_algorithm(
        "EMD (精确解)",
        func=lambda s, t, M, **kw: run_emd_exact(s, t, M),
        description="网络单纯形法 — 按非零支撑压缩后求精确解",
        max_dim=float("inf"),
        max_active_dim=1000,
        supports_sparse=False,
        supports_gpu=False,
        complexity_class="O(n^3 log n)",
        paper_reference="POT: Python Optimal Transport (Flamary et al., 2021)",
        ranking_group="reference",
    )

    register_algorithm(
        "Sinkhorn (reg=0.1)",
        func=lambda s, t, M, **kw: run_sinkhorn(s, t, M, reg=0.1),
        description="熵正则化 OT — 平衡精度与速度",
        supports_sparse=True,
        supports_gpu=True,
        complexity_class="O(n^2)",
        hyperparameters=[{"name": "reg", "default": 0.1, "type": "float", "description": "正则化参数"}],
    )

    register_algorithm(
        "Sinkhorn (reg=0.01)",
        func=lambda s, t, M, **kw: run_sinkhorn(s, t, M, reg=0.01),
        description="低正则化参数 — 高精度但收敛较慢",
        supports_sparse=True,
        supports_gpu=True,
        complexity_class="O(n^2)",
        hyperparameters=[{"name": "reg", "default": 0.01, "type": "float", "description": "正则化参数"}],
    )

    register_algorithm(
        "Greenkhorn",
        func=lambda s, t, M, **kw: run_greenkhorn(s, t, M, reg=0.1),
        description="贪心 Sinkhorn — 每次只更新最大误差的行/列",
        supports_sparse=False,
        supports_gpu=False,
        complexity_class="O(n^2)",
    )

    register_algorithm(
        "Sliced OT (200)",
        func=lambda s, t, M, **kw: run_sliced_ot(
            s, t, M, n_projections=200, support_points=kw.get("support_points")
        ),
        description="切片 OT — 通过随机投影降维, 适合超高维",
        supports_sparse=False,
        supports_gpu=True,
        complexity_class="O(n log n × P)",
        hyperparameters=[{"name": "n_projections", "default": 200, "type": "int", "description": "投影方向数"}],
        ranking_group="sliced",
        uses_support_points=True,
    )

    register_algorithm(
        "Sliced OT (500)",
        func=lambda s, t, M, **kw: run_sliced_ot(
            s, t, M, n_projections=500, support_points=kw.get("support_points")
        ),
        description="切片 OT — 更多投影方向, 精度更高",
        ranking_group="sliced",
        supports_sparse=False,
        supports_gpu=True,
        complexity_class="O(n log n × P)",
        uses_support_points=True,
    )

    register_algorithm(
        "Unbalanced OT",
        func=lambda s, t, M, **kw: run_unbalanced_ot(s, t, M, reg=0.1),
        description="非平衡 OT — 允许质量创建/销毁, 适合噪声数据",
        supports_sparse=False,
        supports_gpu=True,
        complexity_class="O(n^2)",
        ranking_group="unbalanced",
    )

    register_algorithm(
        "Low-Rank OT",
        func=lambda s, t, M, **kw: run_lowrank_ot(s, t, M, rank=10),
        description="低秩 OT — 对代价矩阵做SVD降秩, 大幅降低计算量",
        ranking_group="surrogate_cost",
        supports_sparse=False,
        supports_gpu=True,
        complexity_class="O(n^2 × r)",
        hyperparameters=[{"name": "rank", "default": 10, "type": "int", "description": "低秩近似秩"}],
    )

    # New algorithms
    # ---- 自定义算法示例 ----
    register_algorithm(
        "Mean Baseline",
        func=lambda s, t, M, **kw: run_mean_baseline(s, t, M),
        description="均值基线 — 用加权利差作为参考下界, 不进行实际运输优化",
        supports_sparse=False,
        supports_gpu=False,
        complexity_class="O(n^2)",
        ranking_group="baseline",
    )

    try:
        from .gaussian_ot import run_gaussian_ot
        register_algorithm(
            "Gaussian OT",
            func=lambda s, t, M, **kw: run_gaussian_ot(
                s, t, M, support_points=kw.get("support_points")
            ),
            description="高斯 OT — Bures-Wasserstein 闭式解, O(d^3)",
            supports_sparse=False,
            supports_gpu=True,
            complexity_class="O(d^3)",
            paper_reference="Bures metric / Gaussian OT (Takatsu, 2011)",
            uses_support_points=True,
            ranking_group="model",
        )
    except ImportError:
        pass

    try:
        from .factored_ot import run_factored_ot
        register_algorithm(
            "Factored OT",
            func=lambda s, t, M, **kw: run_factored_ot(s, t, M),
            description="分解 OT — 利用分组结构加速",
            ranking_group="surrogate_cost",
            supports_sparse=False,
            supports_gpu=True,
            complexity_class="O(G × (n/G)^2)",
        )
    except ImportError:
        pass

    try:
        from .sparse_ot import run_sparse_sinkhorn
        register_algorithm(
            "Sparse Sinkhorn",
            func=lambda s, t, M, **kw: run_sparse_sinkhorn(
                s, t, M, cost_matrix_csr=kw.get("cost_matrix_csr"), reg=0.01
            ),
            description="稀疏 Sinkhorn — 稀疏核迭代，不连通时回退对数域",
            supports_sparse=True,
            supports_gpu=False,
            complexity_class="O(nnz)",
            uses_sparse_cost=True,
        )
    except ImportError:
        pass

    # GPU-accelerated Sinkhorn variants
    try:
        from .sinkhorn_gpu import run_sinkhorn_gpu
        register_algorithm(
            "Sinkhorn GPU (reg=0.1)",
            func=lambda s, t, M, **kw: run_sinkhorn_gpu(s, t, M, reg=0.1, **kw),
            description="GPU 加速 Sinkhorn — PyTorch CUDA 原生实现",
            supports_sparse=False,
            supports_gpu=True,
            complexity_class="O(n^2) on GPU",
            hyperparameters=[{"name": "reg", "default": 0.1, "type": "float", "description": "正则化参数"}],
        )
        register_algorithm(
            "Sinkhorn GPU (reg=0.01)",
            func=lambda s, t, M, **kw: run_sinkhorn_gpu(s, t, M, reg=0.01, **kw),
            description="GPU 加速 Sinkhorn (低正则化) — 高精度 GPU 版本",
            supports_sparse=False,
            supports_gpu=True,
            complexity_class="O(n^2) on GPU",
            hyperparameters=[{"name": "reg", "default": 0.01, "type": "float", "description": "正则化参数"}],
        )
    except ImportError:
        pass


_register_defaults()
