"""
高维稀疏数据生成模块
=====================
根据项目书设计，合成具有可控维度、稀疏度及流形结构的人工数据集。
支持：
  - 长尾分布（Zipf's law / Power-law）
  - 结构化稀疏与非结构化稀疏
  - 多噪声水平
  - 输出 CSR/CSC 稀疏格式
"""

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from scipy.spatial.distance import cdist
from scipy.stats import zipf, pareto
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class SparseDataset:
    """稀疏数据集容器"""
    name: str
    source_dist: np.ndarray      # 源分布 (n_samples,)
    target_dist: np.ndarray      # 目标分布 (n_samples,)
    cost_matrix: np.ndarray       # 代价矩阵 (n_samples, n_samples) - 可选稠密存储
    cost_matrix_csr: csr_matrix   # 代价矩阵 CSR 格式
    ground_truth_distance: float  # 真实 Wasserstein 距离（精确解或伪真值）
    dim: int
    sparsity: float
    noise_level: float
    description: str
    ground_truth_method: str = "unknown"
    ground_truth_is_exact: bool = False
    reference_algorithm: str = ""
    data_origin: str = "synthetic"
    source_url: str = ""
    support_points: Optional[np.ndarray] = None
    geometry_description: str = "index position (1D)"
    geometry_dim: int = 1
    original_dim: Optional[int] = None
    dimension_tier: str = "custom"

    # ---- Benchmark v1.0 结构化字段 ----
    dataset_id: str = ""               # 场景唯一ID, 如 "SYN-DENSE-0200-S01"
    dataset_seed: int = 0              # 数据生成种子
    sparsity_semantics: str = "zero_ratio"  # "zero_ratio" | "nonzero_ratio"
    structure: str = "random"          # random | block | group | hierarchical
    balanced: bool = True              # 是否为平衡 OT
    ground_truth_mode: str = "unknown" # exact | sinkhorn_ref | l1_fallback
    n_source: int = 0                  # 源分布样本数 (同 dim)
    n_target: int = 0                  # 目标分布样本数 (同 dim)
    source_observation_id: str = ""
    target_observation_id: str = ""

    @property
    def dimension(self) -> int:
        """维度 = n_source = dim (兼容旧代码)."""
        return self.n_source if self.n_source > 0 else self.dim

    def __post_init__(self):
        # 向后兼容: 如果没有设置 n_source/n_target, 从 dim 推断
        if self.n_source == 0:
            self.n_source = self.dim
        if self.n_target == 0:
            self.n_target = self.dim
        if not self.dataset_id:
            self.dataset_id = self.name

        if self.support_points is None:
            n = len(self.source_dist)
            self.support_points = np.linspace(0.0, 1.0, n).reshape(-1, 1)
        else:
            self.support_points = np.asarray(self.support_points, dtype=np.float64)
            if self.support_points.ndim == 1:
                self.support_points = self.support_points.reshape(-1, 1)
            if self.support_points.shape[0] != len(self.source_dist):
                raise ValueError("support_points must align with source_dist")
        self.geometry_dim = int(self.support_points.shape[1])

    @property
    def active_storage_dim(self) -> int:
        """Number of coordinates materialized after zero-mass compression."""
        return len(self.source_dist)


def generate_sparse_distribution(
    n_samples: int,
    sparsity: float = 0.9,
    distribution: str = "zipf",
    noise_level: float = 0.0,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    生成服从特定分布的稀疏概率向量。

    参数:
        n_samples: 样本/维度数
        sparsity: 稀疏度 (0~1)，接近零值的比例
        distribution: 分布类型 - "zipf", "pareto", "uniform", "gaussian_mixture"
        noise_level: 噪声水平 (0~1)
        random_state: 随机种子

    返回:
        归一化的概率分布向量
    """
    rng = np.random.RandomState(random_state)

    if distribution == "zipf":
        # Zipf 分布模拟长尾稀疏（如推荐系统中热门物品少数、冷门物品大量）
        a = 1.5 + sparsity * 2.0  # 参数越大尾部越重
        raw = zipf.rvs(a, size=n_samples, random_state=rng).astype(np.float64)

    elif distribution == "pareto":
        # Pareto/幂律分布
        b = 1.0 + sparsity * 3.0
        raw = pareto.rvs(b, size=n_samples, random_state=rng).astype(np.float64)

    elif distribution == "uniform":
        raw = rng.uniform(0.1, 1.0, size=n_samples)

    elif distribution == "gaussian_mixture":
        # 混合高斯：模拟多模态分布
        n_components = max(2, int(n_samples * (1 - sparsity) / 50))
        raw = np.zeros(n_samples)
        for _ in range(n_components):
            center = rng.randint(0, n_samples)
            width = max(5, int(n_samples * 0.02))
            raw[center:min(center + width, n_samples)] += rng.uniform(0.5, 2.0)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    # 稀疏化：将部分值置零
    if sparsity > 0:
        mask = rng.random(n_samples) > sparsity
        raw = raw * mask

    # 确保没有全零
    if raw.sum() < 1e-10:
        raw[rng.randint(0, n_samples)] = 1.0

    # 添加噪声
    if noise_level > 0:
        active = raw > 0
        raw[active] += rng.exponential(noise_level, size=int(active.sum()))
        zero_indices = np.flatnonzero(~active)
        n_false_positives = min(
            len(zero_indices),
            int(round(noise_level * max(1, int(active.sum())))),
        )
        if n_false_positives > 0:
            selected = rng.choice(zero_indices, n_false_positives, replace=False)
            raw[selected] = rng.exponential(noise_level, size=n_false_positives)

    # 归一化为概率分布
    raw = raw / raw.sum()
    return raw


def normalize_support_points(points: np.ndarray) -> np.ndarray:
    """Center and scale support coordinates with linear memory usage."""
    values = np.asarray(points, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    values = values - np.mean(values, axis=0, keepdims=True)
    max_radius = float(np.max(np.linalg.norm(values, axis=1))) if values.size else 0.0
    if max_radius > 1e-15:
        values = values / (2.0 * max_radius)
    return values


def generate_cost_matrix(
    source: np.ndarray,
    target: np.ndarray,
    metric: str = "euclidean",
    sparse_output: bool = True,
    support_points: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, csr_matrix]:
    """
    生成源分布与目标分布之间的代价矩阵。

    参数:
        source: 源分布
        target: 目标分布
        metric: 距离度量 - "euclidean", "manhattan", "cosine"
        sparse_output: 是否输出 CSR 稀疏矩阵

    返回:
        (稠密代价矩阵, CSR稀疏代价矩阵)
    """
    n = len(source)
    if len(target) != n:
        raise ValueError("source and target must use the same support size")

    if support_points is None:
        positions = np.linspace(0.0, 1.0, n).reshape(-1, 1)
    else:
        positions = np.asarray(support_points, dtype=np.float64)
        if positions.ndim == 1:
            positions = positions.reshape(-1, 1)
        if positions.shape[0] != n:
            raise ValueError(
                f"support_points has {positions.shape[0]} rows, expected {n}"
            )

    if metric == "euclidean":
        dense = cdist(positions, positions, metric="sqeuclidean")
    elif metric == "manhattan":
        dense = cdist(positions, positions, metric="cityblock")
    elif metric == "cosine":
        dense = cdist(positions, positions, metric="cosine")
        dense = np.nan_to_num(dense, nan=0.0, posinf=1.0, neginf=0.0)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # 归一化
    dense = dense / (dense.max() + 1e-10)

    # CSR is a lossless storage representation, not a kernel truncation.
    # Approximate kernel sparsification belongs to the solver and depends on reg.
    # sparse_output is retained for API compatibility; both paths are lossless.
    csr = csr_matrix(dense)

    return dense, csr


def compute_ot_reference(
    source: np.ndarray,
    target: np.ndarray,
    cost_matrix: np.ndarray,
    max_exact_support: int = 1000,
    allow_exact: bool = True,
) -> tuple[float, str, bool, str]:
    """Compute an exact or Sinkhorn reference according to the protocol.

    ``allow_exact=False`` is used by scenarios explicitly designated
    ``sinkhorn_ref``; sparsity must not silently upgrade those scenarios to
    an exact EMD reference merely because their active support is small.
    """
    try:
        import ot

        a = np.asarray(source, dtype=np.float64)
        b = np.asarray(target, dtype=np.float64)
        if (a.ndim != 1 or b.ndim != 1 or not np.isfinite(a).all()
                or not np.isfinite(b).all() or np.any(a < 0) or np.any(b < 0)
                or a.sum() <= 0 or b.sum() <= 0
                or cost_matrix.shape != (a.size, b.size)
                or not np.isfinite(cost_matrix).all() or np.any(cost_matrix < 0)):
            raise ValueError("invalid OT reference inputs")
        a = a / a.sum()
        b = b / b.sum()
        source_active = np.flatnonzero(a > 0)
        target_active = np.flatnonzero(b > 0)
        active_size = len(np.union1d(source_active, target_active))

        if allow_exact and active_size <= max_exact_support:
            active_a = a[source_active]
            active_b = b[target_active]
            active_a = active_a / active_a.sum()
            active_b = active_b / active_b.sum()
            active_cost = cost_matrix[np.ix_(source_active, target_active)]
            plan, solver_log = ot.emd(active_a, active_b, active_cost, log=True)
            if solver_log.get("warning") or solver_log.get("result_code", 1) != 1:
                raise ValueError("EMD solver did not report optimality")
            residual = validate_transport_plan(plan, active_a, active_b)
            distance = float(np.sum(plan * active_cost))
            method = f"EMD exact validated (active support={active_size}, residual={residual:.3g})"
            return distance, method, True, "EMD (精确解)"

        reg = 0.01
        active_a = a[source_active]
        active_b = b[target_active]
        active_cost = cost_matrix[np.ix_(source_active, target_active)]
        plan = ot.sinkhorn(
            active_a, active_b, active_cost, reg,
            numItermax=10000, stopThr=1e-10, method="sinkhorn_log",
        )
        residual = validate_transport_plan(plan, active_a, active_b)
        distance = float(np.sum(plan * active_cost))
        return (
            distance,
            f"Sinkhorn log pseudo-reference validated (reg={reg}, active support={active_size}, residual={residual:.3g})",
            False,
            "Sinkhorn (reg=0.01)",
        )
    except (ImportError, ValueError, MemoryError, np.linalg.LinAlgError) as exc:
        return float("nan"), f"OT reference unavailable: {type(exc).__name__}: {exc}", False, ""


def validate_transport_plan(plan, source, target, tolerance=1e-7) -> float:
    """Check primal feasibility; this alone is not an optimality certificate."""
    plan = np.asarray(plan)
    if (plan.shape != (len(source), len(target)) or not np.isfinite(plan).all()
            or np.any(plan < 0)):
        raise ValueError("invalid transport plan")
    residual = float(max(np.abs(plan.sum(axis=1) - source).sum(),
                         np.abs(plan.sum(axis=0) - target).sum()))
    if residual > tolerance:
        raise ValueError(f"transport marginal residual {residual:.3g} exceeds {tolerance}")
    return residual


def has_usable_reference(value, method, exact):
    return bool(np.isfinite(value) and value >= 0 and (
        exact or ("Sinkhorn" in method and "validated" in method
                  and "unavailable" not in method)))


def create_benchmark_dataset(
    n_samples: int = 500,
    sparsity: float = 0.85,
    dim_factor: float = 1.0,
    noise_level: float = 0.05,
    source_dist_type: str = "zipf",
    target_dist_type: str = "pareto",
    random_state: int = 42,
    name: str = "benchmark",
    *,
    structure: str = "random",
    sparsity_semantics: str = "zero_ratio",
    balanced: bool = True,
    ground_truth_mode: str = "unknown",
    geometry_dim: int = 1,
) -> SparseDataset:
    """
    创建一个完整的基准测试数据集。

    参数:
        n_samples: 样本数（模拟维度）
        sparsity: 稀疏度 (zero_ratio: 0.95 = 95% 零元素)
        dim_factor: 维度因子（控制有效维度与总维度的比例）
        noise_level: 噪声水平
        source_dist_type: 源分布类型
        target_dist_type: 目标分布类型
        random_state: 随机种子（即 dataset_seed）
        name: 数据集名称
        structure: 稀疏结构 random | block | group | hierarchical
        sparsity_semantics: 稀疏度语义 "zero_ratio" | "nonzero_ratio"
        balanced: 是否为平衡 OT
        ground_truth_mode: exact | sinkhorn_ref | l1_fallback
        geometry_dim: 支撑点几何维度。默认 1 以保持旧结果兼容；高维基准应显式设置 >1。

    返回:
        SparseDataset 对象
    """
    if n_samples < 1 or not 0 < dim_factor <= 1:
        raise ValueError("n_samples must be positive and dim_factor in (0, 1]")
    if not 0 <= sparsity <= 1:
        raise ValueError("sparsity must be in [0, 1]")
    if sparsity_semantics not in ("zero_ratio", "nonzero_ratio"):
        raise ValueError("unknown sparsity semantics")
    if structure not in ("random", "block", "group", "hierarchical"):
        raise ValueError("unknown sparsity structure")
    zero_ratio = sparsity if sparsity_semantics == "zero_ratio" else 1 - sparsity
    effective_dim = max(1, int(n_samples * dim_factor))

    def make_distribution(kind, seed):
        values = generate_sparse_distribution(
            effective_dim, zero_ratio, kind, noise_level, seed
        )
        if structure != "random":
            # Preserve the sampled weights (including noise and long tails),
            # changing only their support layout. Thus structure comparisons
            # at a fixed seed have exactly the same mass spectrum and nnz.
            weights = values[values > 0].copy()
            mask = generate_structured_sparsity(
                effective_dim, 1 - len(weights) / effective_dim,
                structure=structure, random_state=seed,
            ) > 0
            values = np.zeros(effective_dim)
            values[mask] = weights
        return values

    source = make_distribution(source_dist_type, random_state)
    target = make_distribution(target_dist_type, random_state + 1)

    # 如果有效维度与目标不同，补零对齐
    if effective_dim < n_samples:
        source = np.pad(source, (0, n_samples - effective_dim))
        target = np.pad(target, (0, n_samples - effective_dim))

    if geometry_dim < 1:
        raise ValueError("geometry_dim must be >= 1")
    support_points = None
    if geometry_dim > 1:
        geom_rng = np.random.RandomState(random_state + 2_147_483_647)
        support_points = normalize_support_points(
            geom_rng.normal(size=(len(source), int(geometry_dim)))
        )
    cost_dense, cost_csr = generate_cost_matrix(
        source, target, "euclidean", support_points=support_points
    )

    gt_distance, ground_truth_method, ground_truth_is_exact, reference_algorithm = (
        compute_ot_reference(
            source, target, cost_dense,
            allow_exact=(ground_truth_mode != "sinkhorn_ref"),
        )
    )

    actual_sparsity = 1.0 - (
        np.count_nonzero(source) + np.count_nonzero(target)
    ) / (2.0 * len(source))

    return SparseDataset(
        name=name,
        source_dist=source,
        target_dist=target,
        cost_matrix=cost_dense,
        cost_matrix_csr=cost_csr,
        ground_truth_distance=gt_distance,
        dim=effective_dim,
        sparsity=actual_sparsity,
        noise_level=noise_level,
        description=f"dim={effective_dim}/{n_samples}, sparsity={actual_sparsity:.2f}, "
                     f"source={source_dist_type}, target={target_dist_type}",
        ground_truth_method=ground_truth_method,
        ground_truth_is_exact=ground_truth_is_exact,
        reference_algorithm=reference_algorithm,
        dataset_id=name,
        dataset_seed=random_state,
        sparsity_semantics="zero_ratio",
        structure=structure,
        balanced=balanced,
        ground_truth_mode=ground_truth_mode,
        n_source=n_samples,
        n_target=n_samples,
        support_points=support_points,
        geometry_description=(
            "index position (1D)" if geometry_dim == 1
            else f"deterministic Gaussian support ({geometry_dim}D)"
        ),
    )


def create_dataset_suite() -> list[SparseDataset]:
    """
    创建一套覆盖不同维度、稀疏度、噪声水平的测试数据集。
    根据项目书设计：至少5个子集。

    返回:
        数据集列表
    """
    datasets = []

    # 子集 1: 低维低稀疏（基准参考）
    datasets.append(create_benchmark_dataset(
        n_samples=200, sparsity=0.5, noise_level=0.01,
        source_dist_type="uniform", target_dist_type="uniform",
        random_state=10, name="DS1-Baseline"
    ))

    # 子集 2: 中维高稀疏（推荐系统场景）
    datasets.append(create_benchmark_dataset(
        n_samples=500, sparsity=0.90, noise_level=0.05,
        source_dist_type="zipf", target_dist_type="zipf",
        random_state=20, name="DS2-Recommendation"
    ))

    # 子集 3: 高维极端稀疏（文本分类场景）
    datasets.append(create_benchmark_dataset(
        n_samples=1000, sparsity=0.95, noise_level=0.1,
        source_dist_type="zipf", target_dist_type="pareto",
        random_state=30, name="DS3-TextClassification"
    ))

    # 子集 4: 混合高斯（生物信息学场景）
    datasets.append(create_benchmark_dataset(
        n_samples=800, sparsity=0.80, noise_level=0.08,
        source_dist_type="gaussian_mixture",
        target_dist_type="gaussian_mixture",
        random_state=40, name="DS4-Bioinformatics"
    ))

    # 子集 5: 超高维（压力测试）
    datasets.append(create_benchmark_dataset(
        n_samples=2000, sparsity=0.98, noise_level=0.15,
        source_dist_type="pareto", target_dist_type="zipf",
        random_state=50, name="DS5-StressTest"
    ))

    # 子集 6: 结构化稀疏（模拟块稀疏结构）
    datasets.append(create_benchmark_dataset(
        n_samples=600, sparsity=0.85, noise_level=0.03,
        source_dist_type="zipf", target_dist_type="gaussian_mixture",
        random_state=60, name="DS6-StructuredSparse"
    ))

    # 子集 7: 超高维极端稀疏（大规模推荐场景）
    datasets.append(create_benchmark_dataset(
        n_samples=3000, sparsity=0.99, noise_level=0.02,
        source_dist_type="zipf", target_dist_type="zipf",
        random_state=70, name="DS7-ExtremeSparse"
    ))

    # 子集 8: 低维高噪声（鲁棒性测试）
    datasets.append(create_benchmark_dataset(
        n_samples=150, sparsity=0.40, noise_level=0.20,
        source_dist_type="uniform", target_dist_type="pareto",
        random_state=80, name="DS8-HighNoise"
    ))

    return datasets


# ============================================================================
#  结构化稀疏生成
# ============================================================================

def generate_structured_sparsity(
    n_samples: int,
    sparsity: float,
    structure: str = "random",
    block_size: int = 50,
    n_blocks: int = 5,
    n_groups: int = 10,
    random_state: int = 42,
) -> np.ndarray:
    """
    生成具有结构化稀疏模式的概率分布向量。

    参数:
        n_samples: 样本/维度数
        sparsity: 目标稀疏度 (0~1)
        structure: 稀疏结构类型
            - "random": 随机稀疏 (默认)
            - "block": 块稀疏 — 非零元聚集在 n_blocks 个连续块中
            - "group": 组稀疏 — 列分为 n_groups 组，组内密集、组间稀疏
            - "hierarchical": 层级稀疏 — 大块稀疏 + 小块密集的多层级结构
        block_size: 兼容旧接口保留；当前块布局由 n_samples 和 n_blocks 决定
        n_blocks: 块稀疏时的块数量
        n_groups: 组稀疏时的组数量
        random_state: 随机种子

    返回:
        归一化的概率分布向量
    """
    if n_samples < 1 or not 0 <= sparsity <= 1:
        raise ValueError("positive support size and sparsity in [0, 1] required")
    if min(block_size, n_blocks, n_groups) < 1:
        raise ValueError("structure sizes must be positive")
    rng = np.random.RandomState(random_state)
    nnz_target = max(1, min(n_samples, int(round(n_samples * (1 - sparsity)))))
    if structure == "random":
        order = rng.permutation(n_samples)
    elif structure == "block":
        # Fill contiguous runs across disjoint coarse blocks, without overlap.
        chunks = np.array_split(np.arange(n_samples), min(n_blocks, n_samples))
        runs = []
        for i in rng.permutation(len(chunks)):
            chunk = chunks[i]
            runs.append(np.roll(chunk, -rng.randint(len(chunk))))
        order = np.array([run[j] for j in range(max(map(len, runs)))
                          for run in runs if j < len(run)])
    elif structure == "group":
        groups = np.array_split(np.arange(n_samples), min(n_groups, n_samples))
        order = np.concatenate([groups[i] for i in rng.permutation(len(groups))])
    elif structure == "hierarchical":
        # Random coarse-group ordering, then fine-group ordering within it.
        coarse = np.array_split(np.arange(n_samples), min(n_groups, n_samples))
        ordered = []
        for i in rng.permutation(len(coarse)):
            fine = np.array_split(coarse[i], min(n_blocks, len(coarse[i])))
            ordered.extend(fine[j] for j in rng.permutation(len(fine)))
        order = np.concatenate(ordered)
    else:
        raise ValueError(f"Unknown structure type: {structure}")
    raw = np.zeros(n_samples)
    raw[order[:nnz_target]] = np.abs(rng.randn(nnz_target)) + 0.5
    return raw / raw.sum()



def create_benchmark_dataset_with_shift(
    n_samples: int = 500,
    sparsity: float = 0.85,
    noise_level: float = 0.05,
    source_dist_type: str = "zipf",
    shift_type: str = "covariate",
    shift_intensity: float = 0.3,
    random_state: int = 42,
    name: str = "shift_benchmark",
    geometry_dim: int = 1,
) -> SparseDataset:
    """
    创建具有受控分布偏移的基准数据集。

    源分布保持原始合成方式，目标分布通过 apply_distribution_shift 产生。
    """
    source = generate_sparse_distribution(
        n_samples, sparsity, source_dist_type, noise_level, random_state
    )
    target = apply_distribution_shift(source, shift_type, shift_intensity, random_state + 1)

    if geometry_dim < 1:
        raise ValueError("geometry_dim must be >= 1")
    support_points = None
    if geometry_dim > 1:
        geom_rng = np.random.RandomState(random_state + 2_147_483_647)
        support_points = normalize_support_points(
            geom_rng.normal(size=(len(source), int(geometry_dim)))
        )
    cost_dense, cost_csr = generate_cost_matrix(
        source, target, "euclidean", support_points=support_points
    )

    # Ground Truth
    try:
        import ot
        a = source.copy() / source.sum()
        b = target.copy() / target.sum()
        if n_samples <= 300:
            gt_distance = float(ot.emd2(a, b, cost_dense))
            ground_truth_method = "EMD exact"
            ground_truth_is_exact = True
            reference_algorithm = "EMD (精确解)"
        else:
            plan = ot.sinkhorn(a, b, cost_dense, 0.01, numItermax=5000, stopThr=1e-9)
            gt_distance = np.sum(plan * cost_dense)
            ground_truth_method = "Sinkhorn pseudo-reference (reg=0.01)"
            ground_truth_is_exact = False
            reference_algorithm = "Sinkhorn (reg=0.01)"
    except (ImportError, ValueError, MemoryError, np.linalg.LinAlgError):
        gt_distance = np.sum(np.abs(source - target))
        ground_truth_method = "L1 fallback"
        ground_truth_is_exact = False
        reference_algorithm = ""

    return SparseDataset(
        name=name,
        source_dist=source,
        target_dist=target,
        cost_matrix=cost_dense,
        cost_matrix_csr=cost_csr,
        ground_truth_distance=gt_distance,
        dim=n_samples,
        sparsity=sparsity,
        noise_level=noise_level,
        description=f"dim={n_samples}, sparsity={sparsity:.2f}, shift={shift_type}({shift_intensity:.1f})",
        ground_truth_method=ground_truth_method,
        ground_truth_is_exact=ground_truth_is_exact,
        reference_algorithm=reference_algorithm,
        support_points=support_points,
        geometry_description=(
            "index position (1D)" if geometry_dim == 1
            else f"deterministic Gaussian support ({geometry_dim}D)"
        ),
    )


# ============================================================================
#  数据特征诊断
# ============================================================================

def diagnose_dataset(ds: SparseDataset) -> dict:
    """
    对数据集进行多维特征诊断。

    返回的字典包含:
        - n_samples: 维度数
        - n_nonzero_source: 源分布非零元素数
        - n_nonzero_target: 目标分布非零元素数
        - effective_sparsity: 源分布有效稀疏度
        - entropy_source: 源分布熵 (bits)
        - entropy_target: 目标分布熵 (bits)
        - skewness_source: 源分布偏度
        - skewness_target: 目标分布偏度
        - kl_divergence: 源→目标 KL 散度
        - js_divergence: Jensen-Shannon 散度
        - cost_matrix_condition: 代价矩阵条件数估计
        - cost_matrix_mean: 代价矩阵均值
        - gini_source: 源分布基尼系数 (不均衡度)
        - gini_target: 目标分布基尼系数
        - ground_truth: 真实 Wasserstein 距离
    """
    info = {}

    source = ds.source_dist
    target = ds.target_dist
    cost = ds.cost_matrix

    info["n_samples"] = ds.dim
    info["n_nonzero_source"] = int((source > 1e-12).sum())
    info["n_nonzero_target"] = int((target > 1e-12).sum())
    info["effective_sparsity"] = 1.0 - info["n_nonzero_source"] / ds.dim
    info["ground_truth"] = ds.ground_truth_distance

    # 熵
    eps = 1e-12
    s_nz = source[source > eps]
    t_nz = target[target > eps]
    info["entropy_source"] = float(-np.sum(s_nz * np.log2(s_nz)))
    info["entropy_target"] = float(-np.sum(t_nz * np.log2(t_nz)))

    # 偏度
    mean_s, std_s = source.mean(), source.std() + eps
    info["skewness_source"] = float(np.mean(((source - mean_s) / std_s) ** 3))
    mean_t, std_t = target.mean(), target.std() + eps
    info["skewness_target"] = float(np.mean(((target - mean_t) / std_t) ** 3))

    # KL 散度
    p = source + eps
    q = target + eps
    p, q = p / p.sum(), q / q.sum()
    info["kl_divergence"] = float(np.sum(p * np.log2(p / q)))

    # Jensen-Shannon 散度
    m = (p + q) / 2
    info["js_divergence"] = float(
        0.5 * np.sum(p * np.log2(p / m)) + 0.5 * np.sum(q * np.log2(q / m))
    )

    # 代价矩阵特征
    if cost.size > 0:
        info["cost_matrix_mean"] = float(cost.mean())
        info["cost_matrix_max"] = float(cost.max())

        try:
            k = min(50, min(cost.shape) - 1)
            if k >= 2:
                sv = np.linalg.svd(cost, compute_uv=False)
                info["cost_matrix_condition"] = float(sv[0] / (sv[min(k, len(sv)-1)] + eps))
            else:
                info["cost_matrix_condition"] = 1.0
        except (np.linalg.LinAlgError, ValueError):
            info["cost_matrix_condition"] = float("nan")
    else:
        info["cost_matrix_mean"] = 0.0
        info["cost_matrix_max"] = 0.0
        info["cost_matrix_condition"] = float("nan")

    # 基尼系数 (不均衡度度量)
    info["gini_source"] = float(_gini_coefficient(source))
    info["gini_target"] = float(_gini_coefficient(target))

    return info


def _gini_coefficient(x: np.ndarray) -> float:
    """计算基尼系数 (0=完全均匀, 1=极度不均衡)"""
    x_sorted = np.sort(x[x > 0] if (x > 0).any() else x)
    n = len(x_sorted)
    if n == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * x_sorted)) / (n * np.sum(x_sorted)) - (n + 1) / n)


def print_dataset_diagnosis(ds: SparseDataset):
    """格式化打印数据集诊断信息"""
    info = diagnose_dataset(ds)
    print(f"\n{'='*65}")
    print(f"  [Diagnosis] 数据集诊断: {ds.name}")
    print(f"{'='*65}")
    print(f"  维度:        {info['n_samples']}")
    print(f"  非零元:      源 {info['n_nonzero_source']} / 目标 {info['n_nonzero_target']}")
    print(f"  有效稀疏度:  {info['effective_sparsity']:.2%}")
    print(f"  分布熵:      源 {info['entropy_source']:.2f} bits / 目标 {info['entropy_target']:.2f} bits")
    print(f"  分布偏度:    源 {info['skewness_source']:.2f} / 目标 {info['skewness_target']:.2f}")
    print(f"  KL 散度:     {info['kl_divergence']:.4f}")
    print(f"  JS 散度:     {info['js_divergence']:.4f}")
    print(f"  基尼系数:    源 {info['gini_source']:.3f} / 目标 {info['gini_target']:.3f}")
    print(f"  代价矩阵均值: {info['cost_matrix_mean']:.4f}")
    print(f"  代价矩阵条件数: {info['cost_matrix_condition']:.1f}")
    print(f"  Ground Truth: {info['ground_truth']:.6f}")
    print(f"{'='*65}\n")


def create_dataset_suite_with_shifts() -> list[SparseDataset]:
    """创建包含分布偏移的扩展数据集套件"""
    datasets = []

    # Shift 1: Covariate shift — 模拟训练/测试分布不同
    datasets.append(create_benchmark_dataset_with_shift(
        n_samples=400, sparsity=0.80, noise_level=0.03,
        source_dist_type="zipf", shift_type="covariate",
        shift_intensity=0.3, random_state=100,
        name="Shift-DS1-Covariate"
    ))

    # Shift 2: Label shift — 模拟类别先验变化
    datasets.append(create_benchmark_dataset_with_shift(
        n_samples=400, sparsity=0.80, noise_level=0.03,
        source_dist_type="zipf", shift_type="label",
        shift_intensity=0.4, random_state=110,
        name="Shift-DS2-Label"
    ))

    # Shift 3: Concept drift — 模拟特征-标签关系变化
    datasets.append(create_benchmark_dataset_with_shift(
        n_samples=400, sparsity=0.80, noise_level=0.03,
        source_dist_type="zipf", shift_type="concept",
        shift_intensity=0.5, random_state=120,
        name="Shift-DS3-ConceptDrift"
    ))

    return datasets


def create_structured_sparse_datasets() -> list[SparseDataset]:
    """创建结构化稀疏数据集套件"""
    datasets = []

    configs = [
        ("Struct-DS1-Block", "block", 500, 0.85),
        ("Struct-DS2-Group", "group", 600, 0.80),
        ("Struct-DS3-Hierarchical", "hierarchical", 800, 0.90),
        ("Struct-DS4-Random", "random", 500, 0.85),
    ]

    for name, structure, n, sp in configs:
        rng = np.random.RandomState(hash(name) % (2**31))
        source = generate_structured_sparsity(
            n, sp, structure, random_state=rng.randint(0, 2**30)
        )
        target = generate_structured_sparsity(
            n, sp, structure, random_state=rng.randint(0, 2**30) + 1
        )

        cost_dense, cost_csr = generate_cost_matrix(source, target, "euclidean")

        try:
            import ot
            a, b = source / source.sum(), target / target.sum()
            if n <= 300:
                gt = float(ot.emd2(a, b, cost_dense))
                ground_truth_method = "EMD exact"
                ground_truth_is_exact = True
                reference_algorithm = "EMD (精确解)"
            else:
                plan = ot.sinkhorn(a, b, cost_dense, 0.01, numItermax=5000, stopThr=1e-9)
                gt = np.sum(plan * cost_dense)
                ground_truth_method = "Sinkhorn pseudo-reference (reg=0.01)"
                ground_truth_is_exact = False
                reference_algorithm = "Sinkhorn (reg=0.01)"
        except (ImportError, ValueError, MemoryError, np.linalg.LinAlgError):
            gt = np.sum(np.abs(source - target))
            ground_truth_method = "L1 fallback"
            ground_truth_is_exact = False
            reference_algorithm = ""

        datasets.append(SparseDataset(
            name=name,
            source_dist=source,
            target_dist=target,
            cost_matrix=cost_dense,
            cost_matrix_csr=cost_csr,
            ground_truth_distance=gt,
            dim=n,
            sparsity=sp,
            noise_level=0.03,
            description=f"dim={n}, sparsity={sp:.2f}, structure={structure}",
            ground_truth_method=ground_truth_method,
            ground_truth_is_exact=ground_truth_is_exact,
            reference_algorithm=reference_algorithm,
        ))

    return datasets


if __name__ == "__main__":
    print("=" * 60)
    print("  高维稀疏数据生成模块 - 测试")
    print("=" * 60)

    suite = create_dataset_suite()
    for ds in suite:
        nnz = (ds.source_dist > 0).sum()
        print(f"\n📊 {ds.name}")
        print(f"   维度: {ds.dim}, 稀疏度: {ds.sparsity:.2%}")
        print(f"   噪声: {ds.noise_level:.2%}")
        print(f"   源分布非零元素: {nnz}/{len(ds.source_dist)}")
        print(f"   真实距离(伪): {ds.ground_truth_distance:.6f}")
        print(f"   描述: {ds.description}")
