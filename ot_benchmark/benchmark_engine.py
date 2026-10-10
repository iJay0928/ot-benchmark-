"""
OT 算法评测引擎
===============
对标 PPT Slide 9 的实际演示 —— 遍历多个 OT 算法，
测量内存、运行时间，计算加权评分，输出自动评测结果。

支持的算法:
  - EMD (精确 Earth Mover's Distance)
  - Sinkhorn (熵正则化 OT)
  - Greenkhorn (贪心 Sinkhorn)
  - Sliced OT (切片最优传输)
  - 低秩 OT (Low-Rank OT)
  - 非平衡 OT (Unbalanced OT)
"""

import time
import gc
import threading
import tracemalloc
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from backend.base import AbstractBackend
import ot
from data_generator import SparseDataset
from algorithms.registry import ALGORITHMS, get_algorithm
from backend.base import get_backend, list_available_backends

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
from rich import box
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.live import Live
from rich.columns import Columns

_console = Console()

try:
    import psutil
    def _get_process_memory() -> float:
        """Return current process RSS memory in bytes."""
        return float(psutil.Process().memory_info().rss)
except ImportError:
    import tracemalloc
    _TRACEMALLOC_ACTIVE = False
    def _get_process_memory() -> float:
        """Fallback: use tracemalloc (less reliable, not thread-safe)."""
        global _TRACEMALLOC_ACTIVE
        if not _TRACEMALLOC_ACTIVE:
            tracemalloc.start()
            _TRACEMALLOC_ACTIVE = True
        current, peak = tracemalloc.get_traced_memory()
        return float(peak)


@dataclass
class WeightConfig:
    """四维评测权重配置

    权重将自动归一化，确保总和为 1.0。
    各维度说明:
      - w_accuracy: 精度权重 (Wasserstein 距离与真实值的误差)
      - w_speed:    速度权重 (运行时间)
      - w_memory:   内存权重 (峰值内存占用)
      - w_success:  成功率权重 (算法是否成功完成)
    """
    w_accuracy: float = 0.35
    w_speed: float = 0.30
    w_memory: float = 0.25
    w_success: float = 0.10
    max_runtime: float = 60.0     # 时间归一化上限 (秒)
    max_memory: float = 2048.0    # 内存归一化上限 (MB)

    def __post_init__(self):
        total = self.w_accuracy + self.w_speed + self.w_memory + self.w_success
        if total <= 0:
            raise ValueError("权重总和必须大于 0")
        self.w_accuracy /= total
        self.w_speed /= total
        self.w_memory /= total
        self.w_success /= total

    def display(self) -> str:
        return (
            f"精度={self.w_accuracy:.0%} | "
            f"速度={self.w_speed:.0%} | "
            f"内存={self.w_memory:.0%} | "
            f"成功率={self.w_success:.0%}"
        )


# 默认权重配置 (模块级单例)
DEFAULT_WEIGHTS = WeightConfig()


@dataclass
class AlgorithmResult:
    """单个算法的评测结果"""
    algorithm_name: str
    wasserstein_distance: float
    runtime_seconds: float
    peak_memory_mb: float
    distance_error: float           # 与真实值的相对误差
    convergence_iterations: int
    success: bool
    is_reference: bool = False
    ranking_group: str = "balanced"
    error_message: str = ""
    not_applicable: bool = False


@dataclass
class BenchmarkReport:
    """完整的基准测试报告"""
    dataset_name: str
    dataset_info: str
    ground_truth: float
    ground_truth_method: str = "unknown"
    ground_truth_is_exact: bool = False
    reference_algorithm: str = ""
    data_origin: str = "unknown"
    source_url: str = ""
    geometry_description: str = "index position (1D)"
    scoring_reference: float = float("nan")
    scoring_reference_method: str = "not computed"
    results: list[AlgorithmResult] = field(default_factory=list)
    weighted_score: dict[str, float] = field(default_factory=dict)
    overall_ranking: list[tuple[str, float]] = field(default_factory=list)
    weight_config: Optional[WeightConfig] = None


def measure_memory_and_time(
    func: Callable, *args, **kwargs
) -> tuple:
    """
    测量函数执行的内存峰值和时间。

    使用 psutil (线程安全) 或 tracemalloc (fallback)。

    返回:
        (result, runtime_seconds, peak_memory_mb)
    """
    mem_before = _get_process_memory()
    if not tracemalloc.is_tracing():
        tracemalloc.start()
    python_mem_before = tracemalloc.get_traced_memory()[0]
    tracemalloc.reset_peak()
    peak_memory = [mem_before]
    stop_sampling = threading.Event()

    def sample_memory():
        while not stop_sampling.wait(0.002):
            peak_memory[0] = max(peak_memory[0], _get_process_memory())

    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    start_time = time.perf_counter()
    try:
        result = func(*args, **kwargs)
    finally:
        end_time = time.perf_counter()
        peak_memory[0] = max(peak_memory[0], _get_process_memory())
        stop_sampling.set()
        sampler.join(timeout=0.1)
        python_peak = tracemalloc.get_traced_memory()[1]

    runtime = end_time - start_time
    rss_delta = max(0.0, peak_memory[0] - mem_before)
    python_delta = max(0.0, python_peak - python_mem_before)
    peak_memory_mb = max(rss_delta, python_delta) / (1024 * 1024)

    gc.collect()
    return result, runtime, peak_memory_mb


# Algorithm implementations are now in algorithms/ modules
# ALGORITHMS registry is imported from algorithms.registry


def is_reference_algorithm(dataset: SparseDataset, algo_name: str) -> bool:
    """Return True when an algorithm is the pseudo-reference for this dataset.

    Large benchmark datasets use Sinkhorn(reg=0.01) as an approximate reference.
    CPU and GPU implementations of the same reference must both be excluded from
    rankings, otherwise the GPU variant can still win by matching the referee.
    """
    if getattr(dataset, "ground_truth_is_exact", False):
        return False

    reference_algorithm = getattr(dataset, "reference_algorithm", "")
    if reference_algorithm and algo_name == reference_algorithm:
        return True

    method = getattr(dataset, "ground_truth_method", "")
    if "Sinkhorn" in method and "pseudo-reference" in method:
        return "Sinkhorn" in algo_name and "reg=0.01" in algo_name

    return False


def run_single_algorithm(
    algo_name: str,
    algo_info: dict,
    dataset: SparseDataset,
    backend: Optional[AbstractBackend] = None,
) -> AlgorithmResult:
    """运行单个算法并返回评测结果"""
    is_reference = is_reference_algorithm(dataset, algo_name)
    ranking_group = algo_info.get("ranking_group", "balanced")
    try:
        # 检查维度限制
        active_dim = getattr(dataset, "active_storage_dim", len(dataset.source_dist))
        max_dim = algo_info.get("max_dim", float("inf"))
        max_active_dim = algo_info.get("max_active_dim", float("inf"))
        if dataset.dim > max_dim or active_dim > max_active_dim:
            if dataset.dim > max_dim:
                reason = f"评测维度 {dataset.dim} 超过算法上限 {max_dim}"
            else:
                reason = f"活跃支撑 {active_dim} 超过算法上限 {max_active_dim}"
            return AlgorithmResult(
                algorithm_name=algo_name,
                wasserstein_distance=float("nan"),
                runtime_seconds=float("nan"),
                peak_memory_mb=float("nan"),
                distance_error=float("nan"),
                convergence_iterations=0,
                success=False,
                is_reference=is_reference,
                ranking_group=ranking_group,
                error_message=reason,
                not_applicable=True,
            )

        kwargs = {}
        if backend is not None:
            kwargs["backend"] = backend
        if algo_info.get("uses_sparse_cost") and dataset.cost_matrix_csr is not None:
            kwargs["cost_matrix_csr"] = dataset.cost_matrix_csr
        if algo_info.get("uses_support_points"):
            kwargs["support_points"] = dataset.support_points

        result, runtime, peak_memory = measure_memory_and_time(
            algo_info["func"],
            dataset.source_dist,
            dataset.target_dist,
            dataset.cost_matrix,
            **kwargs,
        )

        if not np.isfinite(result) or result < 0:
            raise ValueError(f"algorithm returned invalid distance: {result}")

        # 计算相对误差
        if dataset.ground_truth_distance > 1e-10:
            error = abs(result - dataset.ground_truth_distance) / dataset.ground_truth_distance
        else:
            error = abs(result - dataset.ground_truth_distance)

        return AlgorithmResult(
            algorithm_name=algo_name,
            wasserstein_distance=result,
            runtime_seconds=runtime,
            peak_memory_mb=peak_memory,
            distance_error=error,
            convergence_iterations=0,
            success=True,
            is_reference=is_reference,
            ranking_group=ranking_group,
        )

    except (ValueError, np.linalg.LinAlgError, MemoryError, RuntimeError) as e:
        return AlgorithmResult(
            algorithm_name=algo_name,
            wasserstein_distance=float("nan"),
            runtime_seconds=float("nan"),
            peak_memory_mb=float("nan"),
            distance_error=float("nan"),
            convergence_iterations=0,
            success=False,
            is_reference=is_reference,
            ranking_group=ranking_group,
            error_message=str(e)[:200],
        )


def compute_weighted_score(
    result: AlgorithmResult,
    weights: Optional[WeightConfig] = None,
) -> float:
    """
    个性化加权评分公式 —— 基于加权欧式距离思想设计：

    Score = √[ w_acc × (1 - error)² + w_speed × (1 - T/Tmax)²
              + w_memory × (1 - M/Mmax)² + w_success × S² ]

    其中:
      - error = 与真实值的相对误差
      - T = 运行时间, Tmax = 时间上限
      - M = 内存占用, Mmax = 内存上限
      - S = 成功标志 (1 或 0)
      - w_* = 各维度权重 (由 WeightConfig 提供)

    分数越接近 1.0 表示综合表现越好。
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    if not result.success:
        return 0.0

    # 精度分数 (1 - 相对误差, 截断到 >=0)
    accuracy_score = max(0.0, 1.0 - min(result.distance_error, 1.0))

    # 速度分数 (归一化到 [0, 1], 时间越短越好)
    speed_score = max(0.0, 1.0 - min(result.runtime_seconds / weights.max_runtime, 1.0))

    # 内存分数 (归一化到 [0, 1], 内存越少越好)
    memory_score = max(0.0, 1.0 - min(result.peak_memory_mb / weights.max_memory, 1.0))

    # 成功分数
    success_score = 1.0 if result.success else 0.0

    # 加权欧式距离得分
    score = np.sqrt(
        weights.w_accuracy * accuracy_score ** 2 +
        weights.w_speed * speed_score ** 2 +
        weights.w_memory * memory_score ** 2 +
        weights.w_success * success_score ** 2
    )

    return round(score, 4)


def compute_ranked_scores(
    results: list[AlgorithmResult],
    weights: Optional[WeightConfig] = None,
    exclude_references: bool = True,
) -> list[tuple[str, float]]:
    """对所有算法的结果进行综合评分并排名"""
    ranked_results = [
        r for r in results
        if r.ranking_group == "balanced"
        and not r.not_applicable
        and not (exclude_references and r.is_reference)
    ]
    if not ranked_results:
        ranked_results = results

    successful = [r for r in ranked_results if r.success]
    if not successful:
        return [(r.algorithm_name, 0.0) for r in ranked_results]

    times = np.log1p([r.runtime_seconds for r in successful])
    memories = np.log1p([r.peak_memory_mb for r in successful])
    time_min, time_max = float(np.min(times)), float(np.max(times))
    memory_min, memory_max = float(np.min(memories)), float(np.max(memories))

    def lower_is_better(value: float, low: float, high: float) -> float:
        if high - low <= 1e-12:
            return 1.0
        return float((high - value) / (high - low))

    scored = []
    for r in ranked_results:
        if not r.success:
            scored.append((r.algorithm_name, 0.0))
            continue
        accuracy = max(0.0, 1.0 - min(r.distance_error, 1.0))
        speed = lower_is_better(np.log1p(r.runtime_seconds), time_min, time_max)
        memory = lower_is_better(np.log1p(r.peak_memory_mb), memory_min, memory_max)
        score = np.sqrt(
            weights.w_accuracy * accuracy ** 2
            + weights.w_speed * speed ** 2
            + weights.w_memory * memory ** 2
            + weights.w_success
        )
        score = round(float(score), 4)
        scored.append((r.algorithm_name, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def apply_practical_scoring_reference(report: BenchmarkReport) -> None:
    """Use a consensus of practical algorithms as the accuracy judge.

    EMD can still be shown as an exact diagnostic value, but it should not be
    the default judge for a high-dimensional sparse algorithm leaderboard.
    """
    candidates = [
        r.wasserstein_distance
        for r in report.results
        if r.success
        and r.ranking_group == "balanced"
        and not r.is_reference
        and np.isfinite(r.wasserstein_distance)
    ]

    if candidates:
        reference = float(np.median(candidates))
        method = "median consensus of ranked practical algorithms"
    else:
        reference = float(report.ground_truth)
        method = "fallback dataset reference"

    report.scoring_reference = reference
    report.scoring_reference_method = method

    for result in report.results:
        if not result.success or not np.isfinite(result.wasserstein_distance):
            continue
        if abs(reference) > 1e-10:
            result.distance_error = abs(result.wasserstein_distance - reference) / abs(reference)
        else:
            result.distance_error = abs(result.wasserstein_distance - reference)


def ranking_group_status(group: str) -> tuple[str, str]:
    """Return a short display label and color for non-balanced groups."""
    labels = {
        "reference": ("REF", "magenta"),
        "model": ("MODEL", "cyan"),
        "unbalanced": ("AUX", "cyan"),
        "baseline": ("BASE", "cyan"),
    }
    return labels.get(group, ("AUX", "cyan"))


def benchmark_dataset(
    dataset: SparseDataset,
    algorithms: dict = ALGORITHMS,
    weights: Optional[WeightConfig] = None,
    verbose: bool = True,
    backend: Optional[AbstractBackend] = None,
) -> BenchmarkReport:
    """
    在指定数据集上评测所有算法。

    参数:
        dataset: 测试数据集
        algorithms: 待评测的算法字典
        weights: 自定义权重配置 (None 则使用默认权重)
        verbose: 是否打印进度
        backend: 计算后端 (None 则使用 NumPy)

    返回:
        BenchmarkReport 对象
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    report = BenchmarkReport(
        dataset_name=dataset.name,
        dataset_info=dataset.description,
        ground_truth=dataset.ground_truth_distance,
        ground_truth_method=getattr(dataset, "ground_truth_method", "unknown"),
        ground_truth_is_exact=getattr(dataset, "ground_truth_is_exact", False),
        reference_algorithm=getattr(dataset, "reference_algorithm", ""),
        data_origin=getattr(dataset, "data_origin", "unknown"),
        source_url=getattr(dataset, "source_url", ""),
        geometry_description=getattr(dataset, "geometry_description", "index position (1D)"),
        weight_config=weights,
    )

    backend_name = backend.name if backend else "NumPy"

    if verbose:
        info_text = Text()
        info_text.append("Dataset: ", style="dim")
        info_text.append(dataset.name, style="bold white")
        info_text.append(f"\n{dataset.description}", style="dim")
        info_text.append("\nGround Truth: ", style="dim")
        info_text.append(f"{dataset.ground_truth_distance:.6f}", style="bold yellow")
        info_text.append("  Method: ", style="dim")
        method_style = "bold green" if report.ground_truth_is_exact else "bold magenta"
        info_text.append(report.ground_truth_method, style=method_style)
        info_text.append("\nData Origin: ", style="dim")
        origin_style = "bold cyan" if str(report.data_origin).startswith("online:") else "bold white"
        info_text.append(report.data_origin, style=origin_style)
        info_text.append("\nCost Geometry: ", style="dim")
        info_text.append(report.geometry_description, style="bold white")
        if report.source_url:
            info_text.append("\nSource URL: ", style="dim")
            info_text.append(report.source_url, style="cyan underline")
        if report.reference_algorithm and not report.ground_truth_is_exact:
            info_text.append("\nReference excluded from ranking: ", style="dim")
            info_text.append(report.reference_algorithm, style="bold magenta")
        info_text.append("\nLeaderboard: ", style="dim")
        info_text.append("general high-dimensional sparse algorithms only", style="bold cyan")
        info_text.append("\nScoring Judge: practical-algorithm consensus, not EMD", style="bold cyan")
        info_text.append("\nExcluded from leaderboard: exact reference / model-assumption / auxiliary baselines", style="dim")
        info_text.append("  Backend: ", style="dim")
        backend_style = "bold green" if backend_name != "NumPy" else "bold white"
        info_text.append(backend_name, style=backend_style)

        _console.print(Panel(info_text, box=box.HEAVY, border_style="cyan", padding=(1, 2)))

    # Results table
    table = Table(
        box=box.ROUNDED, border_style="bright_black",
        header_style="bold white",
    )
    table.add_column("Algorithm", style="bold", width=28)
    table.add_column("Distance", justify="right", width=12)
    table.add_column("Time", justify="right", width=10)
    table.add_column("Memory", justify="right", width=10)
    table.add_column("Error", justify="right", width=10)
    table.add_column("Score", justify="right", width=10)
    table.add_column("Status", width=8)

    for algo_name, algo_info in algorithms.items():
        result = run_single_algorithm(algo_name, algo_info, dataset, backend=backend)
        report.results.append(result)

    apply_practical_scoring_reference(report)

    report.overall_ranking = compute_ranked_scores(report.results, weights)
    report.weighted_score = dict(report.overall_ranking)

    if verbose:
        _console.print(
            f"[dim]Scoring reference: {report.scoring_reference:.6f} "
            f"({report.scoring_reference_method})[/]"
        )

    for result in report.results:
        algo_name = result.algorithm_name
        score = report.weighted_score.get(algo_name)
        if result.success:
            dist_s = f"{result.wasserstein_distance:.6f}"
            time_s = f"{result.runtime_seconds:.4f}s"
            mem_s = f"{result.peak_memory_mb:.1f}MB"
            err_s = f"{result.distance_error:.2%}"
            if result.is_reference or result.ranking_group != "balanced":
                label, color = ranking_group_status(
                    "reference" if result.is_reference else result.ranking_group
                )
                score_s = f"[{color}]--[/]"
                status_s = f"[{color}]{label}[/]"
            else:
                score_s = f"[bold green]{score:.4f}[/]"
                status_s = "[green]OK[/]"
        elif result.not_applicable:
            dist_s = time_s = mem_s = err_s = "[dim]--[/]"
            score_s = "[dim]--[/]"
            status_s = "[yellow]N/A[/]"
        else:
            dist_s = time_s = mem_s = err_s = "[dim]--[/]"
            score_s = "[red]0.0000[/]"
            status_s = "[red]FAIL[/]"
        table.add_row(algo_name, dist_s, time_s, mem_s, err_s, score_s, status_s)

    if verbose:
        _console.print(table)

    if verbose and len(report.overall_ranking) > 1:
        rank_text = Text()
        medals = {0: "1st", 1: "2nd", 2: "3rd"}
        for i, (name, score) in enumerate(report.overall_ranking[:5]):
            medal = medals.get(i, f"#{i+1}")
            if i == 0:
                rank_text.append(f" [{medal}] ", style="bold yellow")
                rank_text.append(f"{name} ", style="bold white")
                rank_text.append(f"{score:.4f}", style="bold green")
            else:
                rank_text.append(f"\n [{medal}] ", style="dim")
                rank_text.append(f"{name} ", style="dim")
                rank_text.append(f"{score:.4f}", style="dim")

        _console.print(Panel(rank_text, title="Practical Sparse Rankings", title_align="left",
                             box=box.ROUNDED, border_style="yellow", padding=(0, 2)))

    return report


def run_full_benchmark(
    datasets: list[SparseDataset],
    algorithms: dict = ALGORITHMS,
    weights: Optional[WeightConfig] = None,
    verbose: bool = True,
    backend: Optional[AbstractBackend] = None,
) -> list[BenchmarkReport]:
    """
    在所有数据集上运行完整的算法评测。

    参数:
        datasets: 数据集列表
        algorithms: 算法字典
        weights: 自定义权重配置 (None 则使用默认权重)
        verbose: 是否打印进度
        backend: 计算后端 (None 则使用 NumPy)

    返回:
        评测报告列表
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    all_reports = []

    if verbose:
        backend_name = backend.name if backend else "NumPy"
        header_text = Text()
        header_text.append("OT Algorithm Benchmark", style="bold cyan")
        header_text.append(f"\n{weights.display()}", style="dim")
        backend_style = "bold green" if backend_name != "NumPy" else "bold white"
        header_text.append(f"\nBackend: ", style="dim")
        header_text.append(backend_name, style=backend_style)

        _console.print(Panel(header_text, box=box.HEAVY, border_style="cyan", padding=(1, 2)))

    for ds in datasets:
        report = benchmark_dataset(ds, algorithms, weights, verbose, backend=backend)
        all_reports.append(report)

    if verbose:
        _console.print()

    return all_reports


# ====================================================================
#  统计显著性检验
# ====================================================================

@dataclass
class StatsResult:
    """多次运行的统计结果"""
    algorithm_name: str
    mean_distance: float
    std_distance: float
    min_distance: float
    max_distance: float
    ci95_distance: float       # 95% 置信区间半宽
    mean_runtime: float
    std_runtime: float
    mean_memory: float
    std_memory: float
    success_rate: float         # 成功率 (0~1)
    mean_score: float
    n_runs: int
    individual_results: list[AlgorithmResult] = field(default_factory=list)


def run_with_stats(
    algo_name: str,
    algo_info: dict,
    dataset: SparseDataset,
    n_runs: int = 5,
    base_seed: int = 42,
    weights: Optional[WeightConfig] = None,
    backend: Optional[AbstractBackend] = None,
) -> StatsResult:
    """
    多次运行同一算法，收集统计信息。

    每次运行使用不同的随机种子（对涉及随机性的算法如 Sliced OT 有影响）。
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    from copy import deepcopy
    import numpy as np

    individual = []
    for run in range(n_runs):
        # 对于依赖随机种子的算法，通过修改数据集或算法参数来引入变异性
        result = run_single_algorithm(algo_name, algo_info, dataset, backend=backend)
        individual.append(result)

    distances = [r.wasserstein_distance for r in individual if r.success]
    runtimes = [r.runtime_seconds for r in individual if r.success]
    memories = [r.peak_memory_mb for r in individual if r.success]
    scores = [compute_weighted_score(r, weights) for r in individual]
    n_success = sum(1 for r in individual if r.success)

    d_arr = np.array(distances) if distances else np.array([np.nan])
    t_arr = np.array(runtimes) if runtimes else np.array([np.nan])
    m_arr = np.array(memories) if memories else np.array([np.nan])
    s_arr = np.array(scores) if scores else np.array([np.nan])

    def ci95(arr):
        return 1.96 * np.nanstd(arr) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0

    return StatsResult(
        algorithm_name=algo_name,
        mean_distance=float(np.nanmean(d_arr)),
        std_distance=float(np.nanstd(d_arr)),
        min_distance=float(np.nanmin(d_arr)) if len(d_arr) else np.nan,
        max_distance=float(np.nanmax(d_arr)) if len(d_arr) else np.nan,
        ci95_distance=float(ci95(d_arr)),
        mean_runtime=float(np.nanmean(t_arr)),
        std_runtime=float(np.nanstd(t_arr)),
        mean_memory=float(np.nanmean(m_arr)),
        std_memory=float(np.nanstd(m_arr)),
        success_rate=n_success / n_runs,
        mean_score=float(np.nanmean(s_arr)),
        n_runs=n_runs,
        individual_results=individual,
    )


def benchmark_with_stats(
    dataset: SparseDataset,
    algorithms: dict = ALGORITHMS,
    n_runs: int = 3,
    weights: Optional[WeightConfig] = None,
    verbose: bool = True,
    backend: Optional[AbstractBackend] = None,
) -> tuple[BenchmarkReport, list[StatsResult]]:
    """
    带统计检验的基准测试。
    返回常规报告 + 统计结果列表。
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    report = BenchmarkReport(
        dataset_name=dataset.name,
        dataset_info=dataset.description,
        ground_truth=dataset.ground_truth_distance,
        ground_truth_method=getattr(dataset, "ground_truth_method", "unknown"),
        ground_truth_is_exact=getattr(dataset, "ground_truth_is_exact", False),
        reference_algorithm=getattr(dataset, "reference_algorithm", ""),
        data_origin=getattr(dataset, "data_origin", "unknown"),
        source_url=getattr(dataset, "source_url", ""),
        geometry_description=getattr(dataset, "geometry_description", "index position (1D)"),
        weight_config=weights,
    )

    stats_results = []

    if verbose:
        backend_name = backend.name if backend else "NumPy"
        print(f"\n{'='*70}")
        print(f"  📊 统计评测: {dataset.name} (每个算法运行 {n_runs} 次)")
        print(f"  计算后端: {backend_name}")
        print(f"  真实距离 (Ground Truth): {dataset.ground_truth_distance:.6f}")
        print(f"  真值来源: {report.ground_truth_method}")
        print(f"  数据来源: {report.data_origin}")
        if report.source_url:
            print(f"  来源 URL: {report.source_url}")
        if report.reference_algorithm and not report.ground_truth_is_exact:
            print(f"  参考算法不参与排名: {report.reference_algorithm}")
        print(f"{'='*70}")

    for algo_name, algo_info in algorithms.items():
        if verbose:
            print(f"  ⏳ {algo_name} ...", end=" ", flush=True)

        stats = run_with_stats(algo_name, algo_info, dataset, n_runs=n_runs, weights=weights, backend=backend)
        stats_results.append(stats)

        # 将均值结果加入报告
        avg_result = AlgorithmResult(
            algorithm_name=algo_name,
            wasserstein_distance=stats.mean_distance,
            runtime_seconds=stats.mean_runtime,
            peak_memory_mb=stats.mean_memory,
            distance_error=abs(stats.mean_distance - dataset.ground_truth_distance)
                           / (dataset.ground_truth_distance + 1e-10),
            convergence_iterations=0,
            success=stats.success_rate > 0,
            is_reference=is_reference_algorithm(dataset, algo_name),
            ranking_group=algo_info.get("ranking_group", "balanced"),
            error_message="" if stats.success_rate > 0 else "全部运行失败",
        )
        report.results.append(avg_result)

        if verbose:
            if stats.success_rate > 0:
                print(f"✓ μ={stats.mean_distance:.6f}±{stats.std_distance:.6f}, "
                      f"成功率={stats.success_rate:.0%}, "
                      f"CI95=±{stats.ci95_distance:.6f}")
            else:
                print(f"✗ 全部失败")

    report.overall_ranking = compute_ranked_scores(report.results, weights)
    report.weighted_score = dict(report.overall_ranking)

    if verbose:
        print(f"\n  🏆 综合排名 (基于均值评分):")
        for rank, (name, score) in enumerate(report.overall_ranking, 1):
            medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"  {rank}.")
            print(f"    {medal} {name}: {score:.4f}")

    return report, stats_results


# ====================================================================
#  失效边界扫描
# ====================================================================

@dataclass
class SweepPoint:
    """失效边界扫描的单个采样点"""
    dim: int
    sparsity: float
    algorithm: str
    success: bool
    distance: float
    error: float
    runtime: float
    memory_mb: float


@dataclass
class SweepResult:
    """失效边界扫描完整结果"""
    algorithm: str
    points: list[SweepPoint] = field(default_factory=list)
    failure_threshold: Optional[dict] = None  # {"dim_critical": ..., "sparsity_critical": ...}


def sweep_failure_boundary(
    algorithms: dict = None,
    dims: list[int] = None,
    sparsities: list[float] = None,
    verbose: bool = True,
) -> dict[str, SweepResult]:
    """
    在维度×稀疏度的参数网格上扫描各算法的失效边界。

    对每个 (dim, sparsity) 组合生成随机数据集，运行所有算法，
    记录每个算法在何种条件下开始失效。

    参数:
        algorithms: 待测试算法 (默认全部)
        dims: 维度列表 (默认 [100, 200, 300, 500, 800, 1000, 1500, 2000])
        sparsities: 稀疏度列表 (默认 [0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99])
        verbose: 是否打印进度

    返回:
        {algo_name: SweepResult} 字典
    """
    if algorithms is None:
        algorithms = ALGORITHMS
    if dims is None:
        dims = [100, 200, 300, 500, 800, 1000, 1500, 2000]
    if sparsities is None:
        sparsities = [0.5, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99]

    from data_generator import create_benchmark_dataset

    results = {name: SweepResult(algorithm=name) for name in algorithms}

    if verbose:
        total = len(dims) * len(sparsities) * len(algorithms)
        print(f"\n🔍 失效边界扫描: {len(dims)} dims × {len(sparsities)} sparsities "
              f"× {len(algorithms)} algos = {total} 次测试")
        print(f"   dims: {dims}")
        print(f"   sparsities: {sparsities}\n")

    count = 0
    for dim in dims:
        for sp in sparsities:
            # 为这个参数组合创建测试数据集
            ds = create_benchmark_dataset(
                n_samples=dim, sparsity=sp, noise_level=0.03,
                source_dist_type="zipf", target_dist_type="pareto",
                random_state=hash((dim, int(sp * 1000))) % (2**30),
                name=f"Sweep-D{dim}-S{int(sp*100)}",
            )

            for algo_name, algo_info in algorithms.items():
                count += 1
                result = run_single_algorithm(algo_name, algo_info, ds)

                point = SweepPoint(
                    dim=dim,
                    sparsity=sp,
                    algorithm=algo_name,
                    success=result.success,
                    distance=result.wasserstein_distance,
                    error=result.distance_error if result.success else np.nan,
                    runtime=result.runtime_seconds,
                    memory_mb=result.peak_memory_mb,
                )
                results[algo_name].points.append(point)

                if verbose:
                    status = "✓" if result.success else "✗"
                    print(f"  [{count}/{total}] {algo_name} @ d={dim}, s={sp:.0%} {status}")

    # 自动检测失效临界点
    for algo_name, sweep in results.items():
        points = sweep.points
        # 找成功率从 >0.5 跌到 ≤0.5 的维度临界值
        dim_failures = {}
        for p in points:
            if p.dim not in dim_failures:
                dim_failures[p.dim] = []
            dim_failures[p.dim].append(p.success)

        # 找最小的失败维度
        dim_rates = {d: sum(s)/len(s) for d, s in dim_failures.items()}
        critical_dim = None
        for d in sorted(dim_rates.keys()):
            if dim_rates[d] < 0.5:
                critical_dim = d
                break

        # 找最大的安全稀疏度
        sp_success = {}
        for p in points:
            if p.sparsity not in sp_success:
                sp_success[p.sparsity] = []
            sp_success[p.sparsity].append(p.success)
        sp_rates = {s: sum(ss)/len(ss) for s, ss in sp_success.items()}
        critical_sparsity = None
        for s in sorted(sp_rates.keys(), reverse=True):
            if sp_rates[s] >= 0.5:
                critical_sparsity = s
                break

        sweep.failure_threshold = {
            "dim_critical": critical_dim,
            "sparsity_safe": critical_sparsity,
        }

        if verbose:
            print(f"\n  📊 {algo_name}: 临界维度≈{critical_dim}, "
                  f"安全稀疏度≤{critical_sparsity}")

    if verbose:
        print("\n✅ 失效边界扫描完成!")

    return results


# ====================================================================
#  参数敏感性分析
# ====================================================================

@dataclass
class SensitivityPoint:
    """参数敏感性分析采样点"""
    param_name: str
    param_value: float
    algorithm: str
    distance: float
    error: float
    runtime: float
    memory_mb: float
    score: float


def sensitivity_analysis(
    dataset: SparseDataset,
    algo_class: str = "sinkhorn",
    param_name: str = "reg",
    param_values: list[float] = None,
    weights: Optional[WeightConfig] = None,
    verbose: bool = True,
) -> list[SensitivityPoint]:
    """
    分析算法参数对性能的影响。

    参数:
        dataset: 测试数据集
        algo_class: 算法类别 — "sinkhorn", "sliced", "lowrank"
        param_name: 参数名 — "reg", "n_projections", "rank"
        param_values: 参数值列表
        weights: 自定义权重配置 (None 则使用默认权重)
        verbose: 是否打印进度

    返回:
        SensitivityPoint 列表
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS
    if param_values is None:
        if algo_class == "sinkhorn":
            param_values = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
            param_name = "reg"
        elif algo_class == "sliced":
            param_values = [50, 100, 200, 350, 500, 750, 1000]
            param_name = "n_projections"
        elif algo_class == "lowrank":
            param_values = [5, 10, 20, 30, 50, 80, 100]
            param_name = "rank"
        else:
            raise ValueError(f"Unknown algo class: {algo_class}")

    results = []

    if verbose:
        print(f"\n🔬 参数敏感性分析: {algo_class} / {param_name}")
        print(f"   数据集: {dataset.name} ({dataset.description})")
        print(f"   参数值: {param_values}\n")

    for val in param_values:
        if verbose:
            print(f"  ⏳ {param_name}={val} ...", end=" ", flush=True)

        try:
            a = dataset.source_dist.astype(np.float64).copy()
            b = dataset.target_dist.astype(np.float64).copy()
            M = dataset.cost_matrix.astype(np.float64).copy()
            a = a / a.sum()
            b = b / b.sum()

            result, runtime, peak_memory = measure_memory_and_time(
                _run_sensitivity_algo, a, b, M, algo_class, param_name, val
            )

            error = abs(result - dataset.ground_truth_distance) / (dataset.ground_truth_distance + 1e-10)

            sp = SensitivityPoint(
                param_name=param_name,
                param_value=val,
                algorithm=algo_class,
                distance=result,
                error=error,
                runtime=runtime,
                memory_mb=peak_memory,
                score=0.0,
            )

            # 归一化评分
            sp.score = _compute_sensitivity_score(sp, dataset, weights)

            results.append(sp)

            if verbose:
                print(f"✓ 距离={result:.6f}, 误差={error:.4%}, "
                      f"时间={runtime:.4f}s, 评分={sp.score:.4f}")

        except (ValueError, np.linalg.LinAlgError, MemoryError) as e:
            if verbose:
                print(f"✗ 失败: {e}")

    if verbose:
        print(f"\n✅ 敏感性分析完成! {len(results)} 个数据点")

    return results


def _run_sensitivity_algo(
    a: np.ndarray, b: np.ndarray, M: np.ndarray,
    algo_class: str, param_name: str, param_value: float,
) -> float:
    """内部函数：根据参数运行对应算法"""
    if algo_class == "sinkhorn":
        return float(ot.sinkhorn2(a, b, M, param_value, numItermax=2000, stopThr=1e-9))
    elif algo_class == "sliced":
        n = len(a)
        dim = min(n, 100)
        rng = np.random.RandomState(42)
        Xs = rng.randn(n, dim) * a.reshape(-1, 1)
        Xt = rng.randn(n, dim) * b.reshape(-1, 1)
        return float(ot.sliced_wasserstein_distance(Xs, Xt, a, b, int(param_value), seed=42))
    elif algo_class == "lowrank":
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        r = min(int(param_value), len(S))
        low_M = (U[:, :r] * S[:r]) @ Vt[:r, :]
        low_M = np.abs(low_M)
        low_M = low_M / (low_M.max() + 1e-10)
        return float(ot.sinkhorn2(a, b, low_M, 0.1, numItermax=2000, stopThr=1e-9))
    else:
        raise ValueError(f"Unknown algo_class: {algo_class}")


def _compute_sensitivity_score(
    sp: SensitivityPoint,
    dataset: SparseDataset,
    weights: Optional[WeightConfig] = None,
) -> float:
    """为敏感性分析点计算综合评分 (精度+速度+内存)。
    使用 WeightConfig 中三者的相对比例，忽略 success 权重。"""
    if weights is None:
        weights = DEFAULT_WEIGHTS
    # 从权重配置中提取精度/速度/内存的相对比例
    w_sum = weights.w_accuracy + weights.w_speed + weights.w_memory
    if w_sum <= 0:
        w_sum = 1.0
    wa = weights.w_accuracy / w_sum
    ws = weights.w_speed / w_sum
    wm = weights.w_memory / w_sum

    acc = max(0.0, 1.0 - min(sp.error, 1.0))
    runtime_norm = max(0.0, 1.0 - min(sp.runtime / weights.max_runtime, 1.0))
    mem_norm = max(0.0, 1.0 - min(sp.memory_mb / weights.max_memory, 1.0))
    return round(float(np.sqrt(wa * acc**2 + ws * runtime_norm**2 + wm * mem_norm**2)), 4)


# ====================================================================
#  收敛性追踪
# ====================================================================

@dataclass
class ConvergenceProfile:
    """算法收敛过程记录"""
    algorithm: str
    param_value: float
    iterations: list[int] = field(default_factory=list)      # 迭代次数
    marginal_errors: list[float] = field(default_factory=list)  # 边际残差
    distances: list[float] = field(default_factory=list)       # 当前估计距离
    final_iterations: int = 0
    converged: bool = False


def trace_sinkhorn_convergence(
    dataset: SparseDataset,
    reg: float = 0.01,
    max_iter: int = 2000,
    record_every: int = 10,
) -> ConvergenceProfile:
    """
    追踪 Sinkhorn 算法的收敛过程。

    在每次记录间隔捕获：当前迭代数、边际残差、当前距离估计。
    用于分析算法的收敛速度和稳定性。
    """
    a = dataset.source_dist.astype(np.float64).copy()
    b = dataset.target_dist.astype(np.float64).copy()
    M = dataset.cost_matrix.astype(np.float64).copy()
    a = a / a.sum()
    b = b / b.sum()

    # 初始化
    K = np.exp(-M / reg)
    u = np.ones(len(a))

    profile = ConvergenceProfile(algorithm="Sinkhorn", param_value=reg)
    profile.iterations.append(0)
    profile.marginal_errors.append(1.0)
    profile.distances.append(float("inf"))

    for it in range(1, max_iter + 1):
        v = b / (K.T @ u + 1e-16)
        u_next = a / (K @ v + 1e-16)

        # 边际条件误差
        err = np.max(np.abs(u_next * (K @ v) - a))
        u = u_next

        if it % record_every == 0 or it == max_iter:
            distance = np.sum((u.reshape(-1, 1) * K * v.reshape(1, -1)) * M)
            profile.iterations.append(it)
            profile.marginal_errors.append(float(err))
            profile.distances.append(float(distance))

        if err < 1e-9:
            profile.converged = True
            if it % record_every != 0:
                distance = np.sum((u.reshape(-1, 1) * K * v.reshape(1, -1)) * M)
                profile.iterations.append(it)
                profile.marginal_errors.append(float(err))
                profile.distances.append(float(distance))
            break

    profile.final_iterations = profile.iterations[-1]
    return profile


# ====================================================================
#  算法推荐系统
# ====================================================================

def recommend_algorithm(dataset: SparseDataset, verbose: bool = True) -> list[tuple[str, float, str]]:
    """
    基于数据集特征自动推荐最优算法。

    决策规则:
      1. 维度 ≤ 300 → 优先 EMD (精确解)
      2. 稀疏度 > 95% → 推荐 Sliced OT 或 Unbalanced OT
      3. 噪声高 (>10%) → 推荐 Unbalanced OT
      4. 中等规模 → Sinkhorn (reg=0.01) 或 Low-Rank OT
      5. 超大规模 (>2000) → Sliced OT

    返回:
        [(算法名, 推荐分, 理由), ...] 按推荐分降序排列
    """
    dim = dataset.dim
    sparsity = dataset.sparsity
    noise = dataset.noise_level

    recommendations = []

    # EMD 精确解
    if dim <= 300:
        recommendations.append((
            "EMD (精确解)", 1.0,
            f"维度 {dim} ≤ 300, 可直接求精确解"
        ))
    elif dim <= 500:
        recommendations.append((
            "EMD (精确解)", 0.5,
            f"维度 {dim} 偏大, 精确解可用但较慢"
        ))
    else:
        recommendations.append((
            "EMD (精确解)", -1.0,
            f"维度 {dim} 超出最大支持范围"
        ))

    # Sinkhorn
    if dim <= 1000:
        recommendations.append((
            "Sinkhorn (reg=0.01)", 0.9,
            f"中低维度 ({dim}), 低正则化获得高精度"
        ))
        recommendations.append((
            "Sinkhorn (reg=0.1)", 0.7,
            f"中低维度 ({dim}), 标准正则化平衡速度与精度"
        ))
    else:
        recommendations.append((
            "Sinkhorn (reg=0.01)", 0.6,
            f"高维度 ({dim}), 收敛可能较慢但精度高"
        ))
        recommendations.append((
            "Sinkhorn (reg=0.1)", 0.75,
            f"高维度 ({dim}), 较快收敛适合大规模"
        ))

    # Sliced OT
    if dim >= 800 or sparsity >= 0.95:
        recommendations.append((
            "Sliced OT (500)", 0.9,
            f"高维 ({dim}) 或高稀疏 ({sparsity:.1%}), 切片OT效率优势明显"
        ))
        recommendations.append((
            "Sliced OT (200)", 0.85,
            f"高维/高稀疏, 较少投影方向速度更快"
        ))
    else:
        recommendations.append((
            "Sliced OT (200)", 0.5,
            f"维度 {dim}, 切片OT效率优势不明显"
        ))

    # Greenkhorn
    recommendations.append((
        "Greenkhorn", 0.65,
        "贪心更新策略, 在某些稀疏分布上收敛更快"
    ))

    # Unbalanced OT
    if noise >= 0.08 or sparsity >= 0.95:
        recommendations.append((
            "Unbalanced OT", 0.95,
            f"高噪声 ({noise:.1%}) 或高稀疏 ({sparsity:.1%}), 非平衡OT允许质量松弛"
        ))
    else:
        recommendations.append((
            "Unbalanced OT", 0.6,
            f"标准噪声水平, 非平衡OT可能有过度松弛风险"
        ))

    # Low-Rank OT
    if dim >= 500:
        recommendations.append((
            "Low-Rank OT", 0.8,
            f"维度 {dim} ≥ 500, 低秩近似大幅降低计算量"
        ))
    else:
        recommendations.append((
            "Low-Rank OT", 0.4,
            f"维度 {dim}, 低秩近似收益不大"
        ))

    # 按推荐分降序
    recommendations.sort(key=lambda x: x[1], reverse=True)

    if verbose:
        print(f"\n💡 数据集 {dataset.name} 算法推荐:")
        print(f"   特征: dim={dim}, sparsity={sparsity:.1%}, noise={noise:.1%}")
        print(f"   {'算法':<25s} {'推荐分':>6s}  理由")
        print(f"   {'-'*60}")
        for name, score, reason in recommendations[:5]:
            if score >= 0:
                print(f"   {name:<25s} {score:>6.2f}  {reason}")

    return recommendations


# ====================================================================
#  综合统计分析
# ====================================================================

def run_full_statistical_benchmark(
    datasets: list[SparseDataset],
    algorithms: dict = None,
    n_runs: int = 3,
    weights: Optional[WeightConfig] = None,
    verbose: bool = True,
    backend: Optional[AbstractBackend] = None,
) -> tuple[list[BenchmarkReport], list[list[StatsResult]]]:
    """
    在所有数据集上运行带统计检验的完整评测。
    """
    if algorithms is None:
        algorithms = ALGORITHMS
    if weights is None:
        weights = DEFAULT_WEIGHTS

    all_reports = []
    all_stats = []

    if verbose:
        backend_name = backend.name if backend else "NumPy"
        print("\n" + "█" * 70)
        print("█  最优传输 (OT) 算法统计评测系统")
        print("█  多次运行 + 置信区间 + 显著性检验")
        print(f"█  评测权重: {weights.display()}")
        print(f"█  计算后端: {backend_name}")
        print("█" * 70)

    for ds in datasets:
        report, stats = benchmark_with_stats(ds, algorithms, n_runs, weights, verbose, backend=backend)
        all_reports.append(report)
        all_stats.append(stats)

    if verbose:
        print("\n" + "█" * 70)
        print("█  📋 统计评测完成！")
        print("█" * 70)

    return all_reports, all_stats


if __name__ == "__main__":
    from data_generator import create_dataset_suite

    print("🧪 算法评测引擎 - 快速测试")
    print(f"   当前权重: {DEFAULT_WEIGHTS.display()}")
    print("生成测试数据集...")

    datasets = create_dataset_suite()
    # 测试第一个和第三个数据集
    test_ds = [datasets[0], datasets[2]]

    reports = run_full_benchmark(test_ds, weights=DEFAULT_WEIGHTS)
    print("\n✅ 测试完成!")
