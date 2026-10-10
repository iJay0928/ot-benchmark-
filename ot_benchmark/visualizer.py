"""
可视化与报告生成模块
====================
生成算法对比图表和评测报告。
包括:
  - 算法性能对比柱状图 (时间/内存/精度)
  - 综合评分雷达图
  - 失效边界图 (维度 vs 稀疏度)
  - 评分热力图
  - 自动评测汇总表
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 非交互式后端
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyBboxPatch
from dataclasses import dataclass
from typing import Optional
import os

from benchmark_engine import (
    BenchmarkReport, AlgorithmResult,
    SweepResult, SweepPoint,
    SensitivityPoint, ConvergenceProfile,
    StatsResult,
    ranking_group_status,
)
from data_generator import SparseDataset


# 配色方案：Slate/Indigo 暗色主题
COLORS = {
    "bg_dark": "#0f172a",
    "bg_card": "#1e293b",
    "primary": "#6366f1",
    "accent": "#f59e0b",
    "gold": "#fbbf24",
    "silver": "#94a3b8",
    "bronze": "#d97706",
    "success": "#22c55e",
    "warning": "#f59e0b",
    "error": "#ef4444",
    "text": "#e2e8f0",
    "text_dim": "#64748b",
    "algo_colors": [
        "#6366f1", "#22c55e", "#f59e0b", "#ef4444",
        "#06b6d4", "#8b5cf6", "#f97316", "#10b981",
        "#ec4899", "#14b8a6", "#e11d48", "#3b82f6", "#84cc16",
    ],
}


def _cycle_colors(n: int) -> list[str]:
    """Return n colors by cycling the configured palette."""
    palette = COLORS["algo_colors"]
    return [palette[i % len(palette)] for i in range(n)]


def _find_cjk_font() -> str:
    """查找可用的中文字体"""
    import matplotlib.font_manager as fm
    candidates = ["Noto Sans SC", "Microsoft YaHei", "SimHei", "SimSun",
                  "STXihei", "KaiTi", "FangSong"]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            return font
    return "sans-serif"


def setup_dark_style():
    """设置暗色图表风格"""
    cjk_font = _find_cjk_font()
    plt.rcParams.update({
        "figure.facecolor": COLORS["bg_dark"],
        "axes.facecolor": COLORS["bg_card"],
        "axes.edgecolor": "#334155",
        "axes.labelcolor": COLORS["text"],
        "axes.titlecolor": COLORS["text"],
        "text.color": COLORS["text"],
        "xtick.color": COLORS["text_dim"],
        "ytick.color": COLORS["text_dim"],
        "grid.color": "#1e293b",
        "grid.alpha": 0.6,
        "grid.linewidth": 0.5,
        "legend.facecolor": COLORS["bg_dark"],
        "legend.edgecolor": "#334155",
        "legend.labelcolor": COLORS["text"],
        "legend.framealpha": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "font.family": "sans-serif",
        "font.sans-serif": [cjk_font, "DejaVu Sans", "Arial"],
        "axes.unicode_minus": False,
        "figure.titlesize": 14,
        "figure.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 9,
    })


def _save_figure(fig, save_path: str) -> str:
    """Save a matplotlib figure, falling back to a timestamped file if locked."""
    try:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=COLORS["bg_dark"], edgecolor="none")
        return save_path
    except PermissionError:
        from datetime import datetime
        root, ext = os.path.splitext(save_path)
        fallback = f"{root}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
        fig.savefig(fallback, dpi=150, bbox_inches="tight",
                    facecolor=COLORS["bg_dark"], edgecolor="none")
        return fallback


def plot_algorithm_comparison(
    report: BenchmarkReport,
    save_path: Optional[str] = None,
) -> str:
    """
    算法对比三面板图：运行时间、内存占用、精度误差。
    模拟 PPT 演示中的对比表格的可视化版本。
    """
    setup_dark_style()

    successful = [r for r in report.results if r.success and not getattr(r, "is_reference", False)]
    if not successful:
        return ""

    names = [r.algorithm_name for r in successful]
    times = [r.runtime_seconds for r in successful]
    memories = [r.peak_memory_mb for r in successful]
    errors = [r.distance_error * 100 for r in successful]

    colors = _cycle_colors(len(names))
    x = np.arange(len(names))
    width = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    fig.patch.set_facecolor(COLORS["bg_dark"])
    fig.subplots_adjust(wspace=0.25)

    # 面板 1: 运行时间
    ax = axes[0]
    bars = ax.bar(x, times, width * 2.5, color=colors, edgecolor="none",
                  alpha=0.9, zorder=3)
    ax.set_title("Run Time (seconds)", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Seconds")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    for bar, val in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(times) * 0.02,
                f"{val:.3f}s", ha="center", va="bottom", fontsize=7, color=COLORS["text"])

    # 面板 2: 内存占用
    ax = axes[1]
    bars = ax.bar(x, memories, width * 2.5, color=colors, edgecolor="none",
                  alpha=0.9, zorder=3)
    ax.set_title("Peak Memory (MB)", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("MB")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    for bar, val in zip(bars, memories):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(memories) * 0.02,
                f"{val:.1f}", ha="center", va="bottom", fontsize=7, color=COLORS["text"])

    # 面板 3: 相对误差
    ax = axes[2]
    bars = ax.bar(x, errors, width * 2.5, color=colors, edgecolor="none",
                  alpha=0.9, zorder=3)
    ax.set_title("Relative Error (%)", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Error %")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    for bar, val in zip(bars, errors):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(errors) * 0.02,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=7, color=COLORS["text"])

    fig.suptitle(f"Algorithm Comparison  —  {report.dataset_name}",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_path:
        save_path = _save_figure(fig, save_path)
    plt.close(fig)

    return save_path or ""


def plot_radar_score(
    report: BenchmarkReport,
    save_path: Optional[str] = None,
) -> str:
    """
    综合评分雷达图：展示各算法在多维度上的表现。
    对标 PPT 提到的"多维度动态评测体系"。
    """
    setup_dark_style()

    successful = [r for r in report.results if r.success and not getattr(r, "is_reference", False)]
    if len(successful) < 2:
        return ""

    categories = ["精度", "速度", "内存效率", "稳定性"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor(COLORS["bg_dark"])
    ax.set_facecolor(COLORS["bg_card"])

    colors = _cycle_colors(len(successful))

    for idx, result in enumerate(successful):
        # 计算各维度分数
        max_t = max(r.runtime_seconds for r in successful) or 1
        max_m = max(r.peak_memory_mb for r in successful) or 1

        acc = max(0, 1 - result.distance_error)
        speed = max(0, 1 - result.runtime_seconds / max_t)
        mem = max(0, 1 - result.peak_memory_mb / max_m)
        stability = 1.0  # 成功则满分

        values = [acc, speed, mem, stability]
        values += values[:1]

        ax.fill(angles, values, color=colors[idx], alpha=0.15)
        ax.plot(angles, values, "o-", color=colors[idx], linewidth=2,
                label=result.algorithm_name, markersize=5)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, color=COLORS["text"])
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"],
                       fontsize=8, color=COLORS["text_dim"])
    ax.set_title(f"[Score]{report.dataset_name} — 多维评测雷达图",
                 fontsize=14, fontweight="bold", pad=25, color=COLORS["text"])
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1),
              fontsize=9, facecolor=COLORS["bg_dark"],
              edgecolor=COLORS["text_dim"])

    plt.tight_layout()

    if save_path:
        save_path = _save_figure(fig, save_path)
    plt.close(fig)

    return save_path or ""


def plot_failure_boundary(
    reports: list[BenchmarkReport],
    save_path: Optional[str] = None,
) -> str:
    """
    失效边界图：展示各算法在不同维度与稀疏度下的失效边界。
    根据项目书设计 —— 通过控制变量法绘制"算法失效边界图"。
    """
    setup_dark_style()

    # 提取所有数据集的维度、稀疏度信息
    dims = []
    sparsities = []
    algo_performance = {}  # algo_name -> [(dim, sparsity, score)]

    for report in reports:
        # 从 description 解析维度信息
        desc = report.dataset_info
        parts = desc.split(",")
        dim_part = [p for p in parts if "dim=" in p]
        sp_part = [p for p in parts if "sparsity=" in p]

        if dim_part and sp_part:
            d = float(dim_part[0].split("=")[1].split("/")[0])
            s = float(sp_part[0].split("=")[1])
            dims.append(d)
            sparsities.append(s)

            for result in report.results:
                if result.success and not getattr(result, "is_reference", False):
                    score = result.distance_error
                    if result.algorithm_name not in algo_performance:
                        algo_performance[result.algorithm_name] = []
                    algo_performance[result.algorithm_name].append((d, s, score))

    if not dims:
        return ""

    fig, axes = plt.subplots(
        1, min(3, len(algo_performance)),
        figsize=(6 * min(3, len(algo_performance)), 5)
    )
    if len(algo_performance) == 1:
        axes = [axes]

    fig.patch.set_facecolor(COLORS["bg_dark"])

    for ax, (algo_name, points) in zip(axes, list(algo_performance.items())[:3]):
        pts = np.array(points)
        scatter = ax.scatter(
            pts[:, 0], pts[:, 1],
            c=pts[:, 2], cmap="RdYlGn_r",
            s=200, edgecolors="white", linewidth=1, alpha=0.85
        )
        ax.set_xlabel("维度", fontsize=10)
        ax.set_ylabel("稀疏度", fontsize=10)
        ax.set_title(f"{algo_name}", fontsize=11, fontweight="bold")
        ax.grid(alpha=0.3)

        # 标注每个点
        for pt in pts:
            ax.annotate(f"err={pt[2]:.3f}",
                        (pt[0], pt[1]),
                        textcoords="offset points",
                        xytext=(0, 12),
                        ha="center", fontsize=7,
                        color=COLORS["text_dim"])

        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label("相对误差", fontsize=9, color=COLORS["text"])
        cbar.ax.yaxis.set_tick_params(color=COLORS["text_dim"])
        cbar.outline.set_edgecolor(COLORS["text_dim"])

    fig.suptitle("📈 算法失效边界图 (维度 × 稀疏度 → 误差)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        save_path = _save_figure(fig, save_path)
    plt.close(fig)

    return save_path or ""


def print_ascii_report(reports: list[BenchmarkReport]):
    """
    在终端打印 ASCII 表格形式的评测报告。
    模拟 PPT 中暗色终端界面的效果。
    """
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "    📋 最优传输算法自动评测报告".center(62) + "█")
    print("█" + "    面向高维稀疏数据的数据集构建与算法评测研究".center(62) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)

    for report in reports:
        print(f"\n{'='*80}")
        print(f"  [Summary]数据集: {report.dataset_name}")
        print(f"  📝 {report.dataset_info}")
        print(f"  [Score]Ground Truth: {report.ground_truth:.6f}")
        print(f"{'='*80}")

        # 表头
        header = (
            f"  {'算法':<22s} {'距离':>10s} {'时间(s)':>10s} "
            f"{'内存(MB)':>10s} {'误差%':>8s} {'评分':>8s} {'状态':>6s}"
        )
        print(header)
        print("  " + "-" * 76)

        # 数据行
        for result in report.results:
            from benchmark_engine import compute_weighted_score
            score = compute_weighted_score(result)

            if getattr(result, "is_reference", False):
                status = "REF"
            elif getattr(result, "not_applicable", False):
                status = "N/A"
            else:
                status = "✓ 成功" if result.success else "✗ 失败"
            dist_str = f"{result.wasserstein_distance:.6f}" if not np.isnan(result.wasserstein_distance) else "N/A"
            time_str = f"{result.runtime_seconds:.4f}" if not np.isnan(result.runtime_seconds) else "N/A"
            mem_str = f"{result.peak_memory_mb:.1f}" if not np.isnan(result.peak_memory_mb) else "N/A"
            err_str = f"{result.distance_error:.2%}" if not np.isnan(result.distance_error) else "N/A"

            # 根据评分着色
            if score >= 0.7:
                score_str = f"⭐{score:.3f}"
            elif score >= 0.4:
                score_str = f" {score:.3f}"
            else:
                score_str = f" {score:.3f}"

            row = (
                f"  {result.algorithm_name:<22s} {dist_str:>10s} {time_str:>10s} "
                f"{mem_str:>10s} {err_str:>8s} {score_str:>8s} {status:>6s}"
            )
            print(row)

        # 排名
        print(f"\n  {'─'*76}")
        print(f"  🏆 综合排名:")
        for rank, (name, score) in enumerate(report.overall_ranking, 1):
            medals = {1: "🥇", 2: "🥈", 3: "🥉"}
            prefix = medals.get(rank, f"  {rank}.")
            print(f"    {prefix} {name:<30s} {score:.4f}")

    print(f"\n{'='*80}")
    print("  ✅ 报告生成完毕。")
    print(f"{'='*80}\n")


def generate_output_dir(base_dir: str = ".") -> str:
    """创建输出目录"""
    output_dir = os.path.join(base_dir, "benchmark_results")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def generate_full_report(
    reports: list[BenchmarkReport],
    output_dir: str = "benchmark_results",
) -> dict[str, str]:
    """
    生成完整的可视化报告，保存所有图表。

    返回: 生成的文件路径字典
    """
    os.makedirs(output_dir, exist_ok=True)
    files = {}

    # 为每个数据集生成对比图
    for report in reports:
        safe_name = report.dataset_name.replace(" ", "_").replace("-", "_")
        comparison_path = os.path.join(output_dir, f"{safe_name}_comparison.png")
        radar_path = os.path.join(output_dir, f"{safe_name}_radar.png")

        plot_algorithm_comparison(report, comparison_path)
        plot_radar_score(report, radar_path)

        files[f"{safe_name}_comparison"] = comparison_path
        files[f"{safe_name}_radar"] = radar_path

    # 生成失效边界图
    boundary_path = os.path.join(output_dir, "failure_boundary.png")
    plot_failure_boundary(reports, boundary_path)
    files["failure_boundary"] = boundary_path

    return files


# ====================================================================
#  新增图表: 失效边界热力图
# ====================================================================

def plot_failure_heatmap(
    sweep_results: dict[str, SweepResult],
    save_path: Optional[str] = None,
) -> str:
    """
    失效边界热力图: 以 dim × sparsity 为网格，颜色表示误差/成功率。

    参数:
        sweep_results: sweep_failure_boundary() 的返回值
        save_path: 保存路径
    """
    setup_dark_style()

    n_algos = len(sweep_results)
    n_cols = min(3, n_algos)
    n_rows = (n_algos + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    fig.patch.set_facecolor(COLORS["bg_dark"])

    if n_rows * n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, (algo_name, sweep) in enumerate(sweep_results.items()):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row][col]

        points = sweep.points
        if not points:
            continue

        dims = sorted(set(p.dim for p in points))
        sparsities = sorted(set(p.sparsity for p in points))

        # 构建误差矩阵
        error_matrix = np.zeros((len(sparsities), len(dims)))
        for p in points:
            i = sparsities.index(p.sparsity)
            j = dims.index(p.dim)
            error_matrix[i, j] = p.error if p.success else np.nan

        im = ax.imshow(error_matrix, aspect="auto", origin="lower",
                       cmap="RdYlGn_r", interpolation="nearest",
                       vmin=0, vmax=max(2.0, np.nanmax(error_matrix)))

        ax.set_xticks(range(len(dims)))
        ax.set_xticklabels(dims, rotation=45, fontsize=7)
        ax.set_yticks(range(len(sparsities)))
        ax.set_yticklabels([f"{s:.0%}" for s in sparsities], fontsize=7)
        ax.set_xlabel("维度", fontsize=9)
        ax.set_ylabel("稀疏度", fontsize=9)
        ax.set_title(algo_name, fontsize=11, fontweight="bold")

        for i in range(len(sparsities)):
            for j in range(len(dims)):
                if not np.isnan(error_matrix[i, j]):
                    color = "white" if error_matrix[i, j] > 1.0 else "black"
                    ax.text(j, i, f"{error_matrix[i,j]:.2f}",
                            ha="center", va="center", fontsize=6, color=color)

        plt.colorbar(im, ax=ax, shrink=0.8, label="相对误差")

    # 隐藏多余的子图
    for idx in range(n_algos, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row][col].set_visible(False)

    fig.suptitle("[Phase Diagram]失效边界热力图 (维度 × 稀疏度 → 相对误差)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_path:
        save_path = _save_figure(fig, save_path)
    plt.close(fig)

    return save_path or ""


# ====================================================================
#  新增图表: 参数敏感性曲线
# ====================================================================

def plot_sensitivity_curves(
    sensitivity_results: list[SensitivityPoint],
    save_path: Optional[str] = None,
) -> str:
    """
    绘制参数敏感性曲线: 双 Y 轴 (精度-误差 / 速度-时间)。
    """
    setup_dark_style()

    if not sensitivity_results:
        return ""

    param_vals = [p.param_value for p in sensitivity_results]
    errors = [p.error * 100 for p in sensitivity_results]
    runtimes = [p.runtime * 1000 for p in sensitivity_results]  # ms
    scores = [p.score for p in sensitivity_results]
    param_name = sensitivity_results[0].param_name

    fig, ax1 = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(COLORS["bg_dark"])
    ax1.set_facecolor(COLORS["bg_card"])

    # 左轴: 误差
    ax1.set_xlabel(f"[Param]{param_name}", fontsize=11)
    ax1.set_ylabel("[Error]相对误差 (%)", fontsize=11, color=COLORS["error"])
    line1, = ax1.plot(param_vals, errors, "o-", color=COLORS["error"],
                       linewidth=2, markersize=8, label="相对误差 (%)")
    ax1.tick_params(axis="y", labelcolor=COLORS["error"])
    ax1.set_xscale("log" if param_name == "reg" else "linear")
    ax1.grid(alpha=0.3)

    # 标注最佳参数
    best_idx = np.argmin(errors)
    ax1.axvline(param_vals[best_idx], color=COLORS["gold"], linestyle="--", alpha=0.5)
    ax1.annotate(f"最优: {param_vals[best_idx]}",
                 xy=(param_vals[best_idx], errors[best_idx]),
                 xytext=(10, 15), textcoords="offset points",
                 fontsize=9, color=COLORS["gold"],
                 arrowprops=dict(arrowstyle="->", color=COLORS["gold"]))

    # 右轴: 运行时间
    ax2 = ax1.twinx()
    ax2.set_ylabel("[Time]运行时间 (ms)", fontsize=11, color=COLORS["primary"])
    line2, = ax2.plot(param_vals, runtimes, "s--", color=COLORS["primary"],
                       linewidth=2, markersize=8, label="运行时间 (ms)")
    ax2.tick_params(axis="y", labelcolor=COLORS["primary"])

    # 综合图例
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left",
               facecolor=COLORS["bg_dark"], edgecolor=COLORS["text_dim"])

    plt.title(f"[Sensitivity]{param_name} 参数敏感性分析 (误差 vs 时间)",
              fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()

    if save_path:
        save_path = _save_figure(fig, save_path)
    plt.close(fig)

    return save_path or ""


# ====================================================================
#  新增图表: 收敛曲线
# ====================================================================

def plot_convergence_profile(
    profile: ConvergenceProfile,
    save_path: Optional[str] = None,
) -> str:
    """
    绘制 Sinkhorn 算法的收敛曲线。
    """
    setup_dark_style()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor(COLORS["bg_dark"])

    # 子图 1: 边际误差 vs 迭代次数
    ax = axes[0]
    ax.semilogy(profile.iterations, profile.marginal_errors,
                "o-", color=COLORS["accent"], linewidth=2, markersize=4)
    ax.set_xlabel("迭代次数", fontsize=10)
    ax.set_ylabel("边际残差 (log)", fontsize=10)
    ax.set_title("[Convergence]边际残差收敛", fontsize=12, fontweight="bold")
    ax.axhline(1e-9, color=COLORS["gold"], linestyle="--", alpha=0.5, label="收敛阈值")
    ax.legend(fontsize=8, facecolor=COLORS["bg_dark"], edgecolor=COLORS["text_dim"])
    ax.grid(alpha=0.3)

    # 子图 2: 距离估计 vs 迭代次数
    ax = axes[1]
    ax.plot(profile.iterations, profile.distances,
            "o-", color=COLORS["success"], linewidth=2, markersize=4)
    ax.set_xlabel("迭代次数", fontsize=10)
    ax.set_ylabel("Wasserstein 距离", fontsize=10)
    ax.set_title("[Convergence]距离估计收敛", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)

    conv_text = f"reg={profile.param_value}, "
    conv_text += f"收敛={'✓' if profile.converged else '✗ (未收敛)'}, "
    conv_text += f"最终迭代={profile.final_iterations}"
    fig.suptitle(f"[Convergence]Sinkhorn 收敛过程 — {conv_text}",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        save_path = _save_figure(fig, save_path)
    plt.close(fig)

    return save_path or ""


# ====================================================================
#  新增图表: 算法推荐矩阵
# ====================================================================

def plot_recommendation_matrix(
    reports: list[BenchmarkReport],
    save_path: Optional[str] = None,
) -> str:
    """
    数据集 × 算法评分热力图 —— 一眼看懂哪个算法最适合哪种数据。
    """
    setup_dark_style()

    if not reports:
        return ""

    datasets = [r.dataset_name for r in reports]
    reference_algorithms = {
        res.algorithm_name
        for r in reports
        for res in r.results
        if getattr(res, "is_reference", False)
    }
    # 收集所有算法名
    algo_names = []
    for r in reports:
        for res in r.results:
            if getattr(res, "is_reference", False) or res.algorithm_name in reference_algorithms:
                continue
            if res.algorithm_name not in algo_names:
                algo_names.append(res.algorithm_name)

    # 构建评分矩阵
    score_matrix = np.zeros((len(datasets), len(algo_names)))
    for i, report in enumerate(reports):
        for j, algo_name in enumerate(algo_names):
            for res in report.results:
                if getattr(res, "is_reference", False):
                    continue
                if res.algorithm_name == algo_name:
                    from benchmark_engine import compute_weighted_score
                    score_matrix[i, j] = compute_weighted_score(res)
                    break

    fig, ax = plt.subplots(figsize=(max(10, len(algo_names) * 1.2),
                                   max(5, len(datasets) * 0.8)))
    fig.patch.set_facecolor(COLORS["bg_dark"])
    ax.set_facecolor(COLORS["bg_card"])

    im = ax.imshow(score_matrix, aspect="auto", cmap="YlOrRd",
                   vmin=0, vmax=1.0)

    ax.set_xticks(range(len(algo_names)))
    ax.set_xticklabels(algo_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets, fontsize=9)
    ax.set_xlabel("算法", fontsize=11)
    ax.set_ylabel("数据集", fontsize=11)

    # 标注数值
    for i in range(len(datasets)):
        for j in range(len(algo_names)):
            score = score_matrix[i, j]
            color = "white" if score < 0.6 else "black"
            ax.text(j, i, f"{score:.3f}", ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, shrink=0.8, label="综合评分")
    ax.set_title("[Recommendation]数据集 × 算法 评分矩阵",
                 fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()

    if save_path:
        save_path = _save_figure(fig, save_path)
    plt.close(fig)

    return save_path or ""


# ====================================================================
#  新增图表: 数据集特征雷达图
# ====================================================================

def plot_dataset_features_radar(
    datasets: list[SparseDataset],
    save_path: Optional[str] = None,
) -> str:
    """
    数据集多维特征对比雷达图。
    """
    from data_generator import diagnose_dataset

    if len(datasets) < 2:
        return ""

    setup_dark_style()

    # 收集诊断信息
    diagnoses = [diagnose_dataset(ds) for ds in datasets]

    # 雷达图维度
    categories = ["维度(log)", "稀疏度", "分布熵(log)", "KL散度", "基尼系数", "代价条件数(log)"]
    N = len(categories)

    # 归一化
    def safe_log(x): return np.log10(max(x, 1))

    values_list = []
    for info, ds in zip(diagnoses, datasets):
        vals = [
            safe_log(info["n_samples"]) / safe_log(5000),      # 归一化维度
            info["effective_sparsity"],                         # 稀疏度 (0~1)
            min(info["entropy_source"] / 15.0, 1.0),           # 熵 (截断到 ~15 bits)
            min(info["kl_divergence"] / 10.0, 1.0),             # KL散度
            info["gini_source"],                                # 基尼系数
            min(safe_log(max(info["cost_matrix_condition"], 1)) / 5.0, 1.0),  # 条件数
        ]
        values_list.append(vals)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor(COLORS["bg_dark"])
    ax.set_facecolor(COLORS["bg_card"])

    colors = _cycle_colors(len(datasets))

    for idx, (ds, values) in enumerate(zip(datasets, values_list)):
        vals = values + values[:1]
        ax.fill(angles, vals, color=colors[idx], alpha=0.15)
        ax.plot(angles, vals, "o-", color=colors[idx], linewidth=2,
                label=ds.name, markersize=5)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9, color=COLORS["text"])
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"],
                       fontsize=7, color=COLORS["text_dim"])
    ax.set_title("[Dataset Portrait]数据集多维特征画像",
                 fontsize=14, fontweight="bold", pad=25, color=COLORS["text"])
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1),
              fontsize=8, facecolor=COLORS["bg_dark"],
              edgecolor=COLORS["text_dim"])

    plt.tight_layout()

    if save_path:
        save_path = _save_figure(fig, save_path)
    plt.close(fig)

    return save_path or ""


# ====================================================================
#  新增图表: 统计结果对比 (带误差棒)
# ====================================================================

def plot_stats_comparison(
    stats_list: list[StatsResult],
    dataset_name: str,
    save_path: Optional[str] = None,
) -> str:
    """
    带误差棒的统计对比图。
    """
    setup_dark_style()

    valid = [s for s in stats_list if s.success_rate > 0]
    if not valid:
        return ""

    names = [s.algorithm_name for s in valid]
    means = [s.mean_distance for s in valid]
    stds = [s.std_distance for s in valid]
    ci95s = [s.ci95_distance for s in valid]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(COLORS["bg_dark"])
    ax.set_facecolor(COLORS["bg_card"])

    x = np.arange(len(names))
    colors = _cycle_colors(len(names))

    bars = ax.bar(x, means, 0.5, color=colors, edgecolor="white", linewidth=0.5, alpha=0.85)
    ax.errorbar(x, means, yerr=ci95s, fmt="none", ecolor="white",
                capsize=5, capthick=1.5, linewidth=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Wasserstein 距离", fontsize=10)
    ax.set_title(f"[Stats]{dataset_name} — 统计对比 (均值 ± 95%CI, n={valid[0].n_runs})",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # 标注值
    for bar, mean, ci in zip(bars, means, ci95s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(means) * 0.01,
                f"{mean:.4f}±{ci:.4f}", ha="center", va="bottom",
                fontsize=7, color=COLORS["text"])

    plt.tight_layout()

    if save_path:
        save_path = _save_figure(fig, save_path)
    plt.close(fig)

    return save_path or ""


# ====================================================================
#  HTML 报告生成
# ====================================================================

def generate_html_report(
    reports: list[BenchmarkReport],
    output_dir: str = "benchmark_results",
    stats_list: Optional[list[list[StatsResult]]] = None,
    sweep_results: Optional[dict[str, SweepResult]] = None,
    sensitivity_results: Optional[list[SensitivityPoint]] = None,
    convergence_profile: Optional[ConvergenceProfile] = None,
    all_datasets: Optional[list[SparseDataset]] = None,
) -> str:
    """
    生成完整的 HTML 评测报告。

    包含:
      - 评测概览表
      - 各数据集详细结果
      - 统计显著性 (如有)
      - 失效边界 (如有)
      - 参数敏感性 (如有)
    """
    os.makedirs(output_dir, exist_ok=True)
    html_path = os.path.join(output_dir, "benchmark_report.html")

    now = __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OT 算法评测报告</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif; background: #0d1117; color: #c9d1d9; padding: 20px; }}
  h1 {{ color: #58a6ff; margin-bottom: 10px; }}
  h2 {{ color: #f0883e; margin: 30px 0 15px; border-bottom: 1px solid #30363d; padding-bottom: 8px; }}
  h3 {{ color: #d2a8ff; margin: 15px 0 10px; }}
  .subtitle {{ color: #8b949e; font-size: 14px; margin-bottom: 20px; }}
  table {{ width: 100%%; border-collapse: collapse; margin: 10px 0 20px; font-size: 13px; }}
  th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #30363d; }}
  th {{ background: #161b22; color: #f0f6fc; font-weight: 600; position: sticky; top: 0; }}
  tr:hover {{ background: #1c2129; }}
  .success {{ color: #56d364; }}
  .fail {{ color: #f85149; }}
  .gold {{ color: #f5c518; font-weight: bold; }}
  .score-bar {{ display: inline-block; height: 12px; border-radius: 3px; margin-right: 5px; }}
  .container {{ max-width: 1400px; margin: 0 auto; }}
  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; margin: 15px 0; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin: 15px 0; }}
  .metric {{ background: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 12px; text-align: center; }}
  .metric .value {{ font-size: 24px; font-weight: bold; color: #58a6ff; }}
  .metric .label {{ font-size: 11px; color: #8b949e; }}
  .footer {{ margin-top: 40px; padding: 15px 0; border-top: 1px solid #30363d; color: #8b949e; font-size: 12px; text-align: center; }}
</style>
</head>
<body>
<div class="container">
<h1> 最优传输 (OT) 算法自动评测报告</h1>
<p class="subtitle">面向高维稀疏数据的数据集构建与算法评测研究 | 生成时间: {now}</p>
"""

    # 概览卡片
    html += '<div class="card"><h3> 评测概览</h3><div class="grid">'
    html += f'<div class="metric"><div class="value">{len(reports)}</div><div class="label">数据集</div></div>'
    n_algos = len(reports[0].results) if reports else 0
    html += f'<div class="metric"><div class="value">{n_algos}</div><div class="label">算法</div></div>'
    total_tests = sum(len(r.results) for r in reports)
    n_success = sum(1 for r in reports for res in r.results if res.success)
    html += f'<div class="metric"><div class="value">{n_success}/{total_tests}</div><div class="label">成功/总数</div></div>'
    if stats_list:
        html += f'<div class="metric"><div class="value">{stats_list[0][0].n_runs if stats_list and stats_list[0] else 0}</div><div class="label">重复次数</div></div>'
    html += '</div></div>'

    # 各数据集结果表
    for report in reports:
        html += f'<h2> 数据集: {report.dataset_name}</h2>'
        html += f'<p style="color:#8b949e;">{report.dataset_info} | Ground Truth: {report.ground_truth:.6f}'
        if getattr(report, "ground_truth_method", ""):
            html += f' | 真值来源: {report.ground_truth_method}'
        if getattr(report, "data_origin", ""):
            html += f' | 数据来源: {report.data_origin}'
        if getattr(report, "geometry_description", ""):
            html += f' | 代价几何: {report.geometry_description}'
        if getattr(report, "source_url", ""):
            html += f' | <a href="{report.source_url}" style="color:#58a6ff;">原始数据 URL</a>'
        if getattr(report, "reference_algorithm", "") and not getattr(report, "ground_truth_is_exact", False):
            html += f' | 参考算法不参与排名: {report.reference_algorithm}'
        if np.isfinite(getattr(report, "scoring_reference", float("nan"))):
            html += (
                f' | 评分裁判: {report.scoring_reference:.6f}'
                f' ({report.scoring_reference_method})'
            )
        html += '</p>'

        html += '<table><thead><tr>'
        html += '<th>算法</th><th>距离</th><th>时间(s)</th><th>内存(MB)</th><th>相对裁判误差%</th><th>评分</th><th>状态</th>'
        html += '</tr></thead><tbody>'

        for result in report.results:
            score = report.weighted_score.get(result.algorithm_name)

            if getattr(result, "is_reference", False):
                status_cls = "gold"
                status_text = "REF"
            elif getattr(result, "ranking_group", "balanced") != "balanced":
                status_cls = "gold"
                status_text = ranking_group_status(getattr(result, "ranking_group", "balanced"))[0]
            elif getattr(result, "not_applicable", False):
                status_cls = "gold"
                status_text = "N/A"
            else:
                status_cls = "success" if result.success else "fail"
                status_text = "✓" if result.success else "✗"

            dist = f"{result.wasserstein_distance:.6f}" if not np.isnan(result.wasserstein_distance) else "N/A"
            t = f"{result.runtime_seconds:.4f}" if not np.isnan(result.runtime_seconds) else "N/A"
            mem = f"{result.peak_memory_mb:.1f}" if not np.isnan(result.peak_memory_mb) else "N/A"
            err = f"{result.distance_error:.2%}" if not np.isnan(result.distance_error) else "N/A"

            score_value = score if score is not None else 0.0
            score_bar_color = "#56d364" if score_value >= 0.7 else "#f0883e" if score_value >= 0.4 else "#f85149"
            bar_width = int(score_value * 150)
            score_text = f"{score_value:.4f}" if score is not None else "--"

            html += '<tr>'
            html += f'<td>{result.algorithm_name}</td>'
            html += f'<td>{dist}</td><td>{t}</td><td>{mem}</td><td>{err}</td>'
            html += f'<td><span class="score-bar" style="width:{bar_width}px;background:{score_bar_color};"></span>{score_text}</td>'
            html += f'<td class="{status_cls}">{status_text}</td>'
            html += '</tr>'

        html += '</tbody></table>'

        # 排名
        html += '<p><strong> 通用高维稀疏算法排名: </strong>'
        for rank, (name, score) in enumerate(report.overall_ranking, 1):
            medals = {1: "🥇", 2: "🥈", 3: "🥉"}
            prefix = medals.get(rank, f"#{rank}")
            html += f'{prefix} {name} ({score:.4f})  '
        html += '</p><p style="color:#8b949e;">说明：评分不是按 EMD 裁判，而是按通用算法的共识参考值计算；REF 只是诊断展示，MODEL 是强模型假设算法，BASE/AUX 是辅助基线。</p>'

    # 统计结果 (如有)
    if stats_list:
        html += '<h2> 统计显著性检验</h2>'
        for stats in stats_list:
            for s in stats:
                if s.success_rate > 0:
                    html += '<div class="card">'
                    html += f'<strong>{s.algorithm_name}</strong>: '
                    html += f'距离 μ={s.mean_distance:.6f}±{s.std_distance:.6f}, '
                    html += f'CI95=±{s.ci95_distance:.6f}, '
                    html += f'成功率={s.success_rate:.0%}, '
                    html += f'评分={s.mean_score:.4f}'
                    html += '</div>'

    # 失效边界 (如有)
    if sweep_results:
        html += '<h2> 失效边界分析</h2>'
        for algo_name, sweep in sweep_results.items():
            if sweep.failure_threshold:
                ft = sweep.failure_threshold
                html += '<div class="card">'
                html += f'<strong>{algo_name}</strong>: '
                html += f'临界维度≈{ft.get("dim_critical", "N/A")}, '
                html += f'安全稀疏度≈{ft.get("sparsity_safe", "N/A")}'
                html += '</div>'

    # 收敛分析 (如有)
    if convergence_profile:
        html += '<h2> 收敛性分析</h2>'
        html += '<div class="card">'
        html += f'<strong>Sinkhorn (reg={convergence_profile.param_value})</strong>: '
        html += f'收敛={"✓" if convergence_profile.converged else "✗"}, '
        html += f'最终迭代={convergence_profile.final_iterations}'
        html += '</div>'

    html += '<div class="footer">OT Benchmark v2.0 | 华南理工大学 · 计算机科学与工程学院 | 大学生创新创业训练计划项目</div>'
    html += '</div></body></html>'

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n📄 HTML 报告已生成: {html_path}")
    return html_path


if __name__ == "__main__":
    print("可视化模块已就绪。")
    print(f"可用图表类型: 对比图、雷达图、失效边界图、失效热力图、敏感性曲线、收敛曲线、推荐矩阵、特征雷达图、GPU加速对比")


# ====================================================================
#  新增图表: GPU 加速对比
# ====================================================================

def plot_gpu_speedup(
    algo_names: list[str],
    cpu_times: list[float],
    gpu_times: list[float],
    save_path: Optional[str] = None,
) -> str:
    """绘制 GPU vs CPU 加速比对比图。

    Args:
        algo_names: 算法名称列表。
        cpu_times: CPU 运行时间 (秒)。
        gpu_times: GPU 运行时间 (秒)。
        save_path: 保存路径。
    """
    setup_dark_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(COLORS["bg_dark"])

    x = np.arange(len(algo_names))
    width = 0.35

    # 左图: 时间对比
    ax1.bar(x - width/2, cpu_times, width, label="CPU", color=COLORS["primary"], alpha=0.85)
    ax1.bar(x + width/2, gpu_times, width, label="GPU", color=COLORS["accent"], alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(algo_names, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("运行时间 (s)", fontsize=10)
    ax1.set_title("[Speedup]CPU vs GPU 运行时间", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # 右图: 加速比
    speedups = [c / max(g, 1e-6) for c, g in zip(cpu_times, gpu_times)]
    bar_colors = [COLORS["success"] if s >= 1 else COLORS["error"] for s in speedups]
    ax2.bar(x, speedups, 0.5, color=bar_colors, alpha=0.85)
    ax2.axhline(1.0, color="white", linestyle="--", alpha=0.5, linewidth=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(algo_names, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("加速比 (CPU/GPU)", fontsize=10)
    ax2.set_title("[Speedup]加速倍数", fontsize=12, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    for i, s in enumerate(speedups):
        ax2.text(i, s + 0.1, f"{s:.1f}x", ha="center", fontsize=9,
                 color=COLORS["text"], fontweight="bold")

    fig.suptitle(" GPU 加速效果对比", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        save_path = _save_figure(fig, save_path)
    plt.close(fig)

    return save_path or ""


def plot_parallel_speedup(
    n_workers: list[int],
    throughput: list[float],
    ideal_scaling: bool = True,
    save_path: Optional[str] = None,
) -> str:
    """绘制并行加速比曲线。

    Args:
        n_workers: Worker 数量列表。
        throughput: 对应的吞吐量 (tasks/s)。
        ideal_scaling: 是否绘制理想线性加速。
        save_path: 保存路径。
    """
    setup_dark_style()

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(COLORS["bg_dark"])
    ax.set_facecolor(COLORS["bg_card"])

    base = throughput[0] if throughput else 1.0
    speedups = [t / base for t in throughput]

    ax.plot(n_workers, speedups, "o-", color=COLORS["accent"],
            linewidth=2, markersize=10, label="实际加速比")

    if ideal_scaling:
        ax.plot(n_workers, n_workers, "--", color=COLORS["gold"],
                linewidth=1.5, alpha=0.7, label="理想线性加速")

    ax.set_xlabel("Worker 数量", fontsize=11)
    ax.set_ylabel("加速比", fontsize=11)
    ax.set_title("[Parallel]并行加速比分析", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    for w, s in zip(n_workers, speedups):
        ax.annotate(f"{s:.1f}x", (w, s),
                    textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=9, color=COLORS["text"])

    plt.tight_layout()

    if save_path:
        save_path = _save_figure(fig, save_path)
    plt.close(fig)

    return save_path or ""


if __name__ == "__main__":
    print("可视化模块已就绪。")
    print(f"可用图表类型: 对比图、雷达图、失效边界图、失效热力图、敏感性曲线、收敛曲线、推荐矩阵、特征雷达图、GPU加速对比、并行加速比")
