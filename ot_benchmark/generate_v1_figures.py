#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OTBench Benchmark v1.0 — 论文/结项图表生成器.

从 benchmark_v1_results 目录读取数据，生成：
  - runtime_vs_dimension.png
  - error_vs_dimension.png
  - memory_vs_dimension.png
  - success_rate_heatmap.png
  - algorithm_tradeoff_table.csv
  - paper_ready_summary.md

用法:
  python generate_v1_figures.py --input benchmark_v1_results_small_official
  python generate_v1_figures.py --input benchmark_v1_results
"""

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

# matplotlib 中文配置
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# 尝试设置中文字体
_CJK_FONT = None
for _font in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
               "WenQuanYi Micro Hei", "Arial Unicode MS"]:
    try:
        matplotlib.font_manager.findfont(_font, fallback_to_default=False)
        _CJK_FONT = _font
        break
    except Exception:
        continue

if _CJK_FONT:
    plt.rcParams["font.sans-serif"] = [_CJK_FONT, "DejaVu Sans"]
else:
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["figure.dpi"] = 150

# 配色
ALGO_COLORS = [
    "#6366f1", "#22c55e", "#f59e0b", "#ef4444",
    "#06b6d4", "#8b5cf6", "#f97316", "#10b981",
    "#ec4899", "#14b8a6", "#e11d48", "#3b82f6", "#84cc16", "#d946ef",
]
COLOR_MAP = {}
_algo_name_cache: list[str] = []


def _color_for(algo: str) -> str:
    global _algo_name_cache
    if algo not in COLOR_MAP:
        COLOR_MAP[algo] = ALGO_COLORS[len(COLOR_MAP) % len(ALGO_COLORS)]
        _algo_name_cache.append(algo)
    return COLOR_MAP[algo]


def _safe(v):
    if v is None:
        return None
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(v, str):
        s = v.strip()
        if s == "" or s == "N/A":
            return None
        try:
            return float(s)
        except ValueError:
            return None
    try:
        fv = float(v)
        if math.isnan(fv) or math.isinf(fv):
            return None
        return fv
    except (ValueError, TypeError):
        return None


def _fmt(v):
    if v is None:
        return "N/A"
    if isinstance(v, float):
        if abs(v) < 1e-9:
            return "~0"
        if abs(v) < 0.01 or abs(v) >= 10000:
            return f"{v:.4e}"
        return f"{v:.4g}"
    return str(v)


# ============================================================================
#  数据加载
# ============================================================================

def load_data(input_dir: str) -> tuple[dict, dict]:
    """加载 within-seed 和 cross-seed 汇总数据."""
    within_seed = []
    csv_path = os.path.join(input_dir, "summary_results.csv")
    if os.path.isfile(csv_path):
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            # Normalize column names (strip BOM artifacts)
            reader.fieldnames = [
                (fn or "").lstrip("﻿").strip() for fn in (reader.fieldnames or [])
            ]
            within_seed = list(reader)

    cross_seed_path = os.path.join(input_dir, "summary_cross_seed.csv")
    cross_seed = []
    if os.path.isfile(cross_seed_path):
        with open(cross_seed_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            reader.fieldnames = [
                (fn or "").lstrip("﻿").strip() for fn in (reader.fieldnames or [])
            ]
            cross_seed = list(reader)

    # 也读取 raw_results.csv 用于提取更多信息
    raw_csv = os.path.join(input_dir, "raw_results.csv")
    raw_rows = []
    if os.path.isfile(raw_csv):
        with open(raw_csv, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            reader.fieldnames = [
                (fn or "").lstrip("﻿").strip() for fn in (reader.fieldnames or [])
            ]
            raw_rows = list(reader)

    # 构建场景维度映射
    scene_dims: dict[str, int] = {}
    for row in raw_rows:
        sid = (row.get("scenario_id") or "").strip().lstrip("﻿")
        dim = (row.get("dimension") or "0").strip()
        if sid and dim:
            try:
                scene_dims[sid] = int(float(dim))
            except ValueError:
                pass

    return {
        "within_seed": within_seed,
        "cross_seed": cross_seed,
        "raw_rows": raw_rows,
        "scene_dims": scene_dims,
    }


# ============================================================================
#  图表 1: runtime_vs_dimension.png
# ============================================================================

def plot_runtime_vs_dimension(data: dict, output_dir: str) -> str:
    within = data["within_seed"]
    scene_dims = data["scene_dims"]

    # 按 algorithm 聚合: {algo: {dim: [runtime, ...]}}
    algo_dim_times: dict[str, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    for row in within:
        algo = row.get("algorithm", "")
        sid = row.get("scenario_id", "")
        dim = scene_dims.get(sid, 0)
        if dim <= 0:
            continue
        rt = _safe(row.get("mean_runtime_seconds"))
        if rt is not None:
            algo_dim_times[algo][dim].append(rt)

    fig, ax = plt.subplots(figsize=(10, 6))
    for algo in sorted(algo_dim_times.keys()):
        dims = sorted(algo_dim_times[algo].keys())
        means = [np.mean(algo_dim_times[algo][d]) for d in dims]
        ax.plot(dims, means, "o-", color=_color_for(algo),
                label=algo, linewidth=1.5, markersize=6)

    ax.set_xlabel("Dimension" if not _CJK_FONT else "维度", fontsize=12)
    ax.set_ylabel("Mean Runtime (s)" if not _CJK_FONT else "平均运行时间 (s)",
                  fontsize=12)
    ax.set_title("Runtime vs Dimension" if not _CJK_FONT
                 else "运行时间—维度关系", fontsize=14, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, "runtime_vs_dimension.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ============================================================================
#  图表 2: error_vs_dimension.png
# ============================================================================

def plot_error_vs_dimension(data: dict, output_dir: str) -> str:
    within = data["within_seed"]
    scene_dims = data["scene_dims"]

    algo_dim_errs: dict[str, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    for row in within:
        algo = row.get("algorithm", "")
        sid = row.get("scenario_id", "")
        dim = scene_dims.get(sid, 0)
        if dim <= 0:
            continue
        # 优先 ground_truth_error, 其次 reference_error
        err = _safe(row.get("mean_ground_truth_error") or
                    row.get("mean_reference_error"))
        if err is not None:
            algo_dim_errs[algo][dim].append(err)

    fig, ax = plt.subplots(figsize=(10, 6))
    for algo in sorted(algo_dim_errs.keys()):
        dims = sorted(algo_dim_errs[algo].keys())
        means = [np.mean(algo_dim_errs[algo][d]) for d in dims]
        ax.plot(dims, means, "s-", color=_color_for(algo),
                label=algo, linewidth=1.5, markersize=6)

    ax.set_xlabel("Dimension" if not _CJK_FONT else "维度", fontsize=12)
    ylabel = ("Relative Error (ground_truth or reference)"
              if not _CJK_FONT
              else "相对误差 (exact 或 reference)")
    ax.set_ylabel(ylabel, fontsize=12)
    title = ("Error vs Dimension\n(exact where available, otherwise reference)"
             if not _CJK_FONT
             else "误差—维度关系\n(exact 优先, 无精确解时使用 reference)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, "error_vs_dimension.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ============================================================================
#  图表 3: memory_vs_dimension.png
# ============================================================================

def plot_memory_vs_dimension(data: dict, output_dir: str) -> str:
    within = data["within_seed"]
    scene_dims = data["scene_dims"]

    algo_dim_mem: dict[str, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    for row in within:
        algo = row.get("algorithm", "")
        sid = row.get("scenario_id", "")
        dim = scene_dims.get(sid, 0)
        if dim <= 0:
            continue
        mem = _safe(row.get("mean_peak_memory_mb"))
        if mem is not None:
            algo_dim_mem[algo][dim].append(mem)

    fig, ax = plt.subplots(figsize=(10, 6))
    for algo in sorted(algo_dim_mem.keys()):
        dims = sorted(algo_dim_mem[algo].keys())
        means = [np.mean(algo_dim_mem[algo][d]) for d in dims]
        ax.plot(dims, means, "D-", color=_color_for(algo),
                label=algo, linewidth=1.5, markersize=6)

    ax.set_xlabel("Dimension" if not _CJK_FONT else "维度", fontsize=12)
    ax.set_ylabel("Peak Memory (MB)" if not _CJK_FONT else "峰值内存 (MB)",
                  fontsize=12)
    ax.set_title("Memory vs Dimension" if not _CJK_FONT
                 else "峰值内存—维度关系", fontsize=14, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, "memory_vs_dimension.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ============================================================================
#  图表 4: success_rate_heatmap.png
# ============================================================================

def plot_success_rate_heatmap(data: dict, output_dir: str) -> str:
    cross = data["cross_seed"]
    if not cross:
        return ""

    # 构建矩阵: row=scenario_id, col=algorithm
    scenarios = sorted(set(r["scenario_id"] for r in cross))
    algos = sorted(set(r["algorithm"] for r in cross))

    matrix = np.zeros((len(scenarios), len(algos)))
    for i, sid in enumerate(scenarios):
        for j, algo in enumerate(algos):
            for row in cross:
                if row["scenario_id"] == sid and row["algorithm"] == algo:
                    sr = _safe(row.get("success_rate"))
                    matrix[i, j] = sr if sr is not None else 0.0
                    break

    fig, ax = plt.subplots(figsize=(max(10, len(algos) * 1.2),
                                     max(5, len(scenarios) * 0.6)))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(algos)))
    ax.set_xticklabels(algos, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels(scenarios, fontsize=8)

    for i in range(len(scenarios)):
        for j in range(len(algos)):
            ax.text(j, i, f"{matrix[i, j]:.0%}", ha="center", va="center",
                    fontsize=7, fontweight="bold",
                    color="white" if matrix[i, j] < 0.5 else "black")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Success Rate" if not _CJK_FONT else "成功率", fontsize=10)
    ax.set_title("Success Rate Heatmap (Cross-Seed)" if not _CJK_FONT
                 else "成功率热力图 (跨 Seed)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(output_dir, "success_rate_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ============================================================================
#  CSV 表格: algorithm_tradeoff_table.csv
# ============================================================================

def generate_tradeoff_table(data: dict, output_dir: str) -> str:
    cross = data["cross_seed"]
    if not cross:
        return ""

    # 按算法聚合
    algo_stats: dict[str, dict] = defaultdict(lambda: {
        "errors": [], "times": [], "mems": [],
        "succ_runs": 0, "total_runs": 0,
        "scenarios": set(),
    })

    for row in cross:
        algo = row.get("algorithm", "")
        a = algo_stats[algo]
        a["scenarios"].add(row.get("scenario_id", ""))
        a["succ_runs"] += int(row.get("successful_runs", 0))
        a["total_runs"] += int(row.get("total_runs", 0))

        err = _safe(row.get("mean_ground_truth_error"))
        if err is not None:
            a["errors"].append(err)

        t = _safe(row.get("mean_runtime_seconds_cross"))
        if t is not None:
            a["times"].append(t)

        m = _safe(row.get("mean_peak_memory_mb_cross"))
        if m is not None:
            a["mems"].append(m)

    rows = []
    for algo, a in sorted(algo_stats.items()):
        total = a["total_runs"]
        succ = a["succ_runs"]
        rows.append({
            "algorithm": algo,
            "mean_error": np.mean(a["errors"]) if a["errors"]
                          else None,
            "mean_runtime_s": np.mean(a["times"]) if a["times"]
                               else None,
            "mean_memory_mb": np.mean(a["mems"]) if a["mems"]
                              else None,
            "success_rate": succ / total if total > 0 else 0.0,
            "n_scenarios": len(a["scenarios"]),
            "total_runs": total,
            "successful_runs": succ,
        })

    path = os.path.join(output_dir, "algorithm_tradeoff_table.csv")
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algorithm", "mean_error", "mean_runtime_s",
                     "mean_memory_mb", "success_rate",
                     "n_scenarios", "total_runs", "successful_runs"])
        for r in rows:
            w.writerow([r["algorithm"],
                        _fmt(r["mean_error"]),
                        _fmt(r["mean_runtime_s"]),
                        _fmt(r["mean_memory_mb"]),
                        f"{r['success_rate']:.4f}",
                        r["n_scenarios"],
                        r["total_runs"],
                        r["successful_runs"]])
    return path


# ============================================================================
#  MD 摘要: paper_ready_summary.md
# ============================================================================

def generate_paper_summary(data: dict, output_dir: str,
                             input_dir: str) -> str:
    cross = data["cross_seed"]
    within = data["within_seed"]
    scene_dims = data["scene_dims"]

    # 统计
    n_scenarios = len(set(r["scenario_id"] for r in cross))
    n_algorithms = len(set(r["algorithm"] for r in cross))
    total_planned = sum(int(r.get("total_runs", 0)) for r in cross)
    total_success = sum(int(r.get("successful_runs", 0)) for r in cross)
    overall_sr = total_success / max(total_planned, 1)

    # 场景表
    scene_table = ""
    for sid in sorted(scene_dims.keys()):
        dim = scene_dims[sid]
        scene_table += f"| {sid} | {dim} |\n"

    # 算法表
    algo_table = ""
    for row in sorted(cross, key=lambda r: _safe(r.get("mean_runtime_seconds_cross")) or 999):
        algo = row["algorithm"]
        sr = _safe(row.get("success_rate"))
        t = _safe(row.get("mean_runtime_seconds_cross"))
        m = _safe(row.get("mean_peak_memory_mb_cross"))
        e = _safe(row.get("mean_ground_truth_error"))
        if sr is not None and t is not None:
            algo_table += (f"| {algo} | {sr:.1%} | {t:.3f}s | "
                           f"{_fmt(m)}MB | {_fmt(e)} |\n")

    # 失败统计
    fail_counts: dict[str, int] = {}
    for row in within:
        fbc_raw = row.get("failure_count_by_type", "")
        if fbc_raw and fbc_raw.strip() and fbc_raw != "None":
            try:
                d = json.loads(fbc_raw.replace("'", '"'))
                for k, v in d.items():
                    fail_counts[k] = fail_counts.get(k, 0) + v
            except Exception:
                pass
    fail_sec = ""
    if fail_counts:
        for k, v in sorted(fail_counts.items(), key=lambda x: -x[1]):
            fail_sec += f"- {k}: {v}\n"

    # 找出最快/最准/最省内存算法
    best_err = min(
        (r for r in cross if _safe(r.get("mean_ground_truth_error"))),
        key=lambda r: _safe(r.get("mean_ground_truth_error")) or 999,
        default={})
    best_time = min(
        (r for r in cross if _safe(r.get("mean_runtime_seconds_cross"))),
        key=lambda r: _safe(r.get("mean_runtime_seconds_cross")) or 999,
        default={})
    best_mem = min(
        (r for r in cross if _safe(r.get("mean_peak_memory_mb_cross"))),
        key=lambda r: _safe(r.get("mean_peak_memory_mb_cross")) or 999,
        default={})

    content = f"""# OTBench Benchmark v1.0 — 论文可用实验摘要

## 实验配置

- **配置来源**: {os.path.basename(input_dir)}
- **数据场景数**: {n_scenarios}
- **评测算法数**: {n_algorithms}
- **每场景 seed 数**: 3
- **每 seed 重复次数**: 2
- **总计划运行数**: {total_planned}
- **总成功运行数**: {total_success}
- **总体成功率**: {overall_sr:.1%}

## 数据集构成

| 场景 ID | 维度 |
|---------|------|
{scene_table}

## 算法列表及其性能 (Cross-Seed)

| 算法 | 成功率 | 平均耗时 | 平均内存 | 平均 GT Error |
|------|--------|----------|----------|---------------|
{algo_table}

> 注: GT Error 仅在有 EMD 精确解时有效，否则为空。reference error 是相对高精度 Sinkhorn 参考解的偏差，**不是精确真值误差**。

## 关键发现

1. **最精确** (GT Error 最小): {best_err.get('algorithm', 'N/A')}
2. **最快**: {best_time.get('algorithm', 'N/A')}
3. **最省内存**: {best_mem.get('algorithm', 'N/A')}

## 失败类型统计

{fail_sec if fail_sec else '无失败记录'}

## 局限性

1. **Ground truth 依赖维度**: 仅 ≤300 维场景有 EMD 精确解，更大维度依赖 Sinkhorn 参考解，不能完全验证精度。
2. **内存测量为进程 RSS**: peak_memory_mb 仅反映 CPU 进程 RSS 增量，不含 GPU 显存，GPU 算法实际内存占用量将此数据低估。
3. **单机实验**: 所有结果在单台 Windows PC (RTX 5070 Ti) 上获得，不具备分布式或云环境代表性。
4. **合成数据为主**: 数据集基于统计分布合成，未包含真实世界数据的领域特定结构。
5. **算法参数固定**: 使用算法注册表中的默认参数，未做超参数调优，部分算法可能未达到最佳性能。

## 可用于论文/结项的图表清单

| 文件 | 内容 |
|------|------|
| `runtime_vs_dimension.png` | 各算法运行时间随维度变化曲线 |
| `error_vs_dimension.png` | 各算法相对误差随维度变化曲线 |
| `memory_vs_dimension.png` | 各算法峰值内存随维度变化曲线 |
| `success_rate_heatmap.png` | 场景×算法成功率热力图 |
| `algorithm_tradeoff_table.csv` | 各算法综合指标表 |
| `failure_boundary_*.png` | 失效边界分面图 (需 --sweep 模式) |

---

*自动生成于 OTBench Benchmark v1.0 · 华南理工大学 计算机科学与工程学院*
"""
    path = os.path.join(output_dir, "paper_ready_summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


# ============================================================================
#  main
# ============================================================================

def main():
    p = argparse.ArgumentParser(
        description="OTBench v1 论文图表生成器")
    p.add_argument("--input", type=str, required=True,
                   help="输入目录 (benchmark_v1_results 或 similar)")
    p.add_argument("--output", type=str, default=None,
                   help="输出 figures 子目录")
    args = p.parse_args()

    input_dir = args.input
    if not os.path.isdir(input_dir):
        print(f"错误: 输入目录不存在: {input_dir}")
        return 1

    figures_dir = args.output or os.path.join(input_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    print(f"加载数据: {input_dir}")
    data = load_data(input_dir)
    print(f"  within_seed: {len(data['within_seed'])} rows")
    print(f"  cross_seed:  {len(data['cross_seed'])} rows")
    print(f"  raw:         {len(data['raw_rows'])} rows")
    print(f"  scenes:      {len(data['scene_dims'])}")

    print("\n生成图表...")

    paths = []

    try:
        p = plot_runtime_vs_dimension(data, figures_dir)
        paths.append(("runtime_vs_dimension.png", p))
        print(f"  OK: {p}")
    except Exception as e:
        print(f"  FAIL: runtime_vs_dimension -> {e}")

    try:
        p = plot_error_vs_dimension(data, figures_dir)
        paths.append(("error_vs_dimension.png", p))
        print(f"  OK: {p}")
    except Exception as e:
        print(f"  FAIL: error_vs_dimension -> {e}")

    try:
        p = plot_memory_vs_dimension(data, figures_dir)
        paths.append(("memory_vs_dimension.png", p))
        print(f"  OK: {p}")
    except Exception as e:
        print(f"  FAIL: memory_vs_dimension -> {e}")

    try:
        p = plot_success_rate_heatmap(data, figures_dir)
        paths.append(("success_rate_heatmap.png", p))
        print(f"  OK: {p}")
    except Exception as e:
        print(f"  FAIL: success_rate_heatmap -> {e}")

    try:
        p = generate_tradeoff_table(data, figures_dir)
        paths.append(("algorithm_tradeoff_table.csv", p))
        print(f"  OK: {p}")
    except Exception as e:
        print(f"  FAIL: tradeoff_table -> {e}")

    try:
        p = generate_paper_summary(data, figures_dir, input_dir)
        paths.append(("paper_ready_summary.md", p))
        print(f"  OK: {p}")
    except Exception as e:
        print(f"  FAIL: paper_summary -> {e}")

    print(f"\n完成! 输出目录: {figures_dir}")
    print(f"生成文件: {len(paths)}")
    for name, fpath in paths:
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
