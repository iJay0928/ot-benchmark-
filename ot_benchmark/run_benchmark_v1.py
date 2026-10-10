#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OTBench Benchmark v1.0 — 一键实验入口（修订版）
================================================
支持 dataset_seed / algorithm_seed / repetition_index 区分、
exact / reference / consensus 三种参考类型、两级统计。

用法:
  python run_benchmark_v1.py --quick      # 快速自检 (1 seed, 2 reps)
  python run_benchmark_v1.py              # 标准模式 (3 seeds, 5 reps)
  python run_benchmark_v1.py --runs 5     # 每 seed 重复 5 次
  python run_benchmark_v1.py --output my_results
  python run_benchmark_v1.py --resume     # 续跑
  python run_benchmark_v1.py --algorithms sinkhorn,lowrank
"""

from __future__ import annotations

import argparse
import codecs
import csv
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

# ---- 确保包在路径中 ----
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_DIR = os.path.join(_PROJECT_ROOT, "ot_benchmark")
if _PACKAGE_DIR not in sys.path:
    sys.path.insert(0, _PACKAGE_DIR)

from data_generator import (
    SparseDataset,
    create_benchmark_dataset,
    create_benchmark_dataset_with_shift,
)
from benchmark_engine import (
    AlgorithmResult,
    run_single_algorithm,
    sweep_failure_boundary,
)
from algorithms.registry import ALGORITHMS
from core.benchmark_v1_config import (
    load_benchmark_v1_manifest,
    get_enabled_scenarios,
    get_quick_scenarios,
    ConfigValidationError,
)

# ---- 常量 ----
STATUS_VALUES = {
    "success", "timeout", "out_of_memory", "numerical_error",
    "dependency_error", "not_applicable", "unknown_error",
}

SCORING_REF_TYPES = {"exact", "reference", "consensus", "unavailable"}

CSV_COLUMNS = [
    "experiment_id", "timestamp",
    "scenario_id", "scenario_name",
    "dataset_seed", "algorithm_seed", "repetition_index",
    "algorithm", "ranking_group",
    "status", "failure_type", "error_message",
    "n_source", "n_target", "dimension", "geometry_dimension",
    "sparsity", "sparsity_semantics", "structure",
    "noise_level", "balanced", "data_origin",
    "ground_truth_mode", "ground_truth_method",
    "ground_truth_distance", "reference_distance",
    "wasserstein_distance",
    "ground_truth_error", "reference_error", "consensus_deviation",
    "scoring_reference_type",
    "runtime_seconds", "peak_memory_mb",
    "convergence_iterations",
]

SUMMARY_COLUMNS = [
    "scenario_id", "algorithm", "ranking_group",
    "dataset_seed",
    "total_runs", "successful_runs", "success_rate",
    "mean_distance", "std_distance",
    "mean_ground_truth_error", "std_ground_truth_error",
    "mean_reference_error", "std_reference_error",
    "mean_consensus_deviation", "std_consensus_deviation",
    "mean_runtime_seconds", "std_runtime_seconds", "p95_runtime_seconds",
    "mean_peak_memory_mb", "std_peak_memory_mb",
    "failure_count_by_type",
]

CROSS_SEED_COLUMNS = [
    "scenario_id", "algorithm", "ranking_group",
    "n_seeds", "total_runs", "successful_runs", "success_rate",
    "mean_distance", "std_distance",
    "mean_ground_truth_error", "std_ground_truth_error",
    "mean_runtime_seconds_cross", "std_runtime_seconds_cross",
    "mean_peak_memory_mb_cross", "std_peak_memory_mb_cross",
    "failure_count_by_type",
]


# ============================================================================
#  CLI
# ============================================================================

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OTBench Benchmark v1.0 — 一键实验入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--quick", action="store_true",
                   help="快速模式: 1 个 seed, 2 次重复, 仅小场景")
    p.add_argument("--runs", type=int, default=5,
                   help="每个 (场景, seed) 组合的重复次数 (默认 5)")
    p.add_argument("--algorithms", type=str, default="all",
                   help="逗号分隔的算法过滤")
    p.add_argument("--output", type=str, default="benchmark_v1_results",
                   help="输出目录 (默认 benchmark_v1_results)")
    p.add_argument("--resume", action="store_true",
                   help="跳过已完成记录, 支持中断续跑")
    p.add_argument("--no-sweep", action="store_true",
                   help="跳过失效边界扫描")
    p.add_argument("--no-report", action="store_true",
                   help="跳过 HTML 报告")
    p.add_argument("--config", type=str, default=None,
                   help="YAML 文件路径 (默认使用项目内置)")
    return p.parse_args(argv)


# ============================================================================
#  数据集生成
# ============================================================================

def generate_dataset_from_scenario(sc: dict, dataset_seed: int) -> SparseDataset:
    """从场景配置 + dataset_seed 生成 SparseDataset。"""
    n = int(sc["n_source"])
    sparsity = float(sc["sparsity"])
    noise = float(sc.get("noise_level", 0.03))
    src_type = str(sc.get("source_type", "zipf"))
    tgt_type = str(sc.get("target_type", src_type))
    structure = str(sc.get("structure", "random"))
    geometry_dim = int(sc.get("geometry_dim", 1))
    gt_mode = str(sc.get("ground_truth_mode", "unknown"))
    shift_type = sc.get("shift_type")
    real_dataset = sc.get("real_dataset")

    # 真实数据场景
    if src_type == "real" and real_dataset:
        from real_data import load_real_dataset
        real_params = dict(sc.get("real_params", {}))
        real_params["random_state"] = dataset_seed
        ds = load_real_dataset(real_dataset, **real_params)
        # 覆盖 YAML 指定的结构化字段
        ds.dataset_id = sc["id"]
        ds.dataset_seed = dataset_seed
        ds.ground_truth_mode = gt_mode
        ds.sparsity_semantics = str(sc.get("sparsity_semantics", "zero_ratio"))
        ds.structure = structure
        ds.balanced = bool(sc.get("balanced", True))
        ds.data_origin = "real"
        return ds

    if shift_type:
        return create_benchmark_dataset_with_shift(
            n_samples=n, sparsity=sparsity, noise_level=noise,
            source_dist_type=src_type,
            shift_type=str(shift_type),
            shift_intensity=float(sc.get("shift_intensity", 0.3)),
            random_state=dataset_seed,
            name=sc["name"],
            geometry_dim=geometry_dim,
        )

    return create_benchmark_dataset(
        n_samples=n, sparsity=sparsity, noise_level=noise,
        source_dist_type=src_type, target_dist_type=tgt_type,
        random_state=dataset_seed,
        name=sc["name"],
        structure=structure,
        sparsity_semantics=str(sc.get("sparsity_semantics", "zero_ratio")),
        balanced=bool(sc.get("balanced", True)),
        ground_truth_mode=gt_mode,
        geometry_dim=geometry_dim,
    )


# ============================================================================
#  错误分类与路径清理
# ============================================================================

def classify_status(error: Optional[BaseException],
                    result: Optional[AlgorithmResult]) -> str:
    if error is None and result is not None:
        if result.success:
            return "success"
        if result.not_applicable:
            return "not_applicable"
        msg = (result.error_message or "").lower()
        if "timeout" in msg or "timed out" in msg:
            return "timeout"
        if "memory" in msg or "out of memory" in msg:
            return "out_of_memory"
        if any(token in msg for token in ("numerical", "singular", "nan", "residual", "transport plan")):
            return "numerical_error"
        if "import" in msg or "dependency" in msg or "module" in msg:
            return "dependency_error"
        return "unknown_error"

    if error is None:
        return "unknown_error"
    exc_name = type(error).__name__
    exc_msg = str(error).lower()
    if "memory" in exc_msg:
        return "out_of_memory"
    if "timeout" in exc_msg:
        return "timeout"
    if exc_name in ("ImportError", "ModuleNotFoundError"):
        return "dependency_error"
    if exc_name in ("LinAlgError", "FloatingPointError"):
        return "numerical_error"
    return "unknown_error"


def _sanitize_path(msg: str) -> str:
    if not msg:
        return ""
    msg = re.sub(r"[A-Za-z]:\\Users\\[^\\\s]+", "<HOME>", str(msg))
    msg = re.sub(r"/home/[^/\s]+", "<HOME>", msg)
    return msg


# ============================================================================
#  NaN/Inf 安全序列化
# ============================================================================

def _safe_val(v: Any) -> Any:
    """返回 JSON 安全的值。NaN/Inf -> None。"""
    if v is None:
        return None
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(v, (np.floating, np.integer)):
        fv = float(v)
        if math.isnan(fv) or math.isinf(fv):
            return None
        return float(fv) if isinstance(v, np.floating) else int(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


class _SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.floating, np.integer)):
            fv = float(obj)
            if math.isnan(fv) or math.isinf(fv):
                return None
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _csv_val(v: Any) -> str:
    """Python 值 -> CSV 安全字符串。"""
    if v is None:
        return ""
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return ""
        return f"{v:.8g}"
    if isinstance(v, (np.floating, np.integer)):
        fv = float(v)
        if math.isnan(fv) or math.isinf(fv):
            return ""
        return f"{fv:.8g}"
    if isinstance(v, dict):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


# ============================================================================
#  环境采集
# ============================================================================

def capture_environment(config_path: str) -> dict:
    info: dict[str, Any] = {}
    info["os"] = platform.system()
    info["os_release"] = platform.release()
    info["os_version"] = platform.version()
    info["python_version"] = sys.version

    libs = {}
    for pkg in ["numpy", "scipy", "ot", "matplotlib", "psutil",
                 "torch", "cupy", "scikit-learn", "pyyaml"]:
        try:
            import importlib.metadata
            libs[pkg] = importlib.metadata.version(pkg)
        except Exception:
            libs[pkg] = None
    info["libraries"] = libs

    try:
        import psutil
        info["cpu_logical_cores"] = psutil.cpu_count(logical=True)
        info["cpu_physical_cores"] = psutil.cpu_count(logical=False)
        info["total_ram_gb"] = round(
            psutil.virtual_memory().total / (1024 ** 3), 2)
    except Exception:
        info["cpu_logical_cores"] = os.cpu_count()
        info["cpu_physical_cores"] = None
        info["total_ram_gb"] = None

    info["gpu_name"] = None
    info["gpu_vram_mb"] = None
    info["cuda_version"] = None
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["gpu_vram_mb"] = props.total_memory // (1024 * 1024)
            info["cuda_version"] = torch.version.cuda
    except Exception:
        pass

    info["git_commit"] = None
    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_PROJECT_ROOT,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        pass

    try:
        with open(config_path or os.path.join(
                _PROJECT_ROOT, "configs", "benchmark_v1.yaml"), "rb") as f:
            info["config_sha256"] = hashlib.sha256(f.read()).hexdigest()
    except Exception:
        info["config_sha256"] = None

    return info


# ============================================================================
#  协议文档
# ============================================================================

def generate_protocol_md(output_dir: str, n_seeds: int, n_reps: int,
                          n_scenarios: int, n_algorithms: int) -> str:
    content = f"""# OTBench Benchmark v1.0 — 实验协议 (修订版)

## 1. 数据产生
根据 `configs/benchmark_v1.yaml` 中的 {n_scenarios} 个场景配置生成数据集。
相同场景、相同 dataset_seed 必定生成完全一致的数据。
- `dimension` 表示分布坐标/质量向量长度；`geometry_dimension` 表示 OT 支撑点的几何维度。
- 合成高维场景必须显式设置 `geometry_dim > 1`；未设置时为兼容旧结果使用 1D index geometry。

## 2. 种子体系
- **dataset_seed**: 决定数据集内容。每场景至少 3 个 seed（quick 模式用 1 个）。
- **algorithm_seed**: 仅供算法内部随机过程使用。确定性算法为 ""。
- **repetition_index**: 同一 (场景, seed, 算法) 组合的第几次重复运行。

## 3. 实验结构
场景定义
  ├─ dataset_seed A
  │   ├─ 算法 1, 重复 1～N
  │   ├─ 算法 2, 重复 1～N
  │   └─ ...
  ├─ dataset_seed B
  │   └─ ...
  └─ ...

## 4. Ground Truth / Reference / Consensus

| 类型 | 含义 | 何时可用 |
|------|------|---------|
| ground_truth_distance | EMD 精确解 (数学真值) | 小场景 (有效维度 ≤ 300) |
| reference_distance | Sinkhorn(reg=0.01) 高精度参考解 | 中大规模场景 |
| consensus_deviation | 算法结果偏离候选算法中位数的程度 | 始终可用 (辅助诊断) |

- ground_truth_error: 相对 ground_truth_distance 的误差。无精确真值时为 ""。
- reference_error: 相对 reference_distance 的误差。无参考解时为 ""。
- consensus_deviation: 辅助诊断指标，标题必须写 "共识偏差"。
- 有 exact 时排行榜优先使用 ground_truth_error。
- 无 exact/ reference 时不显示 "真实误差"。
- 综合评分受参考类型和权重影响，不代表算法绝对优劣。

## 5. 时间测量
- 使用 `time.perf_counter()` 高分辨率单调时钟。
- GPU 运算通过 PyTorch CUDA Events 同步。
- runtime_seconds 包含算法执行的全部时间（归一化、迭代、结果提取）。
- 不包含数据生成和代价矩阵构建时间。

## 6. 内存测量
- CPU 内存: `psutil.Process().memory_info().rss` 进程 RSS 增量。
- GPU 显存: `torch.cuda.memory_allocated()` 额外记录（如有 CUDA）。
- peak_memory_mb = max(before_rss, after_rss)，每 2ms 采样。
- **当前 peak_memory_mb 仅反映 CPU 进程 RSS 增量，不含 GPU 显存。**

## 7. 成功 / 失败判定
| status | 含义 |
|--------|------|
| success | 算法正常返回有限值 |
| timeout | 运行时间超限 |
| out_of_memory | 内存不足 |
| numerical_error | 数值错误 (NaN/Inf/singular) |
| dependency_error | 依赖缺失 |
| not_applicable | 算法不适用该场景 |
| unknown_error | 未分类错误 |

## 8. 统计规则
### 同一 seed 内统计
- 时间、内存、距离的均值/标准差仅基于成功运行。
- 成功样本 < 2 时标准差为空。
- P95 运行时间对成功运行取 95 分位。

### 跨 seed 场景统计
- 先计算每 seed 的均值，再跨 seed 计算均值和标准差。
- 区分同一数据上的运行波动 (within-seed) 与不同数据间的差异 (cross-seed)。

## 9. 算法参数
所有算法使用 registry 注册的默认参数。参数详情见 `algorithms/registry.py`。

---

*本文档由 run_benchmark_v1.py 自动生成 · 修订版*
"""
    path = os.path.join(output_dir, "protocol.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


# ============================================================================
#  统计计算
# ============================================================================

def _safe_stats(values: list[float]) -> dict:
    clean = [v for v in values
             if v is not None and not math.isnan(v) and not math.isinf(v)]
    n = len(clean)
    if n == 0:
        return {"mean": None, "std": None, "p95": None, "median": None, "n": 0}
    result = {"mean": float(np.mean(clean)), "n": n}
    if n >= 2:
        result["std"] = float(np.std(clean, ddof=1))
    else:
        result["std"] = None
    result["median"] = float(np.median(clean))
    result["p95"] = float(np.percentile(clean, 95)) if n >= 20 else None
    return result


def compute_within_seed_summaries(raw_rows: list[dict]) -> list[dict]:
    """按 (scenario_id, algorithm, dataset_seed) 聚合。"""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in raw_rows:
        key = (row.get("scenario_id", ""), row.get("algorithm", ""),
               row.get("dataset_seed", 0))
        groups[key].append(row)

    summaries = []
    for (sid, algo, dseed), rows in groups.items():
        success_rows = [r for r in rows if r.get("status") == "success"]
        total = len(rows)
        n_ok = len(success_rows)

        distances = [_safe_val(r.get("wasserstein_distance"))
                     for r in success_rows]
        gt_errors = [_safe_val(r.get("ground_truth_error"))
                     for r in success_rows]
        ref_errors = [_safe_val(r.get("reference_error"))
                      for r in success_rows]
        cons_devs = [_safe_val(r.get("consensus_deviation"))
                     for r in success_rows]
        runtimes = [_safe_val(r.get("runtime_seconds"))
                    for r in success_rows]
        memories = [_safe_val(r.get("peak_memory_mb"))
                    for r in success_rows]

        fail_counts: dict[str, int] = {}
        for r in rows:
            if r.get("status") != "success":
                s = r.get("status", "unknown_error")
                fail_counts[s] = fail_counts.get(s, 0) + 1

        summaries.append({
            "scenario_id": sid,
            "algorithm": algo,
            "ranking_group": rows[0].get("ranking_group", "balanced"),
            "dataset_seed": dseed,
            "total_runs": total,
            "successful_runs": n_ok,
            "success_rate": n_ok / total if total > 0 else 0.0,
            "mean_distance": _safe_stats(distances)["mean"],
            "std_distance": _safe_stats(distances)["std"],
            "mean_ground_truth_error": _safe_stats(gt_errors)["mean"],
            "std_ground_truth_error": _safe_stats(gt_errors)["std"],
            "mean_reference_error": _safe_stats(ref_errors)["mean"],
            "std_reference_error": _safe_stats(ref_errors)["std"],
            "mean_consensus_deviation": _safe_stats(cons_devs)["mean"],
            "std_consensus_deviation": _safe_stats(cons_devs)["std"],
            "mean_runtime_seconds": _safe_stats(runtimes)["mean"],
            "std_runtime_seconds": _safe_stats(runtimes)["std"],
            "p95_runtime_seconds": _safe_stats(runtimes)["p95"],
            "mean_peak_memory_mb": _safe_stats(memories)["mean"],
            "std_peak_memory_mb": _safe_stats(memories)["std"],
            "failure_count_by_type": fail_counts if fail_counts else None,
        })
    return summaries


def compute_cross_seed_summaries(
    within_seed: list[dict],
) -> list[dict]:
    """跨 seed 聚合: 先取每 seed 均值, 再跨 seed 统计。"""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for ws in within_seed:
        key = (ws["scenario_id"], ws["algorithm"])
        groups[key].append(ws)

    summaries = []
    for (sid, algo), seeds in groups.items():
        total_runs = sum(s.get("total_runs", 0) for s in seeds)
        successful_runs = sum(s.get("successful_runs", 0) for s in seeds)

        distances = [s["mean_distance"] for s in seeds
                     if s["mean_distance"] is not None]
        gt_errors = [s["mean_ground_truth_error"] for s in seeds
                     if s["mean_ground_truth_error"] is not None]
        runtimes = [s["mean_runtime_seconds"] for s in seeds
                    if s["mean_runtime_seconds"] is not None]
        memories = [s["mean_peak_memory_mb"] for s in seeds
                    if s["mean_peak_memory_mb"] is not None]

        fail_counts: dict[str, int] = {}
        for s in seeds:
            fc = s.get("failure_count_by_type")
            if fc:
                for k, v in fc.items():
                    fail_counts[k] = fail_counts.get(k, 0) + v

        summaries.append({
            "scenario_id": sid,
            "algorithm": algo,
            "ranking_group": seeds[0].get("ranking_group", "balanced"),
            "n_seeds": len(seeds),
            "total_runs": total_runs,
            "successful_runs": successful_runs,
            "success_rate": (successful_runs / total_runs
                             if total_runs > 0 else 0.0),
            "mean_distance": _safe_stats(distances)["mean"],
            "std_distance": _safe_stats(distances)["std"],
            "mean_ground_truth_error": _safe_stats(gt_errors)["mean"],
            "std_ground_truth_error": _safe_stats(gt_errors)["std"],
            "mean_runtime_seconds_cross": _safe_stats(runtimes)["mean"],
            "std_runtime_seconds_cross": _safe_stats(runtimes)["std"],
            "mean_peak_memory_mb_cross": _safe_stats(memories)["mean"],
            "std_peak_memory_mb_cross": _safe_stats(memories)["std"],
            "failure_count_by_type": fail_counts if fail_counts else None,
        })
    return summaries


# ============================================================================
#  主实验循环
# ============================================================================

def run_v1_experiment(args: argparse.Namespace) -> int:
    start_time = datetime.now(timezone.utc)

    print("=" * 60)
    print("  OTBench Benchmark v1.0  (修订版)")
    print("=" * 60)
    print()

    # ---- 1. 加载配置 ----
    try:
        all_scenarios = load_benchmark_v1_manifest(args.config)
    except (FileNotFoundError, ConfigValidationError) as e:
        print(f"[ERROR] {e}")
        return 1

    if args.quick:
        scenarios = get_quick_scenarios(all_scenarios)
        n_runs = 2
        seeds_count = 1
        print(f"[quick 模式] 场景: {len(scenarios)}, seed 数: {seeds_count}, 重复: {n_runs}")
    else:
        scenarios = get_enabled_scenarios(all_scenarios)
        n_runs = args.runs
        # 正式模式每场景取全部 seed
        seeds_count = None  # 使用全部
        print(f"[标准模式] 场景: {len(scenarios)}, 重复: {n_runs}")

    if len(scenarios) == 0:
        print("[ERROR] 没有启用的场景")
        return 2

    # ---- 2. 算法过滤 ----
    if args.algorithms == "all":
        selected_algos = dict(ALGORITHMS)
    else:
        names = [n.strip() for n in args.algorithms.split(",")]
        selected_algos = {}
        for n in names:
            for reg_name, info in ALGORITHMS.items():
                if n.lower() in reg_name.lower():
                    selected_algos[reg_name] = info

    if len(selected_algos) == 0:
        print("[ERROR] 没有匹配的算法")
        return 1

    print(f"[算法] {len(selected_algos)} 个:")
    for name in selected_algos:
        print(f"  - {name}")

    # 计算总计划运行数
    total_planned = 0
    for sc in scenarios:
        seeds = sc.get("seeds", [sc.get("seed", 42)])
        if args.quick:
            seeds = seeds[:1]
        total_planned += len(seeds) * len(selected_algos) * n_runs

    print(f"\n[计划] 总计 {total_planned} 次运行\n")

    # ---- 3. 输出目录 ----
    output_dir = args.output
    fb_dir = os.path.join(output_dir, "failure_boundary")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fb_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "raw_results.csv")
    jsonl_path = os.path.join(output_dir, "raw_results.jsonl")

    # ---- 4. Resume ----
    completed_keys: set[tuple] = set()
    existing_rows: list[dict] = []
    if args.resume and os.path.isfile(csv_path):
        print("[resume] 读取已有结果...")
        try:
            with open(csv_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = (row.get("scenario_id", ""), row.get("algorithm", ""),
                           int(row.get("dataset_seed", 0)),
                           int(row.get("repetition_index", 0)))
                    if key[2] >= 0 and key[3] >= 1:
                        completed_keys.add(key)
                        existing_rows.append(row)
            print(f"[resume] 已完成 {len(completed_keys)} 条记录")
        except Exception as e:
            print(f"[WARN] 读取 CSV 失败: {e}")

    # 写入表头 (如果是新文件)
    if not args.resume or not os.path.isfile(csv_path):
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            f.write(codecs.BOM_UTF8.decode("utf-8"))
            csv.writer(f).writerow(CSV_COLUMNS)

    # ---- 5. 运行循环 ----
    all_rows: list[dict] = []
    run_index = 0
    success_count = 0
    fail_count = 0
    skip_count = 0
    run_start = time.perf_counter()

    for sc in scenarios:
        sid = sc["id"]
        sname = sc["name"]
        seeds = sc.get("seeds", [sc.get("seed", 42)])
        if args.quick:
            seeds = seeds[:1]

        for dseed in seeds:
            # 生成数据 (每 seed 一次)
            try:
                dataset = generate_dataset_from_scenario(sc, dseed)
            except Exception as e:
                print(f"[SKIP] {sid} seed={dseed} 数据生成失败: {e}")
                continue

            # 确定参考类型
            gt = _safe_val(dataset.ground_truth_distance)
            gt_is_exact = getattr(dataset, "ground_truth_is_exact", False)
            gt_mode = getattr(dataset, "ground_truth_mode", "unknown")
            gt_method = getattr(dataset, "ground_truth_method", "unknown")

            if gt_is_exact:
                scoring_ref_type = "exact"
                ref_distance = None
            elif (gt is not None and gt_mode == "sinkhorn_ref"
                  and "sinkhorn" in gt_method.lower() and "validated" in gt_method.lower()
                  and "fallback" not in gt_method.lower()):
                scoring_ref_type = "reference"
                ref_distance = gt
            else:
                scoring_ref_type = "unavailable"
                ref_distance = None

            for algo_name, algo_info in selected_algos.items():
                ranking_group = algo_info.get("ranking_group", "balanced")
                # algorithm_seed: 优先取算法超参数中的 seed, 否则用 dataset_seed + 偏移
                algo_seed = dseed + 1000

                for rep in range(1, n_runs + 1):
                    run_index += 1

                    # Resume 跳过
                    resume_key = (sid, algo_name, dseed, rep)
                    if resume_key in completed_keys:
                        skip_count += 1
                        continue

                    row_start = time.perf_counter()
                    result: Optional[AlgorithmResult] = None
                    error: Optional[BaseException] = None

                    try:
                        result = run_single_algorithm(
                            algo_name, algo_info, dataset)
                    except Exception as exc:
                        error = exc

                    row_elapsed = time.perf_counter() - row_start
                    status = classify_status(error, result)

                    # 计算各类误差
                    w_dist = _safe_val(
                        result.wasserstein_distance) if result else None

                    ground_truth_error = None
                    reference_error = None
                    consensus_deviation = None

                    if (status == "success" and w_dist is not None
                            and ranking_group in ("balanced", "reference")):
                        if gt_is_exact and gt is not None and gt != 0:
                            ground_truth_error = abs(w_dist - gt) / abs(gt)
                        elif ref_distance is not None and ref_distance != 0:
                            reference_error = abs(
                                w_dist - ref_distance) / abs(ref_distance)

                    # 构建原始记录
                    exp_id = f"{sid}_seed{dseed}_{algo_name}_rep{rep}"
                    row = {
                        "experiment_id": exp_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "scenario_id": sid,
                        "scenario_name": sname,
                        "dataset_seed": dseed,
                        "algorithm_seed": algo_seed,
                        "repetition_index": rep,
                        "algorithm": algo_name,
                        "ranking_group": ranking_group,
                        "status": status,
                        "failure_type": status if status != "success" else "",
                        "error_message": _sanitize_path(
                            str(error) if error
                            else (result.error_message if result else "")
                        ) if status != "success" else "",
                        "n_source": dataset.n_source,
                        "n_target": dataset.n_target,
                        "dimension": dataset.dimension,
                        "geometry_dimension": int(getattr(dataset, "geometry_dim", 1)),
                        "sparsity": float(sc.get("sparsity", 0)),
                        "sparsity_semantics": dataset.sparsity_semantics,
                        "structure": str(sc.get("structure", "random")),
                        "noise_level": float(sc.get("noise_level", 0)),
                        "balanced": bool(sc.get("balanced", True)),
                        "data_origin": getattr(dataset, "data_origin", "synthetic"),
                        "ground_truth_mode": gt_mode,
                        "ground_truth_method": gt_method,
                        "ground_truth_distance": gt,
                        "reference_distance": ref_distance,
                        "wasserstein_distance": w_dist,
                        "ground_truth_error": ground_truth_error,
                        "reference_error": reference_error,
                        "consensus_deviation": consensus_deviation,
                        "scoring_reference_type": scoring_ref_type,
                        "runtime_seconds": row_elapsed,
                        "peak_memory_mb": (
                            _safe_val(result.peak_memory_mb) if result else None),
                        "convergence_iterations": (
                            result.convergence_iterations if result else 0),
                    }

                    # 即时写 CSV
                    try:
                        with open(csv_path, "a", encoding="utf-8",
                                  newline="") as f:
                            csv.writer(f).writerow([_csv_val(row.get(c, ""))
                                                    for c in CSV_COLUMNS])
                    except Exception as e:
                        print(f"  [WARN] CSV 写入失败: {e}")

                    # 即时写 JSONL
                    try:
                        jsonl_obj = dict(row)
                        if result:
                            jsonl_obj["algorithm_parameters"] = (
                                algo_info.get("hyperparameters", []))
                        if error:
                            jsonl_obj["error_details"] = _sanitize_path(
                                str(error))
                        line = json.dumps(jsonl_obj, ensure_ascii=False,
                                          cls=_SafeEncoder) + "\n"
                        with open(jsonl_path, "a", encoding="utf-8") as f:
                            f.write(line)
                    except Exception as e:
                        print(f"  [WARN] JSONL 写入失败: {e}")

                    all_rows.append(row)

                    if status == "success":
                        success_count += 1
                        dist_s = (f"{w_dist:.6f}" if w_dist is not None
                                  else "N/A")
                        print(f"[{run_index}/{total_planned}] "
                              f"DS={sid} seed={dseed} "
                              f"Algo={algo_name} Rep={rep}/{n_runs} "
                              f"OK ({row_elapsed:.3f}s) dist={dist_s}")
                    else:
                        fail_count += 1
                        print(f"[{run_index}/{total_planned}] "
                              f"DS={sid} seed={dseed} "
                              f"Algo={algo_name} Rep={rep}/{n_runs} "
                              f"{status.upper()} ({row_elapsed:.3f}s)")

    # ---- 6. 实验结束 ----
    total_elapsed = time.perf_counter() - run_start
    end_time = datetime.now(timezone.utc)
    executed = success_count + fail_count

    print()
    print("=" * 60)
    print(f"  完成: {executed} 次 ({success_count} OK, {fail_count} FAIL,"
          f" {skip_count} skip)")
    print(f"  耗时: {total_elapsed:.1f}s")
    print("=" * 60)
    print()

    # ---- 7. 计算 consensus_deviation (事后批量计算) ----
    print("[统计] 计算 consensus_deviation...")
    # 对每个 (scenario_id, dataset_seed) 收集成功算法的距离
    seed_algo_dists: dict[tuple, dict[str, float]] = defaultdict(dict)
    for row in all_rows:
        if row.get("status") == "success":
            key = (row["scenario_id"], row["dataset_seed"])
            wd = _safe_val(row.get("wasserstein_distance"))
            if wd is not None:
                seed_algo_dists[key][row["algorithm"]] = wd

    for key, algo_dists in seed_algo_dists.items():
        if len(algo_dists) < 3:
            continue
        dists = list(algo_dists.values())
        median = float(np.median(dists))
        if median == 0:
            continue
        for row in all_rows:
            if (row["scenario_id"] == key[0]
                    and row["dataset_seed"] == key[1]
                    and row.get("status") == "success"):
                wd = _safe_val(row.get("wasserstein_distance"))
                if wd is not None:
                    row["consensus_deviation"] = abs(wd - median) / abs(median)

    # 重新写入修正后的 raw_results (用 consensus_deviation)
    print("[统计] 回写 corrected raw_results...")
    try:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            f.write(codecs.BOM_UTF8.decode("utf-8"))
            csv.writer(f).writerow(CSV_COLUMNS)
            for row in all_rows:
                csv.writer(f).writerow([_csv_val(row.get(c, ""))
                                       for c in CSV_COLUMNS])
    except Exception as e:
        print(f"  [WARN] CSV 重写失败: {e}")

    # ---- 8. 汇总统计 ----
    if existing_rows:
        all_rows = existing_rows + all_rows

    within_seed = compute_within_seed_summaries(all_rows)
    cross_seed = compute_cross_seed_summaries(within_seed)

    # summary_results.csv (两级合一)
    summary_csv = os.path.join(output_dir, "summary_results.csv")
    with open(summary_csv, "w", encoding="utf-8", newline="") as f:
        f.write(codecs.BOM_UTF8.decode("utf-8"))
        w = csv.writer(f)
        w.writerow(SUMMARY_COLUMNS)
        for s in within_seed:
            w.writerow([_csv_val(s.get(c, "")) for c in SUMMARY_COLUMNS])
    print(f"  已保存: {summary_csv} ({len(within_seed)} rows)")

    cross_csv = os.path.join(output_dir, "summary_cross_seed.csv")
    with open(cross_csv, "w", encoding="utf-8", newline="") as f:
        f.write(codecs.BOM_UTF8.decode("utf-8"))
        w = csv.writer(f)
        w.writerow(CROSS_SEED_COLUMNS)
        for s in cross_seed:
            w.writerow([_csv_val(s.get(c, "")) for c in CROSS_SEED_COLUMNS])
    print(f"  已保存: {cross_csv} ({len(cross_seed)} rows)")

    summary_json = os.path.join(output_dir, "summary_results.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump({
            "meta": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "n_scenarios": len(scenarios),
                "n_algorithms": len(selected_algos),
                "n_repetitions": n_runs,
                "total_executed": executed,
                "success_count": success_count,
                "fail_count": fail_count,
            },
            "within_seed": within_seed,
            "cross_seed": cross_seed,
        }, f, ensure_ascii=False, indent=2, cls=_SafeEncoder)
    print(f"  已保存: {summary_json}")

    # ---- 9. 环境 / 协议 / 清单 ----
    print("[环境] 采集系统信息...")
    config_path = args.config or os.path.join(
        _PROJECT_ROOT, "configs", "benchmark_v1.yaml")
    env_info = capture_environment(config_path)
    env_info["start_time"] = start_time.isoformat()
    env_info["end_time"] = end_time.isoformat()
    env_info["total_elapsed_seconds"] = round(total_elapsed, 1)

    env_path = os.path.join(output_dir, "environment.json")
    with open(env_path, "w", encoding="utf-8") as f:
        json.dump(env_info, f, ensure_ascii=False, indent=2)
    print(f"  已保存: {env_path}")

    protocol_path = generate_protocol_md(
        output_dir, seeds_count or 3, n_runs,
        len(scenarios), len(selected_algos))
    print(f"  已保存: {protocol_path}")

    manifest = {
        "config_file": config_path,
        "config_sha256": env_info.get("config_sha256"),
        "quick_mode": args.quick,
        "n_repetitions": n_runs,
        "total_planned_runs": total_planned,
        "total_executed": executed,
        "total_skipped": skip_count,
        "n_success": success_count,
        "n_failed": fail_count,
        "scenarios": [
            {"id": sc["id"], "name": sc["name"],
             "enabled": sc.get("enabled", True),
             "dimension": sc.get("dimension"),
             "sparsity": sc.get("sparsity"),
             "n_seeds": len(sc.get("seeds", []))}
            for sc in scenarios
        ],
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
    }
    manifest_path = os.path.join(output_dir, "run_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"  已保存: {manifest_path}")

    # ---- 10. 失效边界 ----
    sweep_results = None
    if not args.no_sweep and not args.quick:
        print("\n[失效边界] 扫描中...")
        try:
            sweep_results = sweep_failure_boundary(
                algorithms=selected_algos, verbose=False)
            _gen_fb_outputs(sweep_results, fb_dir)
            print(f"  已保存到: {fb_dir}")
        except Exception as e:
            print(f"  [WARN] 失效边界失败: {e}")
    elif args.quick:
        print("\n[失效边界] 跳过 (quick 模式)")
    else:
        print("\n[失效边界] 跳过 (--no-sweep)")

    # ---- 11. HTML 报告 ----
    if not args.no_report:
        print("\n[报告] 生成 HTML...")
        try:
            _gen_html(within_seed, cross_seed, env_info, output_dir,
                      len(scenarios), len(selected_algos), n_runs,
                      total_planned, executed, success_count, fail_count,
                      skip_count, sweep_results, fb_dir)
        except Exception as e:
            print(f"  [WARN] HTML 报告生成失败: {e}")

    print()
    print("=" * 60)
    print(f"  全部完成!  输出: {os.path.abspath(output_dir)}")
    print("=" * 60)

    return 0


# ============================================================================
#  辅助输出
# ============================================================================

def _gen_fb_outputs(sweep_results: dict, fb_dir: str) -> None:
    """生成失效边界 CSV 和图表。"""
    # CSV
    csv_path = os.path.join(fb_dir, "failure_boundary.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        f.write(codecs.BOM_UTF8.decode("utf-8"))
        w = csv.writer(f)
        w.writerow(["algorithm", "dim", "sparsity", "success",
                     "distance", "error", "runtime", "memory_mb"])
        for algo, sr in sweep_results.items():
            for pt in sr.points:
                w.writerow([algo, pt.dim, pt.sparsity,
                            1 if pt.success else 0,
                            pt.distance, pt.error, pt.runtime, pt.memory_mb])
    print(f"  已保存: {csv_path}")

    # 图表
    try:
        from visualizer import (
            plot_failure_boundary_accuracy,
            plot_failure_boundary_runtime,
            plot_failure_boundary_memory,
            plot_failure_boundary_success,
        )
        for func, fname in [
            (plot_failure_boundary_accuracy, "failure_boundary_accuracy.png"),
            (plot_failure_boundary_runtime, "failure_boundary_runtime.png"),
            (plot_failure_boundary_memory, "failure_boundary_memory.png"),
            (plot_failure_boundary_success, "failure_boundary_success.png"),
        ]:
            sp = os.path.join(fb_dir, fname)
            func(sweep_results, save_path=sp)
            print(f"  已保存: {sp}")
    except Exception as e:
        print(f"  [WARN] 图表生成失败: {e}")


def _gen_html(within_seed, cross_seed, env_info, output_dir,
              n_scenarios, n_algorithms, n_runs, total_planned, executed,
              success_count, fail_count, skip_count,
              sweep_results, fb_dir) -> None:
    """生成 HTML 报告。"""
    try:
        from reporting.v1_report import generate_benchmark_v1_report
        generate_benchmark_v1_report(
            within_seed=within_seed,
            cross_seed=cross_seed,
            env_info=env_info,
            output_dir=output_dir,
            sweep_results=sweep_results,
            fb_dir=fb_dir,
            n_scenarios=n_scenarios,
            n_algorithms=n_algorithms,
            n_runs=n_runs,
            total_planned=total_planned,
            executed=executed,
            success_count=success_count,
            fail_count=fail_count,
            skip_count=skip_count,
        )
    except ImportError:
        _gen_simple_html(within_seed, cross_seed, env_info, output_dir,
                         n_scenarios, n_algorithms, n_runs, total_planned,
                         executed, success_count, fail_count, skip_count,
                         sweep_results, fb_dir)


def _gen_simple_html(within_seed, cross_seed, env_info, output_dir,
                     n_scenarios, n_algorithms, n_runs, total_planned,
                     executed, success_count, fail_count, skip_count,
                     sweep_results, fb_dir) -> None:
    """简易 HTML 回退。"""
    rate = success_count / max(executed, 1)
    groups: dict[str, list[dict]] = defaultdict(list)
    for s in within_seed:
        g = s.get("ranking_group", "balanced")
        groups[g].append(s)

    group_labels = {
        "balanced": "Balanced OT",
        "reference": "参考 (不参与排名)",
        "unbalanced": "非平衡 OT",
        "baseline": "基线",
        "model": "模型",
    }

    algo_html = ""
    for g, label in group_labels.items():
        if g not in groups:
            continue
        algo_html += (f'<tr class="gh"><td colspan="7"><b>{label}</b>'
                      f' <span style="color:#64748b">({g})</span></td></tr>\n')
        for s in sorted(groups[g],
                        key=lambda x: x.get("success_rate", 0) or 0,
                        reverse=True):
            sr = s.get("success_rate")
            sr_s = f"{sr:.1%}" if sr is not None else "N/A"
            c = "#22c55e" if (sr or 0) >= 0.9 else (
                "#f59e0b" if (sr or 0) >= 0.5 else "#ef4444")
            algo_html += (
                f"<tr><td>{s['scenario_id']}</td>"
                f"<td>{s['algorithm']}</td>"
                f"<td style='color:{c}'>{sr_s}</td>"
                f"<td>{_csv_val(s.get('mean_runtime_seconds'))}s</td>"
                f"<td>{_csv_val(s.get('mean_peak_memory_mb'))}MB</td>"
                f"<td>{s.get('successful_runs',0)}/{s.get('total_runs',0)}</td>"
                f"<td>seed={s.get('dataset_seed','')}</td></tr>\n")

    fb_html = ""
    for fn in ["failure_boundary_accuracy.png", "failure_boundary_runtime.png",
               "failure_boundary_memory.png", "failure_boundary_success.png"]:
        fp = os.path.join(fb_dir, fn)
        if os.path.isfile(fp):
            fb_html += (f'<img src="failure_boundary/{fn}" '
                        f'style="max-width:48%;margin:1%">\n')

    fail_html = "<p style='color:#22c55e'>无失败</p>"
    fc_all: dict[str, int] = {}
    for s in within_seed:
        fbc = s.get("failure_count_by_type")
        if fbc:
            for k, v in fbc.items():
                fc_all[k] = fc_all.get(k, 0) + v
    if fc_all:
        fail_html = "<ul>" + "".join(
            f"<li>{k}: {v}</li>" for k, v in
            sorted(fc_all.items(), key=lambda x: -x[1])) + "</ul>"

    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8">
<title>OTBench Benchmark v1.0</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Segoe UI','Microsoft YaHei',sans-serif;background:#0f172a;color:#e2e8f0;padding:24px}}
h1{{color:#6366f1;font-size:24px}}h2{{margin:20px 0 10px;border-bottom:2px solid #1e293b;padding-bottom:6px}}
.card{{background:#1e293b;border-radius:12px;padding:18px;margin:12px 0}}
.metrics{{display:flex;gap:16px;flex-wrap:wrap}}
.metric{{background:#0f172a;border-radius:8px;padding:16px;flex:1;min-width:130px;text-align:center}}
.metric .v{{font-size:28px;font-weight:bold;color:#6366f1}}
.metric .l{{font-size:11px;color:#64748b;margin-top:4px}}
table{{width:100%;border-collapse:collapse;font-size:12px}}
th{{background:#0f172a;padding:8px 10px;text-align:left;color:#94a3b8}}
td{{padding:6px 10px;border-bottom:1px solid #1e293b}}
tr.gh td{{background:#334155;font-size:12px;padding:6px 10px}}
tr:hover td{{background:rgba(99,102,241,.06)}}
a{{color:#6366f1}} .warn{{background:#422006;border:1px solid #f59e0b;border-radius:8px;padding:12px;margin:16px 0;font-size:13px}}
</style></head><body>
<h1>OTBench Benchmark v1.0</h1>
<p style="color:#64748b;font-size:13px">实验报告 · {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</p>
<div class="metrics">
<div class="metric"><div class="v">{n_scenarios}</div><div class="l">场景</div></div>
<div class="metric"><div class="v">{n_algorithms}</div><div class="l">算法</div></div>
<div class="metric"><div class="v">{executed}</div><div class="l">执行次数</div></div>
<div class="metric"><div class="v">{rate:.1%}</div><div class="l">成功率</div></div>
<div class="metric"><div class="v">{n_runs}</div><div class="l">重复/Seed</div></div>
</div>
<div class="warn"><b>注意</b> 综合评分受权重和参考类型影响，不代表绝对优劣。ground_truth_error 仅在有精确解时有效。reference_error 标注为"参考解，非精确真值"。consensus_deviation 为辅助诊断指标。</div>
<h2>实验环境</h2>
<div class="card">
OS: {env_info.get('os','?')} {env_info.get('os_release','')} ·
Python {env_info.get('python_version','?').split()[0]} ·
CPU: {env_info.get('cpu_logical_cores','?')} cores ·
RAM: {env_info.get('total_ram_gb','?')}GB ·
GPU: {env_info.get('gpu_name') or 'N/A'}
</div>
<h2>结果 (按 Seed)</h2><div class="card"><table>
<tr><th>场景</th><th>算法</th><th>成功率</th><th>耗时</th><th>内存</th><th>成功/总数</th><th>Seed</th></tr>
{algo_html}</table></div>
<h2>失败统计</h2><div class="card">{fail_html}</div>
<h2>失效边界</h2><div class="card">{fb_html if fb_html else '<p style="color:#64748b">无数据</p>'}</div>
<h2>文件</h2><div class="card"><ul>
<li><a href="raw_results.csv">raw_results.csv</a> · <a href="raw_results.jsonl">raw_results.jsonl</a></li>
<li><a href="summary_results.csv">summary_results.csv</a> · <a href="summary_results.json">summary_results.json</a></li>
<li><a href="summary_cross_seed.csv">summary_cross_seed.csv</a></li>
<li><a href="environment.json">environment.json</a> · <a href="protocol.md">protocol.md</a> · <a href="run_manifest.json">run_manifest.json</a></li>
<li><a href="failure_boundary/">failure_boundary/</a></li>
</ul></div>
</body></html>"""
    hp = os.path.join(output_dir, "report.html")
    with open(hp, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  已保存: {hp}")


# ============================================================================
#  main
# ============================================================================

def main(argv: Optional[list[str]] = None) -> int:
    return run_v1_experiment(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
