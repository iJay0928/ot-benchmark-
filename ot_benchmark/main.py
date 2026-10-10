#!/usr/bin/env python3
"""
============================================================
  最优传输 (OT) 算法自动评测系统 v1.1
  面向高维稀疏数据的数据集构建与算法评测研究

  华南理工大学 · 计算机科学与工程学院
  大学生创新创业训练计划项目
============================================================

用法:
    python main.py                          # 弹出 GUI 配置权重后运行完整评测
    python main.py --no-ui                  # 跳过 GUI, 使用默认权重
    python main.py --quick                  # 快速测试 (仅2个数据集)
    python main.py --dataset DS1            # 仅评测指定数据集
    python main.py --no-viz                 # 仅终端输出, 不生成图表
    python main.py --sweep                  # 失效边界扫描
    python main.py --sensitivity            # 参数敏感性分析
    python main.py --real-data              # 包含真实数据集
    python main.py --diagnose               # 数据特征诊断
    python main.py --html                   # 生成 HTML 报告
    python main.py --runs 5                 # 每算法运行5次取统计值
    python main.py --all                    # 运行所有分析模式

  权重自定义:
    python main.py --preset speed-first     # 预设: 速度优先
    python main.py --preset accuracy-first  # 预设: 精度优先
    python main.py --w-accuracy 0.5 --w-speed 0.3 --w-memory 0.1 --w-success 0.1
    python main.py --no-ui                 # 跳过 GUI, 使用默认权重
"""

import sys
import os
import argparse
import numpy as np

# 修复 Windows 终端编码问题
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_generator import (
    create_dataset_suite, create_benchmark_dataset,
    create_dataset_suite_with_shifts, create_structured_sparse_datasets,
    diagnose_dataset, print_dataset_diagnosis,
)
from algorithms.registry import ALGORITHMS
from benchmark_engine import (
    run_full_benchmark, benchmark_dataset,
    sweep_failure_boundary, sensitivity_analysis,
    benchmark_with_stats, run_full_statistical_benchmark,
    trace_sinkhorn_convergence, ConvergenceProfile,
    recommend_algorithm, WeightConfig, DEFAULT_WEIGHTS,
)
from backend.base import get_backend, list_available_backends
from weight_config_ui import launch_weight_ui
from dimension_tiers import DIMENSION_TIERS, tier_names, tier_label
from visualizer import (
    generate_full_report, print_ascii_report,
    plot_failure_heatmap, plot_sensitivity_curves,
    plot_convergence_profile, plot_recommendation_matrix,
    plot_dataset_features_radar, plot_stats_comparison,
    generate_html_report,
)

# Rich console
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
from rich.columns import Columns
from rich import box

console = Console()

TITLE_ART = r"""
   ▄▄▄▄▄▄▄ ▄▄▄▄▄▄▄    ▄▄▄▄▄▄▄ ▄▄   ▄▄ ▄▄▄▄▄▄▄ ▄▄▄▄▄▄▄ ▄▄▄▄▄▄▄ ▄▄▄▄▄▄▄
  █       █       █  █       █  █ █  █       █       █       █       █
  █   ▄   █▄     ▄▀  █▄▄▄▄   █  █▄█  █   ▄   █  ▄▄▄▄▄█▄     ▄█   ▄   █
  █  █ █  █ █   █     ▄▄▄▄█  █       █  █ █  █ █▄▄▄▄▄  █   █ █  █ █  █
  █  █▄█  █ █   █    █ ▄▄▄▄▄▄█       █  █▄█  █  ▄▄▄▄▄█ █   █ █  █▄█  █
  █       █ █   █    █ █▄▄▄▄▄█   ▄   █       █ █▄▄▄▄▄  █   █ █       █
  █▄▄▄▄▄▄▄█ █▄▄▄█    █▄▄▄▄▄▄▄█▄▄█ █▄▄█▄▄▄▄▄▄▄█▄▄▄▄▄▄▄█ █▄▄▄█ █▄▄▄▄▄▄▄█
"""


# 预设权重方案
WEIGHT_PRESETS = {
    "balanced": WeightConfig(
        w_accuracy=0.35, w_speed=0.30, w_memory=0.25, w_success=0.10
    ),
    "accuracy-first": WeightConfig(
        w_accuracy=0.55, w_speed=0.20, w_memory=0.15, w_success=0.10
    ),
    "speed-first": WeightConfig(
        w_accuracy=0.20, w_speed=0.50, w_memory=0.20, w_success=0.10
    ),
    "memory-first": WeightConfig(
        w_accuracy=0.20, w_speed=0.20, w_memory=0.50, w_success=0.10
    ),
    "success-only": WeightConfig(
        w_accuracy=0.25, w_speed=0.25, w_memory=0.25, w_success=0.25
    ),
}


def show_algorithms():
    """显示已注册算法"""
    table = Table(
        title="已注册算法",
        box=box.ROUNDED,
        border_style="bright_black",
        title_style="bold cyan",
        header_style="bold white",
    )
    table.add_column("算法名称", style="bold", width=28)
    table.add_column("类型", width=8)
    table.add_column("复杂度", width=14)
    table.add_column("说明", style="dim")

    for name, info in ALGORITHMS.items():
        tag = "[bold green]GPU[/]" if "GPU" in name else "[dim]CPU[/]"
        complexity = info.get("complexity_class", "")
        desc = info.get("description", "")
        # Truncate long names
        display_name = name if len(name) <= 26 else name[:24] + ".."
        table.add_row(display_name, tag, complexity, desc)

    console.print(table)
    console.print()


def show_weight_config(weights: WeightConfig, preset_name: str = ""):
    """显示当前权重配置"""
    tag = f" [dim](预设: {preset_name})[/]" if preset_name else ""

    # Weight distribution bars
    dim_colors = ["red", "yellow", "blue", "green"]
    dim_names = ["精度", "速度", "内存", "成功率"]
    weights_list = [weights.w_accuracy, weights.w_speed, weights.w_memory, weights.w_success]

    bar_parts = []
    for w, c, n in zip(weights_list, dim_colors, dim_names):
        bar_width = int(w * 40)
        bar_parts.append(f"[{c}]{'█' * bar_width}[/] [{c}]{n} {w:.0%}[/]")

    bar_text = "  ".join(bar_parts)

    content = Text()
    content.append("权重配置", style="bold cyan")
    if tag:
        content.append(f" {tag}")
    content.append("\n\n")
    content.append(bar_text)
    content.append(f"\n\n时间上限: [yellow]{weights.max_runtime:.0f}s[/]  |  ")
    content.append(f"内存上限: [yellow]{weights.max_memory:.0f}MB[/]")
    content.append(f"\n\n可切换预设: ")
    content.append(", ".join(f"[dim]--preset {k}[/]" for k in WEIGHT_PRESETS.keys()), style="dim")

    panel = Panel(content, box=box.ROUNDED, border_style="bright_black")
    console.print(panel)
    console.print()


def build_datasets(args) -> list:
    """根据命令行参数构建数据集列表"""
    datasets = []

    if args.quick:
        print("⚡ 快速测试模式: 仅生成 2 个数据集\n")
        datasets = [
            create_benchmark_dataset(
                n_samples=300, sparsity=0.70, noise_level=0.03,
                source_dist_type="zipf", target_dist_type="uniform",
                random_state=1, name="Quick-DS1-ZipfUniform"
            ),
            create_benchmark_dataset(
                n_samples=500, sparsity=0.90, noise_level=0.08,
                source_dist_type="zipf", target_dist_type="pareto",
                random_state=2, name="Quick-DS2-HighSparse"
            ),
        ]
    elif args.dataset:
        print(f"🎯 仅评测指定数据集: {args.dataset}\n")
        all_ds = create_dataset_suite()
        datasets = [ds for ds in all_ds if args.dataset.lower() in ds.name.lower()]
        if not datasets:
            print(f"❌ 未找到数据集: {args.dataset}")
            print(f"可用数据集: {[ds.name for ds in all_ds]}")
            return []
    elif args.real_data:
        print("🌍 真实数据模式: 含真实/模拟数据\n")
        datasets = create_dataset_suite()

        try:
            from real_data import create_real_dataset_suite
            real_datasets = create_real_dataset_suite(
                include_online_bow=not args.no_online_bow,
                bow_dataset=args.bow_dataset,
                bow_terms=args.bow_terms,
                bow_max_docs=args.bow_docs,
                include_online_libsvm=not args.no_online_libsvm,
                libsvm_datasets=args.libsvm_datasets,
                libsvm_features=args.libsvm_features,
                libsvm_max_samples=args.libsvm_samples,
                dimension_tier=args.dimension_tier,
            )
            datasets.extend(real_datasets)
        except (ImportError, OSError, ValueError) as e:
            print(f"⚠ 真实数据加载失败: {e}")
    else:
        print("🔬 完整评测模式: 生成全部数据集\n")
        datasets = create_dataset_suite()
        datasets.extend(create_structured_sparse_datasets())
        datasets.extend(create_dataset_suite_with_shifts())

    datasets = ensure_online_real_datasets(datasets, args, min_count=2)
    return datasets


def configure_dimension_tier(args) -> None:
    """Resolve the dimension tier before any online dataset is loaded."""
    args._bow_terms_explicit = args.bow_terms is not None
    args._libsvm_features_explicit = args.libsvm_features is not None
    if not args.no_ui:
        try:
            from selection_config_ui import launch_dimension_tier_ui
            args.dimension_tier = launch_dimension_tier_ui(args.dimension_tier)
        except Exception as exc:
            print(f"高维档位选择界面不可用，使用 {args.dimension_tier}: {exc}")

    concrete_tiers = tier_names(args.dimension_tier)
    default_dim = int(DIMENSION_TIERS[concrete_tiers[0]]["target_dim"])
    if args.bow_terms is None:
        args.bow_terms = default_dim
    if args.libsvm_features is None:
        args.libsvm_features = default_dim

    selected = ", ".join(
        f"{DIMENSION_TIERS[name]['label']} {DIMENSION_TIERS[name]['target_dim']:,}D"
        for name in concrete_tiers
    )
    print(f"\n高维分级评测模式: {tier_label(args.dimension_tier)} ({selected})")
    print("真实数据将同时显示原始维度、实际评测维度和活跃支撑维度。\n")


def _ensure_online_real_datasets_legacy(datasets: list, args, min_count: int = 2) -> list:
    """确保每次运行至少包含 min_count 个真实来源于网上的数据集。"""
    online_count = sum(
        1 for ds in datasets
        if str(getattr(ds, "data_origin", "")).startswith("online:")
    )
    if online_count >= min_count:
        return datasets

    needed = min_count - online_count
    print(f"\n🌐 本次运行将自动补充 {needed} 个真实来源于网上的数据集 (至少 {min_count} 个)。")

    existing_urls = {
        getattr(ds, "source_url", "")
        for ds in datasets
        if getattr(ds, "source_url", "")
    }

    def current_online_count() -> int:
        return sum(
            1 for ds in datasets
            if str(getattr(ds, "data_origin", "")).startswith("online:")
        )

    # 优先使用 UCI Bag of Words: 证书稳定、格式标准、天然高维稀疏。
    if not getattr(args, "no_online_bow", False):
        try:
            from real_data import UCI_BOW_DATASETS, load_uci_bag_of_words

            selection = (getattr(args, "bow_dataset", "") or "").lower().strip()
            if selection == "all":
                bow_names = list(UCI_BOW_DATASETS.keys())
            else:
                bow_names = [name.strip() for name in selection.split(",") if name.strip()]

            # 保底把 kos/nips 放在前面，确保普通运行也至少有两个在线真实数据。
            for fallback_name in ["kos", "nips"]:
                if fallback_name not in bow_names:
                    bow_names.insert(0, fallback_name)

            for bow_name in bow_names:
                if current_online_count() >= min_count:
                    break
                try:
                    ds = load_uci_bag_of_words(
                        dataset_name=bow_name,
                        n_terms=getattr(args, "bow_terms", 1000),
                        max_docs=getattr(args, "bow_docs", 2000),
                    )
                    if ds.source_url not in existing_urls:
                        datasets.append(ds)
                        existing_urls.add(ds.source_url)
                        print(f"    已加入在线真实数据: {ds.name}")
                except (OSError, ValueError, ImportError, np.linalg.LinAlgError) as e:
                    print(f"    在线 UCI 数据 {bow_name} 加载失败 ({e}), 跳过")
        except (ImportError, ValueError) as e:
            print(f"    UCI 在线数据模块不可用 ({e}), 跳过")

    # 如果 UCI 不足，再尝试 LIBSVM 官方稀疏数据。
    if current_online_count() < min_count and not getattr(args, "no_online_libsvm", False):
        try:
            from real_data import LIBSVM_SPARSE_DATASETS, load_libsvm_sparse_dataset

            selection = (getattr(args, "libsvm_datasets", "") or "").lower().strip()
            if selection == "all":
                libsvm_names = list(LIBSVM_SPARSE_DATASETS.keys())
            else:
                libsvm_names = [name.strip() for name in selection.split(",") if name.strip()]

            for libsvm_name in libsvm_names:
                if current_online_count() >= min_count:
                    break
                try:
                    ds = load_libsvm_sparse_dataset(
                        dataset_name=libsvm_name,
                        n_features=getattr(args, "libsvm_features", 1000),
                        max_samples=getattr(args, "libsvm_samples", 3000),
                    )
                    if ds.source_url not in existing_urls:
                        datasets.append(ds)
                        existing_urls.add(ds.source_url)
                        print(f"    已加入在线真实数据: {ds.name}")
                except (OSError, ValueError, ImportError, np.linalg.LinAlgError) as e:
                    print(f"    在线 LIBSVM 数据 {libsvm_name} 加载失败 ({e}), 跳过")
        except (ImportError, ValueError) as e:
            print(f"    LIBSVM 在线数据模块不可用 ({e}), 跳过")

    if current_online_count() < min_count:
        print(f"⚠ 在线真实数据不足 {min_count} 个，请检查网络或降低维度/样本参数后重试。")

    return datasets


def ensure_online_real_datasets(datasets: list, args, min_count: int = 2) -> list:
    """Ensure at least two online datasets for every selected dimension tier."""
    from real_data import UCI_BOW_DATASETS, load_uci_bag_of_words

    concrete_tiers = tier_names(getattr(args, "dimension_tier", "standard"))
    existing_keys = {
        (getattr(ds, "source_url", ""), getattr(ds, "dimension_tier", "custom"))
        for ds in datasets
        if str(getattr(ds, "data_origin", "")).startswith("online:")
    }

    for tier in concrete_tiers:
        info = DIMENSION_TIERS[tier]
        target_dim = int(info["target_dim"])
        if len(concrete_tiers) == 1 and getattr(args, "_bow_terms_explicit", False):
            target_dim = int(args.bow_terms)

        def tier_count() -> int:
            return sum(
                1 for ds in datasets
                if str(getattr(ds, "data_origin", "")).startswith("online:")
                and getattr(ds, "dimension_tier", "custom") == tier
            )

        needed = max(0, min_count - tier_count())
        if needed == 0:
            continue

        print(
            f"\n在线真实数据补充: {info['label']}档，目标 {target_dim:,} 维，"
            f"还需要 {needed} 个数据集。"
        )
        requested = [
            name.strip().lower()
            for name in str(getattr(args, "bow_dataset", "")).split(",")
            if name.strip().lower() in UCI_BOW_DATASETS
        ]
        candidates = list(info["preferred_bow"]) + requested + list(UCI_BOW_DATASETS)
        seen_names: set[str] = set()
        max_docs = max(int(getattr(args, "bow_docs", 2000)), 5000 if tier == "ultra" else 3000)

        for dataset_name in candidates:
            if tier_count() >= min_count:
                break
            if dataset_name in seen_names:
                continue
            seen_names.add(dataset_name)
            url = f"https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/{UCI_BOW_DATASETS[dataset_name]['docword']}"
            if (url, tier) in existing_keys:
                continue
            if int(UCI_BOW_DATASETS[dataset_name]["words"]) < target_dim:
                continue
            try:
                ds = load_uci_bag_of_words(
                    dataset_name=dataset_name,
                    n_terms=target_dim,
                    max_docs=max_docs,
                    dimension_tier=tier,
                )
                datasets.append(ds)
                existing_keys.add((ds.source_url, tier))
                print(
                    f"    已加入 {ds.name}: 原始 {ds.original_dim:,} 维，"
                    f"评测 {ds.dim:,} 维，活跃支撑 {ds.active_storage_dim} 维"
                )
            except (OSError, ValueError, ImportError, np.linalg.LinAlgError) as exc:
                print(f"    {dataset_name} 加载失败 ({exc})，继续尝试其他真实数据。")

        if tier_count() < min_count:
            print(
                f"警告: {info['label']}档只加载到 {tier_count()} 个在线真实数据集，"
                "请检查网络、缓存或数据维度。"
            )

    return datasets


def print_dataset_overview(datasets, algorithms=None):
    """打印数据集概览"""
    if algorithms is None:
        algorithms = ALGORITHMS
    table = Table(
        title=f"数据集概览 ({len(datasets)} 个数据集, {len(algorithms)} 个算法)",
        box=box.ROUNDED,
        border_style="bright_black",
        title_style="bold cyan",
        header_style="bold white",
    )
    table.add_column("#", style="dim", width=4)
    table.add_column("数据集名称", style="bold")
    table.add_column("维度(评测/原始/活跃)", justify="right")
    table.add_column("稀疏度", justify="right")
    table.add_column("噪声", justify="right")
    table.add_column("源分布")
    table.add_column("目标分布")
    table.add_column("代价几何", overflow="fold")
    table.add_column("来源", overflow="fold")

    for i, ds in enumerate(datasets, 1):
        # Parse distribution info from description
        desc = ds.description if hasattr(ds, 'description') else ""
        src = "zipf" if "zipf" in desc else ("uniform" if "uniform" in desc else "-")
        tgt = "pareto" if "pareto" in desc else ("uniform" if "target=uniform" in desc else "-")
        original_dim = getattr(ds, "original_dim", None) or ds.dim
        active_dim = getattr(ds, "active_storage_dim", len(ds.source_dist))
        table.add_row(
            str(i), ds.name, f"{ds.dim}/{original_dim}/{active_dim}",
            f"{ds.sparsity:.1%}", f"{ds.noise_level:.1%}",
            src, tgt,
            getattr(ds, "geometry_description", "index position (1D)"),
            getattr(ds, "data_origin", "synthetic"),
        )

    console.print(table)
    console.print()


def print_online_data_sources(datasets):
    """运行时明确展示哪些数据是真实来源于互联网。"""
    online_datasets = [
        ds for ds in datasets
        if str(getattr(ds, "data_origin", "")).startswith("online:")
    ]

    if not online_datasets:
        console.print(Panel(
            "本次运行没有加载真实来源于网上的数据集。",
            title="在线真实数据源",
            title_align="left",
            box=box.ROUNDED,
            border_style="bright_black",
        ))
        console.print()
        return

    table = Table(
        title="在线真实数据源 (本次运行)",
        box=box.ROUNDED,
        border_style="cyan",
        title_style="bold cyan",
        header_style="bold white",
    )
    table.add_column("#", style="dim", width=4)
    table.add_column("数据集", style="bold")
    table.add_column("档位/维度")
    table.add_column("来源")
    table.add_column("原始下载 URL", overflow="fold")

    for i, ds in enumerate(online_datasets, 1):
        table.add_row(
            str(i),
            ds.name,
            f"{getattr(ds, 'dimension_tier', 'custom')} | {ds.dim}/{getattr(ds, 'original_dim', None) or ds.dim}/{getattr(ds, 'active_storage_dim', len(ds.source_dist))}",
            getattr(ds, "data_origin", "online"),
            getattr(ds, "source_url", ""),
        )

    console.print(table)
    console.print()


def main():
    parser = argparse.ArgumentParser(
        description="OT 算法自动评测系统 v1.1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py                       # 弹出 GUI 配置权重, 然后运行
  python main.py --no-ui               # 跳过 GUI, 使用默认权重
  python main.py --quick               # 快速评测
  python main.py --no-viz              # 仅终端输出
  python main.py --sweep               # 失效边界扫描
  python main.py --sensitivity         # 参数敏感性分析
  python main.py --real-data           # 含真实数据
  python main.py --diagnose            # 数据特征诊断
  python main.py --html                # 生成 HTML 报告
  python main.py --runs 5              # 统计评测 (5次)
  python main.py --all                 # 全部分析

权重自定义:
  python main.py --preset speed-first      # 预设: 速度优先
  python main.py --preset accuracy-first   # 预设: 精度优先
  python main.py --w-accuracy 0.5 --w-speed 0.3 --w-memory 0.1 --w-success 0.1
        """
    )
    # 基础运行模式
    parser.add_argument("--quick", action="store_true",
                        help="快速测试模式 (仅2个数据集)")
    parser.add_argument("--no-viz", action="store_true",
                        help="不生成可视化图表")
    parser.add_argument("--dataset", type=str, default=None,
                        help="仅评测指定数据集 (如 DS1-Baseline)")

    # 新增: 数据集选项
    parser.add_argument("--real-data", action="store_true",
                        help="加载真实/模拟数据集 (MovieLens, scRNA-seq, UCI Bag of Words)")
    parser.add_argument("--no-online-bow", action="store_true",
                        help="真实数据模式下不下载 UCI Bag of Words 在线文本数据")
    parser.add_argument("--bow-dataset", type=str, default="kos,nips,enron",
                        help="UCI Bag of Words 数据集，逗号分隔或 all (默认: kos,nips,enron)")
    parser.add_argument("--bow-terms", type=int, default=None,
                        help="UCI Bag of Words 选取的高频词维度 (默认: 1000)")
    parser.add_argument("--bow-docs", type=int, default=2000,
                        help="UCI Bag of Words 扫描的最大文档数 (默认: 2000)")
    parser.add_argument("--no-online-libsvm", action="store_true",
                        help="真实数据模式下不下载 LIBSVM 在线稀疏数据")
    parser.add_argument("--libsvm-datasets", type=str,
                        default="colon-cancer,duke,leu,rcv1,news20,real-sim",
                        help="LIBSVM 数据集，逗号分隔或 all")
    parser.add_argument("--libsvm-features", type=int, default=None,
                        help="LIBSVM 选取的高频特征维度 (默认: 1000)")
    parser.add_argument("--libsvm-samples", type=int, default=3000,
                        help="LIBSVM 扫描的最大样本数 (默认: 3000)")
    parser.add_argument("--diagnose", action="store_true",
                        help="输出数据特征诊断信息")

    # 新增: 配置和计算后端
    parser.add_argument("--config", type=str, default=None,
                        help="YAML 配置文件路径")
    parser.add_argument(
        "--dimension-tier", type=str, default="standard",
        choices=["standard", "high", "ultra", "all"],
        help="高维评测档位: standard=1000, high=5000, ultra=10000, all=全部",
    )
    parser.add_argument("--backend", type=str, default="auto",
                        choices=["auto", "numpy", "torch", "cupy"],
                        help="计算后端 (默认: auto)")

    # 新增: 高级分析模式
    parser.add_argument("--sweep", action="store_true",
                        help="运行失效边界扫描")
    parser.add_argument("--sensitivity", action="store_true",
                        help="运行参数敏感性分析")
    parser.add_argument("--convergence", action="store_true",
                        help="追踪算法收敛过程")

    # 新增: 统计选项
    parser.add_argument("--runs", type=int, default=1,
                        help="每个算法运行次数 (取统计值, 默认1)")

    # 新增: 输出选项
    parser.add_argument("--html", action="store_true",
                        help="生成 HTML 报告")

    # 一键全开
    parser.add_argument("--all", action="store_true",
                        help="运行所有分析模式")

    # ---- 权重自定义 ----
    weight_group = parser.add_argument_group("个性化权重配置")
    weight_group.add_argument("--preset", type=str, default=None,
                              choices=list(WEIGHT_PRESETS.keys()),
                              help=f"使用预设权重方案: {', '.join(WEIGHT_PRESETS.keys())}")
    weight_group.add_argument("--w-accuracy", type=float, default=None,
                              help="精度权重 (默认 0.35)")
    weight_group.add_argument("--w-speed", type=float, default=None,
                              help="速度权重 (默认 0.30)")
    weight_group.add_argument("--w-memory", type=float, default=None,
                              help="内存权重 (默认 0.25)")
    weight_group.add_argument("--w-success", type=float, default=None,
                              help="成功率权重 (默认 0.10)")
    weight_group.add_argument("--max-runtime", type=float, default=60.0,
                              help="时间归一化上限/秒 (默认 60.0)")
    weight_group.add_argument("--max-memory", type=float, default=2048.0,
                              help="内存归一化上限/MB (默认 2048.0)")
    weight_group.add_argument("--no-ui", action="store_true",
                              help="跳过 GUI，直接使用默认权重")

    args = parser.parse_args()

    # ---- BANNER ----
    console.print(Text(TITLE_ART, style="bold cyan", justify="center"))
    console.print(Panel(
        Align.center("[bold white]面向高维稀疏数据的数据集构建与算法评测研究[/]\n"
                     "[dim]华南理工大学 · 计算机科学与工程学院  |  大学生创新创业训练计划项目[/]"),
        box=box.HEAVY, border_style="cyan", padding=(1, 2)
    ))

    # ---- 加载配置文件 ----
    if args.config:
        from core.config import load_config
        cfg = load_config(args.config)
        if not args.no_ui:
            args.max_runtime = cfg.weight_config.max_runtime
            args.max_memory = cfg.weight_config.max_memory

    # ---- 初始化计算后端 ----
    backend = get_backend(args.backend)
    available = list_available_backends()
    is_gpu = any(gpu in backend.name.lower() for gpu in ("cuda", "gpu", "cupy", "mps"))

    if is_gpu:
        import torch
        gpu_name_t = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "GPU"
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0

        gpu_text = Text()
        gpu_text.append("  GPU 加速引擎", style="bold green")
        gpu_text.append(f"  {backend.name}", style="dim")
        gpu_text.append(f"\n  {gpu_name_t}  |  {vram:.1f} GB VRAM  |  CUDA {torch.version.cuda}", style="bright_white")
        gpu_text.append("\n  加速算法: ")
        gpu_text.append("Sinkhorn GPU (reg=0.1)", style="bold yellow")
        gpu_text.append("  ")
        gpu_text.append("Sinkhorn GPU (reg=0.01)", style="bold yellow")

        console.print(Panel(gpu_text, box=box.HEAVY, border_style="green", padding=(1, 2)))
    else:
        cpu_text = Text()
        cpu_text.append("  CPU 模式", style="bold white")
        cpu_text.append(f"\n  可用后端: {', '.join(available)}", style="dim")
        if "torch" in available:
            cpu_text.append("\n  提示: 使用 ", style="yellow")
            cpu_text.append("--backend torch", style="bold yellow")
            cpu_text.append(" 启用 GPU 加速", style="yellow")
        console.print(Panel(cpu_text, box=box.ROUNDED, border_style="bright_black", padding=(1, 2)))

    console.print()

    # ---- 构建权重配置 ----
    has_cli_weights = (
        args.preset
        or any(v is not None for v in [args.w_accuracy, args.w_speed, args.w_memory, args.w_success])
    )

    if args.preset:
        weight_config = WEIGHT_PRESETS[args.preset]
        preset_name = args.preset
    elif any(v is not None for v in [args.w_accuracy, args.w_speed, args.w_memory, args.w_success]):
        # 用户通过命令行自定义权重：未指定的保持默认值
        def_w = DEFAULT_WEIGHTS
        weight_config = WeightConfig(
            w_accuracy=args.w_accuracy if args.w_accuracy is not None else def_w.w_accuracy,
            w_speed=args.w_speed if args.w_speed is not None else def_w.w_speed,
            w_memory=args.w_memory if args.w_memory is not None else def_w.w_memory,
            w_success=args.w_success if args.w_success is not None else def_w.w_success,
            max_runtime=args.max_runtime,
            max_memory=args.max_memory,
        )
        preset_name = ""
    elif args.no_ui:
        # 跳过 GUI，使用默认权重
        weight_config = WeightConfig(
            max_runtime=args.max_runtime,
            max_memory=args.max_memory,
        )
        preset_name = ""
    else:
        # 默认：启动 GUI 窗口让用户拖拽配置
        print("\n🎛️  正在启动权重配置界面...\n")
        try:
            weight_config = launch_weight_ui()
        except Exception as e:
            console.print(Panel(
                f"权重配置界面不可用，已使用默认权重继续运行。\n{e}",
                box=box.ROUNDED,
                border_style="yellow",
            ))
            weight_config = WeightConfig(
                max_runtime=args.max_runtime,
                max_memory=args.max_memory,
            )
        preset_name = ""
        # 重新显示 Banner
        console.print(Text(TITLE_ART, style="bold cyan", justify="center"))
        console.print(Panel(
            Align.center("[bold white]面向高维稀疏数据的数据集构建与算法评测研究[/]\n"
                         "[dim]华南理工大学 · 计算机科学与工程学院  |  大学生创新创业训练计划项目[/]"),
            box=box.HEAVY, border_style="cyan", padding=(1, 2)
        ))

    show_algorithms()
    show_weight_config(weight_config, preset_name)

    # --all 覆盖其他选项
    if args.all:
        args.sweep = True
        args.sensitivity = True
        args.convergence = True
        args.diagnose = True
        args.real_data = True
        args.html = True
        if args.runs < 3:
            args.runs = 3

    configure_dimension_tier(args)

    output_dir = os.path.join(os.path.dirname(__file__), "benchmark_results")
    os.makedirs(output_dir, exist_ok=True)

    # ---- 模式 1: 失效边界扫描 ----
    if args.sweep:
        print("\n" + "=" * 70)
        print("  🔍 模式: 失效边界扫描")
        print("=" * 70)
        sweep_results = sweep_failure_boundary(
            algorithms=ALGORITHMS,
            dims=[100, 200, 300, 500, 800, 1000],
            sparsities=[0.5, 0.70, 0.85, 0.95, 0.99],
        )
        heatmap_path = os.path.join(output_dir, "failure_heatmap.png")
        plot_failure_heatmap(sweep_results, heatmap_path)
        print(f"📁 失效热力图已保存: {heatmap_path}")
        if not args.all:
            return

    # ---- 模式 2: 参数敏感性分析 ----
    if args.sensitivity:
        print("\n" + "=" * 70)
        print("  🔬 模式: 参数敏感性分析")
        print("=" * 70)

        # 使用中等规模数据集
        ds = create_benchmark_dataset(
            n_samples=400, sparsity=0.85, noise_level=0.03,
            source_dist_type="zipf", target_dist_type="pareto",
            random_state=99, name="SensitivityTest"
        )

        all_sensitivity = []

        print("\n--- Sinkhorn 正则化参数 ---")
        sinkhorn_sens = sensitivity_analysis(ds, algo_class="sinkhorn", param_name="reg")
        all_sensitivity.extend(sinkhorn_sens)
        sens_path = os.path.join(output_dir, "sensitivity_sinkhorn.png")
        plot_sensitivity_curves(sinkhorn_sens, sens_path)

        print("\n--- Sliced OT 投影数 ---")
        sliced_sens = sensitivity_analysis(ds, algo_class="sliced", param_name="n_projections")
        all_sensitivity.extend(sliced_sens)
        sens_path2 = os.path.join(output_dir, "sensitivity_sliced.png")
        plot_sensitivity_curves(sliced_sens, sens_path2)

        print("\n--- Low-Rank OT 秩 ---")
        lr_sens = sensitivity_analysis(ds, algo_class="lowrank", param_name="rank")
        all_sensitivity.extend(lr_sens)
        sens_path3 = os.path.join(output_dir, "sensitivity_lowrank.png")
        plot_sensitivity_curves(lr_sens, sens_path3)

        print(f"📁 敏感性图表已保存至: {output_dir}")
        if not args.all:
            return

    # ---- 模式 3: 收敛追踪 ----
    if args.convergence:
        print("\n" + "=" * 70)
        print("  📉 模式: 收敛性追踪")
        print("=" * 70)

        ds = create_benchmark_dataset(
            n_samples=300, sparsity=0.85, noise_level=0.03,
            source_dist_type="zipf", target_dist_type="pareto",
            random_state=77, name="ConvergenceTest"
        )

        profiles = []
        for reg_val in [0.01, 0.1, 1.0]:
            print(f"\n  追踪 Sinkhorn (reg={reg_val})...")
            profile = trace_sinkhorn_convergence(ds, reg=reg_val, record_every=5)
            profiles.append(profile)
            conv_path = os.path.join(output_dir, f"convergence_reg{reg_val}.png")
            plot_convergence_profile(profile, conv_path)
            print(f"  最终迭代: {profile.final_iterations}, 收敛: {profile.converged}")

        print(f"📁 收敛曲线已保存至: {output_dir}")
        if not args.all:
            return

    # ---- 主评测流程 ----
    datasets = build_datasets(args)
    if not datasets:
        return

    algorithms_to_run = ALGORITHMS
    if not args.no_ui:
        try:
            from selection_config_ui import launch_selection_ui
            console.print(Panel(
                "请选择本次要评测的算法和数据集。",
                box=box.ROUNDED,
                border_style="bright_black",
            ))
            algorithms_to_run, datasets = launch_selection_ui(ALGORITHMS, datasets)
        except Exception as e:
            console.print(Panel(
                f"选择界面启动失败，默认使用全部算法和数据集。\n{e}",
                box=box.ROUNDED,
                border_style="yellow",
            ))

    if not algorithms_to_run or not datasets:
        console.print(Panel("未选择算法或数据集，评测已取消。", box=box.ROUNDED, border_style="yellow"))
        return

    datasets = ensure_online_real_datasets(datasets, args, min_count=2)

    # 数据诊断
    if args.diagnose:
        print("\n" + "=" * 70)
        print("  🩺 数据特征诊断")
        print("=" * 70)
        for ds in datasets:
            print_dataset_diagnosis(ds)

        # 所有数据集特征雷达图
        radar_path = os.path.join(output_dir, "dataset_features_radar.png")
        plot_dataset_features_radar(datasets, radar_path)
        print(f"📁 数据集特征雷达图: {radar_path}")

    print_dataset_overview(datasets, algorithms_to_run)
    print_online_data_sources(datasets)

    # 运行评测
    if args.runs > 1:
        console.print(Panel(f"统计评测 (每个算法运行 {args.runs} 次)",
                     box=box.HEAVY, border_style="cyan"))
        reports, stats_list = run_full_statistical_benchmark(
            datasets, algorithms_to_run, n_runs=args.runs, weights=weight_config,
            backend=backend,
        )
        for report, stats in zip(reports, stats_list):
            stats_path = os.path.join(output_dir, f"{report.dataset_name}_stats.png")
            plot_stats_comparison(stats, report.dataset_name, stats_path)
    else:
        reports = run_full_benchmark(datasets, algorithms_to_run, weights=weight_config, backend=backend)
        stats_list = None

    # Generate visualizations
    if not args.no_viz:
        console.print(Panel("Generating charts...", box=box.ROUNDED, border_style="dim"))
        files = generate_full_report(reports, output_dir)
        rec_path = os.path.join(output_dir, "recommendation_matrix.png")
        plot_recommendation_matrix(reports, rec_path)
        files["recommendation_matrix"] = rec_path

        file_table = Table(box=box.ROUNDED, border_style="bright_black", header_style="bold white")
        file_table.add_column("Chart", style="bold")
        file_table.add_column("Size", justify="right")
        for name, path in files.items():
            if os.path.exists(path):
                file_table.add_row(f"{name}.png", f"{os.path.getsize(path)/1024:.1f} KB")
        file_table.add_row("[bold]Output directory[/]", f"[bold]{output_dir}[/]")
        console.print(file_table)

    if args.html:
        console.print(Panel("Generating HTML report...", box=box.ROUNDED, border_style="dim"))
        html_path = generate_html_report(
            reports, output_dir, stats_list=stats_list, all_datasets=datasets,
        )

    # Final summary
    reference_algorithms = {
        result.algorithm_name
        for report in reports
        for result in report.results
        if getattr(result, "is_reference", False)
    }
    algo_scores = {}
    for report in reports:
        for name, score in report.overall_ranking:
            if name in reference_algorithms:
                continue
            if name not in algo_scores:
                algo_scores[name] = []
            algo_scores[name].append(score)

    avg_scores = [(name, sum(scores)/len(scores)) for name, scores in algo_scores.items()]
    avg_scores.sort(key=lambda x: x[1], reverse=True)

    summary_table = Table(
        title="Final Practical Sparse Rankings (cross-dataset average)",
        box=box.ROUNDED, border_style="cyan",
        title_style="bold cyan", header_style="bold white",
    )
    summary_table.add_column("#", style="dim", width=4)
    summary_table.add_column("Algorithm", style="bold")
    summary_table.add_column("Avg Score", justify="right")
    summary_table.add_column("Distribution")

    for rank, (name, avg) in enumerate(avg_scores, 1):
        bar_len = int(avg * 25)
        bar = "[green]" + "█" * bar_len + "[/]" + "[dim]" + "░" * (25 - bar_len) + "[/]"
        rank_style = "bold yellow" if rank <= 3 else "dim"
        summary_table.add_row(f"[{rank_style}]{rank}[/]", name, f"[bold]{avg:.4f}[/]", bar)

    console.print(summary_table)
    console.print("[dim]Ranking note: EMD exact/reference, Gaussian model-assumption, and auxiliary baselines are shown in reports but excluded from the practical leaderboard.[/]")
    console.print(Panel("[bold green]Benchmark Complete[/]", box=box.HEAVY, border_style="green"))


if __name__ == "__main__":
    main()
