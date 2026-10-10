#!/usr/bin/env python3
"""
OT Benchmark 数据集使用演示
============================
展示如何创建、加载、使用和导出标准化 OT 评测数据集。
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "ot_benchmark"))

import numpy as np
import ot
from data_generator import (
    create_benchmark_dataset, create_dataset_suite,
    generate_sparse_distribution, generate_structured_sparsity,
    apply_distribution_shift, diagnose_dataset, print_dataset_diagnosis,
)
from real_data import create_real_dataset_suite


def demo_1_create_single_dataset():
    """演示 1: 创建一个单独的数据集并跑 OT"""
    print("=" * 60)
    print("  Demo 1: 创建单个数据集 + OT 计算")
    print("=" * 60)

    ds = create_benchmark_dataset(
        n_samples=300, sparsity=0.80, noise_level=0.03,
        source_dist_type="zipf", target_dist_type="pareto",
        random_state=42, name="Demo-Dataset"
    )

    print(f"数据集: {ds.name}")
    print(f"维度: {ds.dim}, 稀疏度: {ds.sparsity:.1%}, 噪声: {ds.noise_level:.1%}")
    print(f"源分布非零元素: {(ds.source_dist > 1e-10).sum()}/{ds.dim}")
    print(f"目标分布非零元素: {(ds.target_dist > 1e-10).sum()}/{ds.dim}")
    print(f"Ground Truth: {ds.ground_truth_distance:.6f}\n")

    # 用不同 reg 跑 Sinkhorn
    a = ds.source_dist / ds.source_dist.sum()
    b = ds.target_dist / ds.target_dist.sum()
    M = ds.cost_matrix

    for reg in [0.1, 0.01, 0.001]:
        d = ot.sinkhorn2(a, b, M, reg, numItermax=2000, stopThr=1e-9)
        err = abs(float(d) - ds.ground_truth_distance) / ds.ground_truth_distance * 100
        print(f"  Sinkhorn (reg={reg:<6})  距离={float(d):.6f}, 误差={err:.2f}%")


def demo_2_batch_load():
    """演示 2: 批量加载所有合成数据集"""
    print("\n" + "=" * 60)
    print("  Demo 2: 批量加载 + 遍历评测")
    print("=" * 60)

    datasets = create_dataset_suite()
    print(f"加载了 {len(datasets)} 个数据集:\n")

    for ds in datasets:
        a = ds.source_dist / ds.source_dist.sum()
        b = ds.target_dist / ds.target_dist.sum()
        M = ds.cost_matrix

        # 用中等正则化跑一把
        d = float(ot.sinkhorn2(a, b, M, 0.01, numItermax=1000, stopThr=1e-7))
        err = abs(d - ds.ground_truth_distance) / (ds.ground_truth_distance + 1e-10) * 100

        print(f"  {ds.name:<30s} dim={ds.dim:>5d}  sp={ds.sparsity:.1%}  "
              f"GT={ds.ground_truth_distance:.4f}  Sinkhorn={d:.4f}  err={err:.1f}%")


def demo_3_structured_sparsity():
    """演示 3: 结构化稀疏数据集"""
    print("\n" + "=" * 60)
    print("  Demo 3: 结构化稀疏")
    print("=" * 60)

    for structure in ["random", "block", "group", "hierarchical"]:
        source = generate_structured_sparsity(
            n_samples=200, sparsity=0.85, structure=structure,
            n_blocks=4, n_groups=8, random_state=42
        )
        nnz = (source > 1e-10).sum()
        print(f"  {structure:<15s}: 非零元={nnz}/200, 稀疏度={1-nnz/200:.1%}")


def demo_4_distribution_shift():
    """演示 4: 分布偏移"""
    print("\n" + "=" * 60)
    print("  Demo 4: 分布偏移 (Distribution Shift)")
    print("=" * 60)

    # 创建源分布
    source = generate_sparse_distribution(300, 0.8, "zipf", 0.03, 42)

    for shift_type in ["covariate", "label", "concept"]:
        target = apply_distribution_shift(
            source, shift_type=shift_type, shift_intensity=0.4, random_state=42
        )
        # 计算源-目标差异
        kl = np.sum(source[source>0] * np.log2(source[source>0] / (target[source>0] + 1e-12)))
        print(f"  {shift_type:<12s}: KL={kl:.4f}, 源非零元={(source>1e-10).sum()}, 目标非零元={(target>1e-10).sum()}")


def demo_5_dataset_diagnosis():
    """演示 5: 数据集诊断"""
    print("\n" + "=" * 60)
    print("  Demo 5: 数据特征诊断")
    print("=" * 60)

    ds = create_benchmark_dataset(
        n_samples=400, sparsity=0.90, noise_level=0.05,
        source_dist_type="zipf", target_dist_type="gaussian_mixture",
        random_state=42, name="Diagnosis-Demo"
    )
    print_dataset_diagnosis(ds)


def demo_6_real_data():
    """演示 6: 真实数据（需要网络）"""
    print("\n" + "=" * 60)
    print("  Demo 6: 真实数据加载")
    print("=" * 60)

    try:
        real_suite = create_real_dataset_suite()
        for ds in real_suite:
            print(f"  {ds.name}: dim={ds.dim}, sparsity={ds.sparsity:.2%}")
    except Exception as e:
        print(f"   真实数据加载失败: {e}")
        print("  跳过 (可能是网络不可用)")


def demo_7_export():
    """演示 7: 导出数据集"""
    print("\n" + "=" * 60)
    print("  Demo 7: 导出数据集为 .npz")
    print("=" * 60)

    ds = create_benchmark_dataset(
        n_samples=200, sparsity=0.75, noise_level=0.02,
        source_dist_type="zipf", target_dist_type="uniform",
        random_state=42, name="Export-Demo"
    )

    export_dir = os.path.join(os.path.dirname(__file__), "exported_data")
    os.makedirs(export_dir, exist_ok=True)

    filepath = os.path.join(export_dir, f"{ds.name}.npz")
    np.savez(filepath,
             name=ds.name,
             source_dist=ds.source_dist,
             target_dist=ds.target_dist,
             cost_matrix=ds.cost_matrix,
             ground_truth=ds.ground_truth_distance,
             dim=ds.dim, sparsity=ds.sparsity, noise_level=ds.noise_level)

    # 验证可以正确加载
    loaded = np.load(filepath)
    loaded_source = loaded["source_dist"]
    loaded_target = loaded["target_dist"]
    loaded_M = loaded["cost_matrix"]

    a = loaded_source / loaded_source.sum()
    b = loaded_target / loaded_target.sum()
    d = float(ot.sinkhorn2(a, b, loaded_M, 0.01, numItermax=1000, stopThr=1e-7))

    print(f"  导出文件: {filepath}")
    print(f"  重新加载后 OT 距离: {d:.6f}")
    print(f"  原始 Ground Truth:   {loaded['ground_truth']:.6f}")
    print(f"  加载验证: OK")


if __name__ == "__main__":
    print("""

     OT Benchmark 数据集使用演示                    
     标准化高维稀疏 OT 评测数据集                   

""")

    demo_1_create_single_dataset()
    demo_2_batch_load()
    demo_3_structured_sparsity()
    demo_4_distribution_shift()
    demo_5_dataset_diagnosis()
    demo_6_real_data()
    demo_7_export()

    print("\n" + "=" * 60)
    print("  全部 Demo 完成！")
    print("=" * 60)
