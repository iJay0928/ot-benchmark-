"""Run one small end-to-end evaluation of the external plugin example."""

import os
import sys

# The existing project modules use script-style imports (``from algorithms``),
# so expose the source directory in the same way as ot_benchmark/main.py.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ot_benchmark"))

from algorithms.plugin_loader import load_algorithm_plugins
from algorithms.registry import ALGORITHMS
from benchmark_engine import benchmark_dataset
from data_generator import create_benchmark_dataset


report = load_algorithm_plugins("external_algorithms", ALGORITHMS)
if report.errors:
    raise RuntimeError(f"插件加载失败: {report.errors}")

plugin_name = "GitHub Demo Sinkhorn"
comparison_name = "Sinkhorn (reg=0.1)"
plugin_algorithms = {
    plugin_name: ALGORITHMS[plugin_name],
    comparison_name: ALGORITHMS[comparison_name],
}

dataset = create_benchmark_dataset(
    n_samples=80,
    sparsity=0.60,
    noise_level=0.02,
    source_dist_type="zipf",
    target_dist_type="pareto",
    random_state=42,
    name="Plugin-Demo-80D",
)

benchmark_report = benchmark_dataset(
    dataset=dataset,
    algorithms=plugin_algorithms,
    verbose=False,
)

result = benchmark_report.results[0]
score = benchmark_report.weighted_score.get(plugin_name, 0.0)
reference_error = abs(
    result.wasserstein_distance - dataset.ground_truth_distance
) / max(dataset.ground_truth_distance, 1e-10)

print(f"插件: {plugin_name}")
print(f"数据集: {dataset.name}（{dataset.dim} 维，稀疏度 {dataset.sparsity:.0%}）")
print(f"参考距离: {dataset.ground_truth_distance:.8f}")
print(f"算法距离: {result.wasserstein_distance:.8f}")
print(f"相对数据集参考值误差: {reference_error:.2%}")
print(f"相对实用算法共识误差: {result.distance_error:.2%}")
print(f"运行时间: {result.runtime_seconds:.6f} 秒")
print(f"峰值内存: {result.peak_memory_mb:.3f} MB")
print(f"综合得分: {score:.4f}")
print(f"运行状态: {'成功' if result.success else result.error_message}")

comparison = benchmark_report.results[1]
print("\n对照算法")
print(f"算法: {comparison.algorithm_name}")
print(f"算法距离: {comparison.wasserstein_distance:.8f}")
print(f"运行时间: {comparison.runtime_seconds:.6f} 秒")
print(f"综合得分: {benchmark_report.weighted_score.get(comparison_name, 0.0):.4f}")
