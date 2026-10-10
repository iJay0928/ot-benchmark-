"""Correctness tests for geometry-aware sparse OT benchmarking."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from algorithms.sinkhorn import run_sinkhorn
from algorithms.sliced_ot import run_sliced_ot
from algorithms.sparse_ot import run_sparse_sinkhorn
from benchmark_engine import (
    AlgorithmResult,
    WeightConfig,
    compute_ranked_scores,
    run_single_algorithm,
)
from data_generator import (
    SparseDataset,
    compute_ot_reference,
    generate_cost_matrix,
    generate_sparse_distribution,
    normalize_support_points,
)
from dimension_tiers import tier_names, tier_target_dim


def test_noise_preserves_sparse_structure():
    distribution = generate_sparse_distribution(
        1000, sparsity=0.95, distribution="uniform",
        noise_level=0.1, random_state=7,
    )
    actual_sparsity = 1.0 - np.count_nonzero(distribution) / len(distribution)
    assert actual_sparsity > 0.9


def test_support_geometry_controls_ground_cost():
    weights = np.array([0.5, 0.3, 0.2])
    index_cost, _ = generate_cost_matrix(weights, weights)
    semantic_points = np.array([[0.0], [10.0], [0.1]])
    semantic_cost, _ = generate_cost_matrix(
        weights, weights, support_points=semantic_points,
    )
    assert index_cost[0, 2] > index_cost[0, 1]
    assert semantic_cost[0, 2] < semantic_cost[0, 1]


def test_exact_reference_uses_active_support():
    source = np.zeros(1000)
    target = np.zeros(1000)
    source[[2, 500]] = [0.6, 0.4]
    target[[2, 900]] = [0.2, 0.8]
    cost, _ = generate_cost_matrix(source, target)
    distance, method, is_exact, _ = compute_ot_reference(source, target, cost)
    assert distance >= 0
    assert is_exact
    assert "active support=3" in method


def test_sinkhorn_reference_mode_cannot_upgrade_to_exact():
    source = np.zeros(40)
    target = np.zeros(40)
    source[[2, 20]] = [0.6, 0.4]
    target[[2, 30]] = [0.2, 0.8]
    cost, _ = generate_cost_matrix(source, target)
    distance, method, is_exact, algorithm = compute_ot_reference(
        source, target, cost, allow_exact=False,
    )
    assert distance >= 0
    assert not is_exact
    assert "Sinkhorn" in method
    assert algorithm == "Sinkhorn (reg=0.01)"


def test_sparse_sinkhorn_agrees_with_dense_sinkhorn():
    rng = np.random.RandomState(11)
    source = rng.random(40)
    target = rng.random(40)
    source[source < 0.7] = 0
    target[target < 0.7] = 0
    source /= source.sum()
    target /= target.sum()
    cost, cost_csr = generate_cost_matrix(source, target)
    dense = run_sinkhorn(source, target, cost, reg=0.01, max_iter=5000)
    sparse = run_sparse_sinkhorn(
        source, target, cost, cost_csr, reg=0.01, max_iter=5000,
    )
    assert np.isclose(sparse, dense, rtol=2e-3, atol=1e-6)


def test_sliced_ot_identity_is_zero():
    weights = np.array([0.1, 0.2, 0.3, 0.4])
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    cost, _ = generate_cost_matrix(weights, weights, support_points=points)
    distance = run_sliced_ot(
        weights, weights, cost, n_projections=100,
        support_points=points,
    )
    assert distance < 1e-12


def test_auxiliary_objectives_are_excluded_from_balanced_ranking():
    def result(name, error, runtime, memory, group="balanced"):
        return AlgorithmResult(
            algorithm_name=name,
            wasserstein_distance=1.0,
            runtime_seconds=runtime,
            peak_memory_mb=memory,
            distance_error=error,
            convergence_iterations=10,
            success=True,
            ranking_group=group,
        )

    results = [
        result("balanced-a", 0.01, 1.0, 10.0),
        result("balanced-b", 0.10, 0.5, 5.0),
        result("unbalanced", 0.0, 0.01, 1.0, "unbalanced"),
    ]
    ranking = compute_ranked_scores(results, WeightConfig())
    assert {name for name, _ in ranking} == {"balanced-a", "balanced-b"}


def test_dimension_tiers_expand_to_expected_targets():
    assert tier_target_dim("standard") == 1000
    assert tier_target_dim("high") == 5000
    assert tier_target_dim("ultra") == 10000
    assert tier_names("all") == ["standard", "high", "ultra"]


def test_support_normalization_handles_ten_thousand_points():
    points = np.random.RandomState(13).normal(size=(10000, 4))
    normalized = normalize_support_points(points)
    assert normalized.shape == points.shape
    assert np.isfinite(normalized).all()


def test_benchmark_geometry_dimension_is_explicit_and_reproducible():
    from data_generator import create_benchmark_dataset

    first = create_benchmark_dataset(
        n_samples=40, sparsity=0.5, random_state=123,
        source_dist_type="uniform", target_dist_type="uniform",
        geometry_dim=4,
    )
    second = create_benchmark_dataset(
        n_samples=40, sparsity=0.5, random_state=123,
        source_dist_type="uniform", target_dist_type="uniform",
        geometry_dim=4,
    )
    assert first.geometry_dim == 4
    assert first.support_points.shape == (len(first.source_dist), 4)
    assert np.array_equal(first.support_points, second.support_points)
    assert np.array_equal(first.cost_matrix, second.cost_matrix)


def test_not_applicable_algorithm_is_excluded_from_ranking():
    weights = np.array([0.2, 0.3, 0.5])
    cost, cost_csr = generate_cost_matrix(weights, weights)
    dataset = SparseDataset(
        name="ambient-high-dimensional",
        source_dist=weights,
        target_dist=weights,
        cost_matrix=cost,
        cost_matrix_csr=cost_csr,
        ground_truth_distance=0.0,
        dim=10000,
        sparsity=0.999,
        noise_level=0.0,
        description="test",
    )
    skipped = run_single_algorithm(
        "limited",
        {
            "func": lambda *_args, **_kwargs: 0.0,
            "max_dim": float("inf"),
            "max_active_dim": 2,
            "ranking_group": "balanced",
        },
        dataset,
    )
    valid = AlgorithmResult(
        algorithm_name="valid",
        wasserstein_distance=0.0,
        runtime_seconds=0.1,
        peak_memory_mb=1.0,
        distance_error=0.0,
        convergence_iterations=0,
        success=True,
    )
    assert skipped.not_applicable
    assert compute_ranked_scores([skipped, valid], WeightConfig()) == [("valid", 1.0)]
