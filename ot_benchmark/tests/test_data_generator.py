"""Tests for data generation module."""

import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

from data_generator import (
    create_benchmark_dataset,
    create_dataset_suite,
    generate_sparse_distribution,
    generate_structured_sparsity,
    generate_cost_matrix,
    diagnose_dataset,
)


class TestSparseDistribution:
    def test_sums_to_one(self):
        dist = generate_sparse_distribution(100, sparsity=0.8, distribution="zipf", random_state=42)
        assert abs(dist.sum() - 1.0) < 1e-10

    def test_sparsity_level(self):
        dist = generate_sparse_distribution(200, sparsity=0.9, distribution="pareto", random_state=1)
        actual_sparsity = 1.0 - (dist > 1e-12).sum() / len(dist)
        assert actual_sparsity > 0.7  # should be high

    def test_all_distribution_types(self):
        for dist_type in ["zipf", "pareto", "uniform", "gaussian_mixture"]:
            dist = generate_sparse_distribution(50, sparsity=0.5, distribution=dist_type, random_state=42)
            assert len(dist) == 50
            assert abs(dist.sum() - 1.0) < 1e-10

    def test_default_not_all_zero(self):
        dist = generate_sparse_distribution(10, sparsity=0.99, distribution="uniform", random_state=42)
        assert dist.sum() > 1e-10


class TestStructuredSparsity:
    def test_block_structure(self):
        dist = generate_structured_sparsity(200, 0.85, structure="block", n_blocks=4, random_state=42)
        assert abs(dist.sum() - 1.0) < 1e-10

    def test_group_structure(self):
        dist = generate_structured_sparsity(200, 0.85, structure="group", n_groups=8, random_state=42)
        assert abs(dist.sum() - 1.0) < 1e-10

    def test_hierarchical_structure(self):
        dist = generate_structured_sparsity(200, 0.90, structure="hierarchical", random_state=42)
        assert abs(dist.sum() - 1.0) < 1e-10


class TestCostMatrix:
    def test_shape(self):
        source = generate_sparse_distribution(50, 0.7, "zipf", random_state=1)
        target = generate_sparse_distribution(50, 0.7, "pareto", random_state=2)
        dense, csr = generate_cost_matrix(source, target, "euclidean")
        assert dense.shape == (50, 50)
        assert csr.shape == (50, 50)

    def test_symmetry_zero_dist(self):
        x = generate_sparse_distribution(30, 0.5, "uniform", random_state=1)
        dense, _ = generate_cost_matrix(x, x, "euclidean")
        assert np.allclose(np.diag(dense), 0)


class TestBenchmarkDataset:
    def test_create_small(self):
        ds = create_benchmark_dataset(n_samples=100, sparsity=0.5, random_state=42, name="test")
        assert ds.name == "test"
        assert ds.dim == 100
        assert ds.source_dist.shape == (100,)
        assert ds.cost_matrix.shape == (100, 100)
        assert ds.ground_truth_distance >= 0

    def test_dataset_suite(self):
        suite = create_dataset_suite()
        assert len(suite) >= 8
        names = {ds.name for ds in suite}
        assert "DS1-Baseline" in names
        assert "DS5-StressTest" in names


class TestDiagnosis:
    def test_diagnose(self):
        ds = create_benchmark_dataset(n_samples=100, sparsity=0.6, random_state=42, name="test")
        info = diagnose_dataset(ds)
        assert info["n_samples"] == 100
        assert 0 <= info["effective_sparsity"] <= 1
        assert info["entropy_source"] >= 0
        assert "gini_source" in info


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
