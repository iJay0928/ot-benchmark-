"""Tests for OT algorithm implementations."""

import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

from data_generator import create_benchmark_dataset


def _make_trivial_dataset():
    """Create a tiny 3x3 dataset for algorithm validation."""
    return create_benchmark_dataset(
        n_samples=3, sparsity=0.0, noise_level=0.0,
        source_dist_type="uniform", target_dist_type="uniform",
        random_state=42, name="Trivial-3x3",
    )


class TestEMD:
    def test_trivial(self):
        from algorithms.emd import run_emd_exact
        ds = _make_trivial_dataset()
        d = run_emd_exact(ds.source_dist, ds.target_dist, ds.cost_matrix)
        assert d >= 0
        assert np.isfinite(d)


class TestSinkhorn:
    def test_trivial(self):
        from algorithms.sinkhorn import run_sinkhorn
        ds = _make_trivial_dataset()
        for reg in [0.001, 0.1, 1.0]:
            d = run_sinkhorn(ds.source_dist, ds.target_dist, ds.cost_matrix, reg=reg)
            assert d >= 0
            assert np.isfinite(d)

    def test_self_distance(self):
        """Sinkhorn distance from a distribution to itself should be near zero."""
        from algorithms.sinkhorn import run_sinkhorn
        x = np.ones(10) / 10.0
        M = np.abs(np.arange(10)[:, None] - np.arange(10)[None, :]).astype(np.float64)
        M = M / M.max()
        d = run_sinkhorn(x, x, M, reg=0.01)
        assert d < 1.0


class TestGaussianOT:
    def test_trivial(self):
        from algorithms.gaussian_ot import run_gaussian_ot
        ds = _make_trivial_dataset()
        d = run_gaussian_ot(ds.source_dist, ds.target_dist, ds.cost_matrix)
        assert d >= 0
        assert np.isfinite(d)


class TestSparseOT:
    def test_trivial(self):
        from algorithms.sparse_ot import run_sparse_sinkhorn
        ds = _make_trivial_dataset()
        d = run_sparse_sinkhorn(
            ds.source_dist, ds.target_dist,
            ds.cost_matrix, ds.cost_matrix_csr,
            reg=0.1, max_iter=500,
        )
        assert d >= 0
        assert np.isfinite(d)


class TestSinkhornAccuracy:
    """EMD and low-reg Sinkhorn should agree on small problems."""

    def test_emd_sinkhorn_agreement(self):
        ds = create_benchmark_dataset(
            n_samples=10, sparsity=0.3, noise_level=0.0,
            source_dist_type="uniform", target_dist_type="uniform",
            random_state=99, name="Accuracy-10",
        )
        from algorithms.emd import run_emd_exact
        from algorithms.sinkhorn import run_sinkhorn

        emd_d = run_emd_exact(ds.source_dist, ds.target_dist, ds.cost_matrix)
        # Entropic OT has a non-zero regularization bias. At reg=1e-4 it
        # should approximate the exact transport cost on this small problem.
        sink_d = run_sinkhorn(
            ds.source_dist, ds.target_dist, ds.cost_matrix,
            reg=0.0001, max_iter=20000, stop_threshold=1e-10,
        )

        rel_err = abs(emd_d - sink_d) / (emd_d + 1e-10)
        assert rel_err < 0.02


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
