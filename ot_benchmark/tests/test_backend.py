"""Tests for compute backend abstraction layer."""

import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))


class TestNumPyBackend:
    def test_available(self):
        from backend.numpy_backend import NumPyBackend
        assert NumPyBackend.available()

    def test_basic_ops(self):
        from backend.numpy_backend import NumPyBackend
        be = NumPyBackend()
        arr = be.ones(10)
        assert len(be.to_numpy(arr)) == 10
        assert abs(be.sum(arr) - 10.0) < 1e-10


class TestBackendAutoDetect:
    def test_auto_detect(self):
        from backend.base import get_backend, list_available_backends
        be = get_backend("auto")
        assert be is not None
        assert "numpy" in list_available_backends()

    def test_explicit_numpy(self):
        from backend.base import get_backend
        be = get_backend("numpy")
        assert "NumPy" in be.name

    def test_explicit_invalid_falls_back(self):
        from backend.base import get_backend
        be = get_backend("nonexistent_backend")
        assert be is not None
        assert "NumPy" in be.name


class TestTorchBackend:
    def test_importable(self):
        try:
            from backend.torch_backend import TorchBackend
            be = TorchBackend()
            assert be.name is not None
        except ImportError:
            pytest.skip("PyTorch not installed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
