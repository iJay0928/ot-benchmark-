"""Tests for drop-in external algorithm discovery."""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ot_benchmark"))

from algorithms.plugin_loader import load_algorithm_plugins


def test_loads_conventional_single_file(tmp_path: Path):
    plugin = tmp_path / "demo_algo.py"
    plugin.write_text(
        "ALGORITHM_NAME = 'Demo Plugin'\n"
        "def run_ot(source, target, cost_matrix, **kwargs):\n"
        "    return 0.0\n",
        encoding="utf-8",
    )
    registry = {}
    report = load_algorithm_plugins(tmp_path, registry)
    assert report.errors == []
    assert report.loaded == ["Demo Plugin"]
    assert registry["Demo Plugin"]["func"](None, None, None) == 0.0


def test_loads_repository_adapter(tmp_path: Path):
    repo = tmp_path / "downloaded_repo"
    repo.mkdir()
    (repo / "solver.py").write_text("def solve(a, b, m): return 1.25\n", encoding="utf-8")
    (repo / "ot_plugin.py").write_text(
        "from solver import solve\n"
        "ALGORITHM = {'name': 'Repo Plugin', 'func': solve}\n",
        encoding="utf-8",
    )
    registry = {}
    report = load_algorithm_plugins(tmp_path, registry)
    assert report.errors == []
    assert report.loaded == ["Repo Plugin"]


def test_rejects_duplicate_name(tmp_path: Path):
    (tmp_path / "duplicate.py").write_text(
        "ALGORITHM_NAME = 'Existing'\n"
        "def run_ot(a, b, m): return 0.0\n",
        encoding="utf-8",
    )
    report = load_algorithm_plugins(tmp_path, {"Existing": {}})
    assert not report.loaded
    assert "冲突" in report.errors[0][1]
