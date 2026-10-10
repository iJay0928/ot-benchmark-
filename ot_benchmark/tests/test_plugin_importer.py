import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ot_benchmark"))

from algorithms.plugin_importer import run_import_wizard, scan_repository
from algorithms.plugin_loader import load_algorithm_plugins


def test_auto_import_and_load_transport_plan(tmp_path: Path):
    (tmp_path / "method.py").write_text(
        "import numpy as np\n"
        "def solve_ot(source, target, cost_matrix):\n"
        "    return np.outer(source, target)\n",
        encoding="utf-8",
    )
    candidates = scan_repository(tmp_path)
    assert candidates[0].function == "solve_ot"

    manifest = run_import_wizard(tmp_path, auto=True)
    assert json.loads(manifest.read_text(encoding="utf-8"))["algorithm"]["output"] == "auto"

    registry = {}
    report = load_algorithm_plugins(tmp_path, registry)
    assert not report.errors
    assert report.loaded == ["Solve Ot"]

    import numpy as np
    a = np.array([0.5, 0.5])
    b = np.array([0.25, 0.75])
    cost = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert registry["Solve Ot"]["func"](a, b, cost) == 0.5
