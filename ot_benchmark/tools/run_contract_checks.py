"""Run data-contract and existing fixture-free scientific regression tests."""
import inspect
import json
import platform
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / 'ot_benchmark'))
from tests import test_benchmark_correctness as existing
from tests.test_data_contract_v02 import DataContractTests

suite = unittest.defaultTestLoader.loadTestsFromTestCase(DataContractTests)
for name, function in inspect.getmembers(existing, inspect.isfunction):
    if name.startswith('test_'):
        suite.addTest(unittest.FunctionTestCase(function, description=name))
result = unittest.TextTestRunner(verbosity=2).run(suite)
output = ROOT / 'validation' / '2026-10-05'
output.mkdir(parents=True, exist_ok=True)
(output / 'contract_checks.json').write_text(json.dumps({
    'python': sys.executable, 'version': platform.python_version(),
    'tests_run': result.testsRun, 'failures': len(result.failures),
    'errors': len(result.errors), 'success': result.wasSuccessful(),
    'scope': '5 new data contract tests and 11 existing scientific checks; not full pytest suite',
}, indent=2), encoding='utf-8')
raise SystemExit(0 if result.wasSuccessful() else 1)
