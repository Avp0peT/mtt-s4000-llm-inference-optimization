"""Run the declared offline CI scope while rejecting imports of ML runtimes.

Run from the repository root. This is a dependency-boundary check, not MUSA
or CPU model acceptance. The excluded numerical class stays in its test file.
"""
import importlib.abc
import sys

import pytest


class RejectMLImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'torch_musa'}:
            raise ModuleNotFoundError('ML runtime deliberately unavailable: ' + fullname)
        return None


sys.meta_path.insert(0, RejectMLImports())
raise SystemExit(pytest.main([
    'tests/test_workload.py', 'tests/test_metrics.py',
    'tests/test_controller_gate_mode.py', 'tests/test_execution_shape.py',
    'tests/test_cost_execution_contract.py', '-q',
    '-k', 'not TestExecutionShapeRealCpu',
]))
