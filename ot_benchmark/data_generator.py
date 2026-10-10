"""Backward-compatible source-tree shim.

The installable implementation lives in ``ot_benchmark.data_generator``.
This module preserves the legacy ``from data_generator import ...`` imports
used by older scripts and tests when executed from this project directory.
"""

from ot_benchmark.data_generator import *  # noqa: F401,F403
