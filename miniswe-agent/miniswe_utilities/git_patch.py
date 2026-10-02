"""Compatibility import for the packaged benchmark patch utilities."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module(
    "minisweagent.run.benchmarks.utils.git_patch"
)
