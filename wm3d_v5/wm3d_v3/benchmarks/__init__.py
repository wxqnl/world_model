"""Benchmark harness probes for wm3d_v3.

The project can be a useful world model before external robot benchmarks are
installed. This package makes that distinction explicit: internal offline
checks can run now, while professional suites report whether their Python
packages or entrypoints are available.
"""

from .adapter import BenchmarkAdapter, BenchmarkTask, EpisodeResult, TokenizedObservation
from .registry import BenchmarkProbe, probe_benchmarks

__all__ = [
    "BenchmarkAdapter",
    "BenchmarkProbe",
    "BenchmarkTask",
    "EpisodeResult",
    "TokenizedObservation",
    "probe_benchmarks",
]
