"""Run reproducible shape sweeps across NumPy and native CPU/CUDA engines.

This file is the stable entry point. The implementation lives in
``scripts/benchmarks/shape_sweep/``:

===================== ======================================================
``shapes``            record schema, benchmark shapes, dataset generation
``environment``       comparison metadata for the capturing machine
``timing``            lifecycle calibration and the measurement discipline
``runners``           one runner per engine and transport contract
``cli``               argument parsing, orchestration and JSON output
===================== ======================================================

The names re-exported below are the ones
``scripts/benchmarks/benchmark_native_cpu_scaling.py`` and
``tests/test_benchmark_performance_policy.py`` import. They keep their original
spelling, including the leading underscore, so that neither the split nor a
later move can be mistaken for an intentional change to what those consumers
depend on. The emitted JSON is byte-for-byte the same schema-v2 record.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Running this file directly puts ``scripts/benchmarks`` on the path, not the
# repository root, so the package below would not resolve. ``src`` is added for
# the same reason the rest of the benchmark scripts add it: these run straight
# from a checkout.
for _entry in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from scripts.benchmarks.shape_sweep.cli import (  # noqa: E402
    _add_throughput,
    _print_result,
    _record_skip,
    main,
)
from scripts.benchmarks.shape_sweep.environment import (  # noqa: E402
    environment_metadata,
)
from scripts.benchmarks.shape_sweep.runners import (  # noqa: E402
    benchmark_cupy,
    benchmark_native_cpu,
    benchmark_native_cuda,
    benchmark_numpy,
)
from scripts.benchmarks.shape_sweep.shapes import (  # noqa: E402
    PROFILES,
    SCHEMA,
    SCHEMA_VERSION,
    Shape,
    _dataset_checksum,
    _estimated_host_mib,
    make_batches,
)
from scripts.benchmarks.shape_sweep.timing import (  # noqa: E402
    _benchmark_engine,
    _calibration_run,
    _fit_batch,
    _lifecycle_metadata,
    _measure,
    _new_model,
    _restore_empty_state,
    _run_operation,
    _sample_repetitions,
)

__all__ = [
    "PROFILES",
    "PROJECT_ROOT",
    "SCHEMA",
    "SCHEMA_VERSION",
    "Shape",
    "_add_throughput",
    "_benchmark_engine",
    "_calibration_run",
    "_dataset_checksum",
    "_estimated_host_mib",
    "_fit_batch",
    "_lifecycle_metadata",
    "_measure",
    "_new_model",
    "_print_result",
    "_record_skip",
    "_restore_empty_state",
    "_run_operation",
    "_sample_repetitions",
    "benchmark_cupy",
    "benchmark_native_cpu",
    "benchmark_native_cuda",
    "benchmark_numpy",
    "environment_metadata",
    "main",
    "make_batches",
]


if __name__ == "__main__":
    raise SystemExit(main())
