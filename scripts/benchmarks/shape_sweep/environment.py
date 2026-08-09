"""Comparison metadata describing the machine a record was captured on.

A record is only comparable with another when this metadata agrees, so every
probe here reports its own unavailability explicitly rather than omitting a
field: a missing GPU must read as ``gpu_unavailable``, never as silence.
"""

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path
from time import get_clock_info
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]

_THREAD_ENVIRONMENT_KEYS = (
    "MATMUL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "RAYON_NUM_THREADS",
)


def _git_revision() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip()


def environment_metadata() -> dict[str, Any]:
    """Return comparison metadata without requiring a CUDA installation."""

    timer = get_clock_info("perf_counter")
    metadata: dict[str, Any] = {
        "git_revision": _git_revision(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
        "numpy": np.__version__,
        "threading": {key: os.environ.get(key, "unset") for key in _THREAD_ENVIRONMENT_KEYS},
        "numpy_blas_provider": _numpy_blas_provider(),
        "perf_counter": {
            "implementation": timer.implementation,
            "resolution_seconds": timer.resolution,
            "monotonic": timer.monotonic,
            "adjustable": timer.adjustable,
        },
    }
    try:
        from renewable_huber import _native_cpu

        metadata["native_cpu"] = dict(_native_cpu.version())
    except Exception as error:
        metadata["native_cpu_unavailable"] = str(error)
    try:
        import cupy as cp

        properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        name = properties["name"]
        if isinstance(name, bytes):
            name = name.decode(errors="replace")
        metadata.update(
            {
                "cupy": cp.__version__,
                "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
                "gpu": name,
                "gpu_compute_capability": (f"{properties['major']}.{properties['minor']}"),
            }
        )
    except Exception as error:
        metadata["gpu_unavailable"] = str(error)
    try:
        from renewable_huber import _native_cuda

        metadata.update(
            {
                "native_cuda_abi": _native_cuda.version(),
                "native_cuda_available": bool(_native_cuda.is_available()),
            }
        )
    except (ImportError, OSError, RuntimeError) as error:
        metadata["native_cuda_unavailable"] = str(error)
    return metadata


def _numpy_blas_provider() -> dict[str, str]:
    """Return the BLAS/LAPACK identity without embedding build-machine paths."""

    config = getattr(np.__config__, "CONFIG", {})
    dependencies = config.get("Build Dependencies", {}) if isinstance(config, dict) else {}
    provider: dict[str, str] = {}
    for library in ("blas", "lapack"):
        details = dependencies.get(library, {})
        if not isinstance(details, dict):
            provider[library] = "unknown"
            continue
        provider[library] = (
            " | ".join(
                str(details[key])
                for key in ("name", "version", "openblas configuration")
                if details.get(key)
            )
            or "unknown"
        )
    return provider
