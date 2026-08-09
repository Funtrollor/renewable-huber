"""Record schema, benchmark shapes and deterministic dataset generation.

Everything here is pure data: no estimator is constructed and nothing is timed,
so a shape or profile can be reviewed without reading the measurement rules.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

#: Identity of the emitted record. ``scripts/benchmarks/performance_policy.py``
#: refuses any record that does not carry exactly these two values.
SCHEMA = "renewable-huber-shape-sweep"
SCHEMA_VERSION = 2


@dataclass(frozen=True, slots=True)
class Shape:
    name: str
    samples: int
    features: int
    batch_size: int


PROFILES = {
    "smoke": (
        Shape("latency-smoke", 2_048, 8, 1_024),
        Shape("reference-smoke", 8_192, 32, 4_096),
    ),
    "standard": (
        Shape("latency", 4_096, 16, 4_096),
        Shape("reference", 100_000, 90, 32_768),
        Shape("wide", 16_384, 256, 4_096),
        Shape("streaming", 1_000_000, 32, 65_536),
    ),
}


def make_batches(
    shape: Shape,
    *,
    seed: int,
    dtype: str,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create deterministic host batches outside measured regions."""

    rng = np.random.default_rng(seed)
    array_dtype = np.dtype(dtype)
    X = rng.normal(size=(shape.samples, shape.features)).astype(array_dtype)
    coefficients = rng.normal(size=shape.features).astype(array_dtype)
    y = X @ coefficients + rng.normal(scale=0.2, size=shape.samples).astype(array_dtype)
    if shape.samples >= 100:
        outlier_rows = np.arange(0, shape.samples, max(100, shape.samples // 100))
        y[outlier_rows] += rng.normal(scale=8.0, size=outlier_rows.size).astype(array_dtype)
    return [
        (X[start : start + shape.batch_size], y[start : start + shape.batch_size])
        for start in range(0, shape.samples, shape.batch_size)
    ]


def _dataset_checksum(batches: list[tuple[np.ndarray, np.ndarray]]) -> str:
    """Fingerprint actual generated values, not only their seed and shape."""

    digest = hashlib.sha256()
    for X_batch, y_batch in batches:
        for array in (X_batch, y_batch):
            contiguous = np.ascontiguousarray(array)
            digest.update(contiguous.dtype.str.encode("ascii"))
            digest.update(repr(contiguous.shape).encode("ascii"))
            digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _estimated_host_mib(shape: Shape, dtype: str) -> float:
    itemsize = np.dtype(dtype).itemsize
    values = shape.samples * shape.features + shape.samples + shape.features
    return values * itemsize / (1024 * 1024)
