"""Reference backend backed by NumPy and its linked BLAS/LAPACK library."""

from __future__ import annotations

from typing import Any

import numpy as np


class NumPyBackend:
    """The stable reference implementation for CPU calculations."""

    name = "numpy"
    device = "cpu"
    xp = np

    def __init__(self, dtype: str = "float64") -> None:
        self.dtype = np.dtype(dtype)

    def asarray(self, value: Any) -> np.ndarray:
        if hasattr(value, "to_numpy"):
            value = value.to_numpy()
        return np.asarray(value, dtype=self.dtype)

    def to_numpy(self, value: Any) -> np.ndarray:
        return np.asarray(value)

    def scalar(self, value: Any) -> float:
        return float(np.asarray(value))

    def solve(self, matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
        try:
            return np.linalg.solve(matrix, vector)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(matrix, vector, rcond=None)[0]

    def norm(self, value: np.ndarray) -> float:
        return float(np.linalg.norm(value))

    def is_finite(self, value: np.ndarray) -> bool:
        return bool(np.isfinite(value).all())

    def synchronize(self) -> None:
        """NumPy is eager and has no device queue to synchronize."""
