"""Minimal backend contract used by the portable numerical core."""

from __future__ import annotations

from typing import Any, Protocol


class ArrayBackend(Protocol):
    """Adapter over an eager array namespace such as NumPy or CuPy.

    The core deliberately uses only an Array-API-shaped ``xp`` namespace plus
    these three conversion methods.  GPU implementations can therefore retain
    state on-device instead of copying a batch through NumPy.
    """

    name: str
    device: str
    dtype: Any
    xp: Any

    def asarray(self, value: Any) -> Any: ...

    def to_numpy(self, value: Any) -> Any: ...

    def scalar(self, value: Any) -> float: ...

    def solve(self, matrix: Any, vector: Any) -> Any: ...

    def norm(self, value: Any) -> float: ...

    def is_finite(self, value: Any) -> bool: ...

    def synchronize(self) -> None: ...
