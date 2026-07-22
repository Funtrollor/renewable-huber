"""Backend selection and public backend protocol."""

from __future__ import annotations

from ..exceptions import BackendUnavailableError
from .numpy_backend import NumPyBackend
from .protocol import ArrayBackend


def resolve_backend(name: str, *, device: str = "auto", dtype: str = "float64") -> ArrayBackend:
    """Return an installed backend for the requested stable API name.

    ``auto`` resolves deterministically to NumPy unless the caller explicitly
    requests ``device='cuda'``.  This avoids silently moving a CPU workflow to
    a GPU while still making ``backend='auto', device='cuda'`` useful.
    """

    if name == "auto":
        name = "cupy" if device == "cuda" else "numpy"
    if name == "numpy":
        if device == "cuda":
            raise BackendUnavailableError("backend='numpy' cannot target device='cuda'")
        return NumPyBackend(dtype)
    if name == "cupy":
        if device == "cpu":
            raise BackendUnavailableError("backend='cupy' requires a CUDA device")
        from .cupy_backend import CuPyBackend

        return CuPyBackend(dtype)
    if name in {"torch", "tensorflow"}:
        raise BackendUnavailableError(
            f"The '{name}' backend is part of the public API but is not implemented yet. "
            "Use backend='numpy' for the reference implementation."
        )
    raise BackendUnavailableError(f"Unknown backend: {name!r}")


__all__ = ["ArrayBackend", "NumPyBackend", "resolve_backend"]
