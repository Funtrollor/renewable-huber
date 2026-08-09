"""Backend selection and public backend protocol."""

from __future__ import annotations

from ..exceptions import BackendUnavailableError
from .numpy_backend import NumPyBackend
from .protocol import ArrayBackend


def resolve_backend(
    name: str,
    *,
    device: str = "auto",
    dtype: str = "float64",
    n_jobs: int | None = None,
    cuda_graphs: bool = False,
    cuda_fast_math: bool = False,
) -> ArrayBackend:
    """Return an installed backend for the requested stable API name.

    ``auto`` resolves deterministically to NumPy unless the caller explicitly
    requests ``device='cuda'``.  This avoids silently moving a CPU workflow to
    a GPU while still making ``backend='auto', device='cuda'`` useful.

    This function deliberately does **not** run the CPU dispatch policy.  It
    has no workload: the choice between NumPy and the native CPU engine needs
    the batch shape, which only exists after validation, so the estimator makes
    it one level up through
    :mod:`renewable_huber.backends.cpu_dispatch`.  Callers of
    ``resolve_backend`` see exactly the behaviour they always did.
    """

    if name == "auto":
        name = "cupy" if device == "cuda" else "numpy"
    if name == "numpy":
        if device == "cuda":
            raise BackendUnavailableError("backend='numpy' cannot target device='cuda'")
        return NumPyBackend(dtype)
    if name == "native_cpu":
        if device == "cuda":
            raise BackendUnavailableError("backend='native_cpu' requires device='cpu'")
        from .native_cpu_backend import NativeCpuBackend

        return NativeCpuBackend(dtype, n_threads=n_jobs)
    if name == "cupy":
        if device == "cpu":
            raise BackendUnavailableError("backend='cupy' requires a CUDA device")
        from .cupy_backend import CuPyBackend

        return CuPyBackend(dtype)
    if name == "native_cuda":
        if device == "cpu":
            raise BackendUnavailableError("backend='native_cuda' requires a CUDA device")
        from .native_cuda_backend import NativeCudaBackend

        return NativeCudaBackend(
            dtype,
            cuda_graphs=cuda_graphs,
            cuda_fast_math=cuda_fast_math,
        )
    if name == "torch":
        from .torch_backend import TorchBackend

        return TorchBackend(dtype, device)
    if name == "tensorflow":
        from .tensorflow_backend import TensorFlowBackend

        return TensorFlowBackend(dtype, device)
    raise BackendUnavailableError(f"Unknown backend: {name!r}")


__all__ = ["ArrayBackend", "NumPyBackend", "resolve_backend"]
