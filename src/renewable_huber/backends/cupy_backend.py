"""CuPy/CUDA backend, imported lazily so CPU users never require CUDA."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from ..exceptions import BackendUnavailableError
from ._cuda_kernels import CudaKernels


class CuPyBackend:
    """Eager CUDA backend that keeps model state and batches on one GPU.

    The constructor intentionally imports CuPy lazily.  This keeps
    ``import renewable_huber`` safe on CPU-only hosts and gives users an
    actionable error only when they explicitly select ``backend='cupy'``.
    """

    name = "cupy"

    def __init__(self, dtype: str = "float64") -> None:
        try:
            import cupy as cp
            import cupyx
        except ImportError as error:
            raise BackendUnavailableError(
                "The CuPy backend requires a CUDA-specific CuPy wheel. "
                "For CUDA 12, install: pip install 'renewable-huber[gpu-cupy]'"
            ) from error

        try:
            device_count = cp.cuda.runtime.getDeviceCount()
        except cp.cuda.runtime.CUDARuntimeError as error:
            raise BackendUnavailableError(
                "CuPy is installed but could not initialize CUDA. Check the NVIDIA driver "
                "and CUDA-compatible CuPy wheel."
            ) from error
        if device_count < 1:
            raise BackendUnavailableError("No CUDA device is available for the CuPy backend")

        self.xp = cp
        self._cupyx = cupyx
        self.dtype = cp.dtype(dtype)
        self._cuda_dll_directory = self._configure_windows_cuda_dll_path()
        self.device_id = int(cp.cuda.runtime.getDevice())
        self.device = f"cuda:{self.device_id}"
        self._cuda_kernels: CudaKernels | None = None
        self._cuda_kernels_attempted = False
        self._cuda_kernel_error: Exception | None = None
        self._validate_linear_algebra()

    def _configure_windows_cuda_dll_path(self) -> Any | None:
        """Make an installed CUDA Toolkit visible to lazy CuPy DLL imports.

        From Python 3.8 onwards, Windows no longer reliably searches ``PATH``
        for DLL dependencies.  CuPy defers loading cuBLAS until the first matrix
        operation, so register the toolkit's ``bin`` directory before that call.
        """

        if os.name != "nt" or not hasattr(os, "add_dll_directory"):
            return None
        candidates = [os.environ.get("CUDA_PATH"), self.xp.cuda.get_cuda_path()]
        if nvcc_path := shutil.which("nvcc"):
            candidates.append(str(Path(nvcc_path).resolve().parent.parent))

        # CuPy can pick the newest Toolkit on PATH even when its wheel targets
        # another CUDA major version.  Prefer a Toolkit that matches the
        # runtime version embedded in the installed CuPy wheel.
        runtime_major = self.xp.cuda.runtime.runtimeGetVersion() // 1000

        def priority(candidate: str | None) -> tuple[int, str]:
            if not candidate:
                return (2, "")
            name = Path(candidate).name.lower()
            return (0 if name.startswith(f"v{runtime_major}.") else 1, candidate)

        for cuda_path in sorted({candidate for candidate in candidates if candidate}, key=priority):
            bin_path = Path(cuda_path) / "bin"
            if bin_path.is_dir():
                return os.add_dll_directory(str(bin_path))
        return None

    def _validate_linear_algebra(self) -> None:
        """Fail early with a useful error if the CUDA runtime lacks cuBLAS."""

        try:
            with self.xp.cuda.Device(self.device_id):
                value = self.xp.asarray([1.0], dtype=self.dtype)
                _ = value @ value
                self.xp.cuda.get_current_stream(self.device_id).synchronize()
        except Exception as error:
            raise BackendUnavailableError(
                "CuPy found a CUDA device but could not execute cuBLAS. Install a matching "
                "CUDA runtime (or cupy-cuda12x[ctk]) and ensure the NVIDIA driver is available."
            ) from error

    def asarray(self, value: Any) -> Any:
        if hasattr(value, "to_numpy"):
            value = value.to_numpy()
        with self.xp.cuda.Device(self.device_id):
            return self.xp.asarray(value, dtype=self.dtype)

    def copy(self, value: Any) -> Any:
        with self.xp.cuda.Device(self.device_id):
            return value.copy()

    def reshape(self, value: Any, shape: tuple[int, ...]) -> Any:
        with self.xp.cuda.Device(self.device_id):
            return value.reshape(shape)

    def to_numpy(self, value: Any) -> Any:
        return self.xp.asnumpy(value)

    def scalar(self, value: Any) -> float:
        return float(self.xp.asnumpy(value))

    def solve(self, matrix: Any, vector: Any) -> Any:
        try:
            # CuPy otherwise warns and may return NaN for singular systems
            # instead of raising, which would bypass the least-squares path.
            with self._cupyx.errstate(linalg="raise"):
                return self.xp.linalg.solve(matrix, vector)
        except self.xp.linalg.LinAlgError:
            return self.xp.linalg.lstsq(matrix, vector, rcond=None)[0]

    def norm(self, value: Any) -> float:
        return self.scalar(self.xp.linalg.norm(value))

    def is_finite(self, value: Any) -> bool:
        return bool(self.xp.asnumpy(self.xp.all(self.xp.isfinite(value))))

    def synchronize(self) -> None:
        self.xp.cuda.get_current_stream(self.device_id).synchronize()

    @property
    def cuda_kernels_available(self) -> bool:
        """Whether the optional NVRTC-compiled CUDA C++ fast path is usable."""

        return self._get_cuda_kernels() is not None

    @property
    def cuda_kernel_error(self) -> Exception | None:
        """Compilation error retained when the backend falls back to generic CuPy."""

        self._get_cuda_kernels()
        return self._cuda_kernel_error

    def cuda_huber_loss(self, residual: Any, tau: float) -> Any | None:
        """Evaluate elementwise Huber loss with one CUDA C++ kernel."""

        kernels = self._get_cuda_kernels()
        if kernels is None:
            return None
        with self.xp.cuda.Device(self.device_id):
            return kernels.huber_loss(residual, tau)

    def cuda_smoothed_score(self, residual: Any, tau: float, bandwidth: float) -> Any | None:
        """Evaluate score with one CUDA C++ kernel when its input is supported."""

        kernels = self._get_cuda_kernels()
        if kernels is None:
            return None
        with self.xp.cuda.Device(self.device_id):
            return kernels.smoothed_score(residual, tau, bandwidth)

    def cuda_smoothed_curvature(self, residual: Any, tau: float, bandwidth: float) -> Any | None:
        """Evaluate curvature with one CUDA C++ kernel when its input is supported."""

        kernels = self._get_cuda_kernels()
        if kernels is None:
            return None
        with self.xp.cuda.Device(self.device_id):
            return kernels.smoothed_curvature(residual, tau, bandwidth)

    def cuda_smoothed_terms(
        self, residual: Any, tau: float, bandwidth: float
    ) -> tuple[Any, Any] | None:
        """Evaluate score and curvature together in one CUDA C++ kernel."""

        kernels = self._get_cuda_kernels()
        if kernels is None:
            return None
        with self.xp.cuda.Device(self.device_id):
            return kernels.smoothed_terms(residual, tau, bandwidth)

    def cuda_huber_score_and_smoothed_curvature(
        self, residual: Any, tau: float, bandwidth: float
    ) -> tuple[Any, Any] | None:
        """Evaluate the current Huber score and smoothed curvature in one launch."""

        kernels = self._get_cuda_kernels()
        if kernels is None:
            return None
        with self.xp.cuda.Device(self.device_id):
            return kernels.huber_score_and_smoothed_curvature(residual, tau, bandwidth)

    def _get_cuda_kernels(self) -> CudaKernels | None:
        """Compile the internal CUDA C++ module once and retain a safe fallback."""

        if self._cuda_kernels_attempted:
            return self._cuda_kernels
        self._cuda_kernels_attempted = True
        try:
            with self.xp.cuda.Device(self.device_id):
                self._cuda_kernels = CudaKernels(self.xp)
        except Exception as error:
            # The generic CuPy expressions remain a correct route on unusual
            # drivers, CUDA versions, or environments without NVRTC.
            self._cuda_kernel_error = error
        return self._cuda_kernels
