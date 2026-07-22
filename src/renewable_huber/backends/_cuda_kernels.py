"""Lazily compiled CUDA C++ kernels used by :mod:`cupy_backend`.

CuPy already routes dense matrix products and linear solves to cuBLAS and
cuSOLVER.  These kernels focus on the small, branch-heavy vector operations
around them, where a generic Array API expression otherwise launches several
kernels and materialises temporary masks.
"""

from __future__ import annotations

from typing import Any

import numpy as np

_KERNEL_NAMES = (
    "huber_loss_f32",
    "huber_loss_f64",
    "smoothed_score_f32",
    "smoothed_score_f64",
    "smoothed_curvature_f32",
    "smoothed_curvature_f64",
    "smoothed_terms_f32",
    "smoothed_terms_f64",
)


_CUDA_SOURCE = r"""
template <typename T>
__device__ __forceinline__ void smoothed_terms(
    const T residual, const T tau, const T bandwidth, T* score, T* curvature
) {
    const T negative_left = -tau - bandwidth;
    const T negative_right = -tau + bandwidth;
    const T positive_left = tau - bandwidth;
    const T positive_right = tau + bandwidth;

    if (residual < negative_left) {
        *score = -tau;
        *curvature = T(0);
    } else if (residual <= negative_right) {
        *score = T(0.5) * (residual - tau + bandwidth);
        *curvature = T(0.5);
    } else if (residual < positive_left) {
        *score = residual;
        *curvature = T(1);
    } else if (residual <= positive_right) {
        *score = T(0.5) * (residual + tau - bandwidth);
        *curvature = T(0.5);
    } else {
        *score = tau;
        *curvature = T(0);
    }
}

template <typename T>
__device__ __forceinline__ T huber_loss_value(const T residual, const T tau) {
    const T absolute = residual < T(0) ? -residual : residual;
    return absolute <= tau ? T(0.5) * residual * residual : tau * absolute - T(0.5) * tau * tau;
}

template <typename T>
__device__ __forceinline__ T smoothed_score_value(
    const T residual, const T tau, const T bandwidth
) {
    T score;
    T curvature;
    smoothed_terms(residual, tau, bandwidth, &score, &curvature);
    return score;
}

template <typename T>
__device__ __forceinline__ T smoothed_curvature_value(
    const T residual, const T tau, const T bandwidth
) {
    T score;
    T curvature;
    smoothed_terms(residual, tau, bandwidth, &score, &curvature);
    return curvature;
}

extern "C" __global__ void smoothed_score_f32(
    const float* residual,
    float* score,
    const long long size,
    const float tau,
    const float bandwidth
) {
    const long long index = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < size) {
        score[index] = smoothed_score_value(residual[index], tau, bandwidth);
    }
}

extern "C" __global__ void huber_loss_f32(
    const float* residual, float* loss, const long long size, const float tau
) {
    const long long index = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < size) {
        loss[index] = huber_loss_value(residual[index], tau);
    }
}

extern "C" __global__ void huber_loss_f64(
    const double* residual, double* loss, const long long size, const double tau
) {
    const long long index = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < size) {
        loss[index] = huber_loss_value(residual[index], tau);
    }
}

extern "C" __global__ void smoothed_score_f64(
    const double* residual,
    double* score,
    const long long size,
    const double tau,
    const double bandwidth
) {
    const long long index = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < size) {
        score[index] = smoothed_score_value(residual[index], tau, bandwidth);
    }
}

extern "C" __global__ void smoothed_curvature_f32(
    const float* residual,
    float* curvature,
    const long long size,
    const float tau,
    const float bandwidth
) {
    const long long index = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < size) {
        curvature[index] = smoothed_curvature_value(residual[index], tau, bandwidth);
    }
}

extern "C" __global__ void smoothed_curvature_f64(
    const double* residual,
    double* curvature,
    const long long size,
    const double tau,
    const double bandwidth
) {
    const long long index = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < size) {
        curvature[index] = smoothed_curvature_value(residual[index], tau, bandwidth);
    }
}

extern "C" __global__ void smoothed_terms_f32(
    const float* residual,
    float* score,
    float* curvature,
    const long long size,
    const float tau,
    const float bandwidth
) {
    const long long index = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < size) {
        smoothed_terms(residual[index], tau, bandwidth, &score[index], &curvature[index]);
    }
}

extern "C" __global__ void smoothed_terms_f64(
    const double* residual,
    double* score,
    double* curvature,
    const long long size,
    const double tau,
    const double bandwidth
) {
    const long long index = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < size) {
        smoothed_terms(residual[index], tau, bandwidth, &score[index], &curvature[index]);
    }
}

"""


class CudaKernels:
    """Compiled CUDA C++ fast path for contiguous float32/float64 CuPy arrays."""

    _threads_per_block = 256

    def __init__(self, cp: Any) -> None:
        self._cp = cp
        self._module = cp.RawModule(
            code=_CUDA_SOURCE,
            options=("--std=c++14",),
            name_expressions=_KERNEL_NAMES,
        )
        self._functions = {name: self._module.get_function(name) for name in _KERNEL_NAMES}

    def huber_loss(self, residual: Any, tau: float) -> Any | None:
        """Return a Huber-loss vector with one CUDA C++ branch kernel."""

        specification = self._loss_specification(residual, tau)
        if specification is None:
            return None
        suffix, size, tau_value = specification
        loss = self._cp.empty_like(residual)
        blocks = (size + self._threads_per_block - 1) // self._threads_per_block
        self._functions[f"huber_loss_{suffix}"](
            (blocks,),
            (self._threads_per_block,),
            (residual, loss, np.int64(size), tau_value),
        )
        return loss

    def smoothed_score(self, residual: Any, tau: float, bandwidth: float) -> Any | None:
        """Return one fused score array, or ``None`` for unsupported inputs."""

        specification = self._vector_specification(residual, tau, bandwidth)
        if specification is None:
            return None
        suffix, size, tau_value, bandwidth_value = specification
        score = self._cp.empty_like(residual)
        self._launch_vector(
            f"smoothed_score_{suffix}", residual, score, size, tau_value, bandwidth_value
        )
        return score

    def smoothed_curvature(self, residual: Any, tau: float, bandwidth: float) -> Any | None:
        """Return one fused curvature array, or ``None`` for unsupported inputs."""

        specification = self._vector_specification(residual, tau, bandwidth)
        if specification is None:
            return None
        suffix, size, tau_value, bandwidth_value = specification
        curvature = self._cp.empty_like(residual)
        self._launch_vector(
            f"smoothed_curvature_{suffix}", residual, curvature, size, tau_value, bandwidth_value
        )
        return curvature

    def smoothed_terms(self, residual: Any, tau: float, bandwidth: float) -> tuple[Any, Any] | None:
        """Return score and curvature in one CUDA C++ launch when possible."""

        specification = self._vector_specification(residual, tau, bandwidth)
        if specification is None:
            return None
        suffix, size, tau_value, bandwidth_value = specification
        score = self._cp.empty_like(residual)
        curvature = self._cp.empty_like(residual)
        blocks = (size + self._threads_per_block - 1) // self._threads_per_block
        self._functions[f"smoothed_terms_{suffix}"](
            (blocks,),
            (self._threads_per_block,),
            (residual, score, curvature, np.int64(size), tau_value, bandwidth_value),
        )
        return score, curvature

    def _vector_specification(
        self, residual: Any, tau: float, bandwidth: float
    ) -> tuple[str, int, Any, Any] | None:
        if residual.ndim != 1 or not self._is_c_contiguous(residual):
            return None
        suffix = self._dtype_suffix(residual)
        if suffix is None:
            return None
        scalar_type = np.float32 if suffix == "f32" else np.float64
        h = min(bandwidth, tau * 0.5)
        return suffix, int(residual.size), scalar_type(tau), scalar_type(h)

    def _loss_specification(self, residual: Any, tau: float) -> tuple[str, int, Any] | None:
        if residual.ndim != 1 or not self._is_c_contiguous(residual):
            return None
        suffix = self._dtype_suffix(residual)
        if suffix is None:
            return None
        scalar_type = np.float32 if suffix == "f32" else np.float64
        return suffix, int(residual.size), scalar_type(tau)

    def _launch_vector(
        self, name: str, residual: Any, output: Any, size: int, tau: Any, bandwidth: Any
    ) -> None:
        blocks = (size + self._threads_per_block - 1) // self._threads_per_block
        self._functions[name](
            (blocks,),
            (self._threads_per_block,),
            (residual, output, np.int64(size), tau, bandwidth),
        )

    def _dtype_suffix(self, array: Any) -> str | None:
        if array.dtype == self._cp.dtype("float32"):
            return "f32"
        if array.dtype == self._cp.dtype("float64"):
            return "f64"
        return None

    @staticmethod
    def _is_c_contiguous(array: Any) -> bool:
        return bool(array.flags.c_contiguous)
