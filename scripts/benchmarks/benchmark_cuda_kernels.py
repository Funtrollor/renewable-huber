"""Measure CUDA-event timing for the CuPy and fused CUDA C++ Huber kernels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _mean_cuda_milliseconds(operation, cp, repeats: int, warmup: int = 10) -> float:
    for _ in range(warmup):
        operation()
    cp.cuda.get_current_stream().synchronize()
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()
    for _ in range(repeats):
        operation()
    end.record()
    end.synchronize()
    return cp.cuda.get_elapsed_time(start, end) / repeats


def _report(name: str, generic_ms: float, fused_ms: float) -> None:
    print(
        f"{name}: generic CuPy {generic_ms:.4f} ms | CUDA C++ {fused_ms:.4f} ms "
        f"| {generic_ms / fused_ms:.2f}x"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=1_000_000)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    args = parser.parse_args()

    try:
        import cupy as cp
    except ImportError:
        print("CuPy is required. Install the GPU extra before running this benchmark.")
        return 0

    from renewable_huber.backends.cupy_backend import CuPyBackend
    from renewable_huber.core.loss import huber_loss, smoothed_score_and_curvature

    dtype = cp.dtype(args.dtype)
    backend = CuPyBackend(dtype=args.dtype)
    if not backend.cuda_kernels_available:
        print(f"CUDA C++ kernels are unavailable: {backend.cuda_kernel_error!r}")
        return 1

    residual = cp.random.standard_normal(args.samples).astype(dtype)
    tau = 1.345
    bandwidth = 0.001

    generic_terms_ms = _mean_cuda_milliseconds(
        lambda: smoothed_score_and_curvature(residual, tau, bandwidth, cp), cp, args.repeats
    )
    fused_terms_ms = _mean_cuda_milliseconds(
        lambda: backend.cuda_smoothed_terms(residual, tau, bandwidth), cp, args.repeats
    )
    _report("Smoothed score + curvature", generic_terms_ms, fused_terms_ms)

    generic_loss_ms = _mean_cuda_milliseconds(
        lambda: huber_loss(residual, tau, cp), cp, args.repeats
    )
    fused_loss_ms = _mean_cuda_milliseconds(
        lambda: backend.cuda_huber_loss(residual, tau), cp, args.repeats
    )
    _report("Huber loss", generic_loss_ms, fused_loss_ms)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
