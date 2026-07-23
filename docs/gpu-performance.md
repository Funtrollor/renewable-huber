# CUDA performance path

`backend="cupy"` keeps arrays, model coefficients, and the renewable information
matrix on the active CUDA device. The default numerical contract remains strict
`float32` or `float64`; this path does not silently enable reduced-precision
Tensor Core math.

The CUDA optimisation is loaded lazily through CuPy's NVRTC `RawModule`:

- CUDA C++ fuses the branch-heavy Huber loss and the smoothed score/curvature
  calculation into one device kernel per operation.
- CuPy continues to use cuBLAS for GEMV/GEMM and cuSOLVER for linear solves,
  which is faster and more portable than replacing dense linear algebra with
  handwritten kernels.
- Newton updates reuse a single `X * curvature[:, None]` workspace during a
  batch instead of allocating it for every Hessian evaluation.
- If NVRTC compilation is unavailable, the backend automatically falls back to
  the existing generic CuPy expressions without changing results.

Run the device-only microbenchmark after installing the CuPy extra:

```powershell
python scripts/benchmarks/benchmark_cuda_kernels.py --samples 1000000 --dtype float32
```

It uses CUDA events, warms up the device, and reports the generic CuPy versus
fused CUDA C++ timings. It is deliberately a kernel benchmark rather than an
end-to-end throughput promise: final throughput also depends on batch shape,
the number of Newton or LAMM iterations, host-to-device transfer, and the
linear algebra workload.

Run the repeatable end-to-end comparison separately:

```powershell
python scripts/benchmarks/benchmark_numpy_cupy.py `
  --samples 100000 --features 90 --batch-size 32768 `
  --dtype float32 --repeats 5 --seed 42 `
  --output benchmark.json
```

The JSON record contains the exact shape, dtype, seed, individual timings,
median/best throughput, Python/NumPy/CuPy/CUDA versions, CPU platform, and GPU
name. It reports two CUDA paths:

- `cupy_cuda_host_input` includes conversion and host-to-device transfer for
  every submitted batch.
- `cupy_cuda_device_input` preloads batches before timing and measures the
  intended long-running, device-resident path.

The difference is reported as transfer/conversion overhead. Compare JSON
records only when batch shape, dtype, solver settings, and hardware metadata
are compatible. Scalar convergence tests still cross the device/host
synchronisation boundary on each solver iteration; use Nsight Systems or
CuPy's profiler around this benchmark before replacing convergence logic with
a device-side implementation.

For hardware validation, the repository includes a manual-only GitHub Actions
workflow (`GPU validation`). Attach a trusted self-hosted runner with the labels
`self-hosted`, `windows`, `x64`, and `gpu`; it never runs on pull-request events.
