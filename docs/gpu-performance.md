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

For hardware validation, the repository includes a manual-only GitHub Actions
workflow (`GPU validation`). Attach a trusted self-hosted runner with the labels
`self-hosted`, `windows`, `x64`, and `gpu`; it never runs on pull-request events.
