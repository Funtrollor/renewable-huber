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

Hardware validation is local-only and must not be dispatched through GitHub
Actions. Use the fixed GPU host with the `cuda-full` environment, record the
exact commit and dependency versions, and retain the generated JSON outside the
repository for Codex review:

```bash
bash scripts/setup-wsl-venv.sh --profile cuda-full
.venv/bin/python scripts/run_test_profile.py cuda --verbose
```

For a release candidate, the required `cuda` profile is only the first gate.
On the exact release SHA, also run CUDA C ABI CTest, clean-install/smoke all
three CPython candidate CUDA wheels, capture the schema-v2 standard shape
sweep, and run the interleaved A/B performance gate. Retain the commit SHA,
environment fingerprint, JSON output and SHA-256 outside Git. After the release
workflow builds the final wheels, repeat the smoke against those downloaded
artifacts before approving PyPI. PR and general CI workflows do not run GPU
runtime tests.

The `cuda` profile is *required*: it probes CuPy and the native CUDA extension
for a real device before loading anything and exits with status 2 when either
is absent. That is the point. The equivalent explicit module list skips itself
on a machine without a GPU and reports success, which is exactly the evidence a
local-only validation policy must not accept. Run
`python scripts/run_test_profile.py --list` to see what each profile covers.

The native-engine migration has a broader shape sweep and an NVTX-instrumented
profiling workload:

```powershell
python scripts/benchmarks/benchmark_shape_sweep.py `
  --profile standard --backend all --penalty both --dtype both `
  --lifecycle cold --operation both --warmup 3 --repeats 9 `
  --output artifacts/shape-sweep.json

.\scripts\profiling\run_nsight_systems.ps1 -Python python
```

See the [native-core RFC](native-core-rfc.md) for the accepted migration
boundary and the [P0 baseline](native-core-p0-baseline.md) for the committed
pre-native measurements and profiler findings. Fair cold/steady comparisons,
fixed-runner gates, and the calibration-only native dispatch rule are in the
[native performance policy](native-performance-policy.md).

P4 adds opt-in CUDA Graph replay and float32 TF32 tuning while preserving
strict execution by default. See [P4 native CUDA tuning](native-core-p4.md) for
the public flags, error contract, fallback rules, benchmark, and Nsight
reproduction commands.
