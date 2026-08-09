# Native-core P0 baseline

> Historical pre-native capture only. These measurements do not describe the
> 0.6.1 release and must not be used as current performance headlines.

This document records the pre-native correctness and performance baseline. It
is a comparison point for the Rust CPU and CUDA engines, not a general
throughput promise.

## Captured environment

The baseline was captured on 2026-07-28 from commit
`fffffabef278ac1d9162141151f98da960f194c4` plus the P0 measurement scripts.

| Component | Value |
|---|---|
| OS | Windows 10.0.26200 x86-64 |
| CPU | AMD Ryzen 9 9900X 12-Core Processor |
| GPU | NVIDIA GeForce RTX 5070 Ti, 16,303 MiB |
| GPU compute capability | 12.0 |
| NVIDIA driver | 596.49 |
| CUDA toolkit/runtime | 12.9 |
| Python | 3.11.0 |
| NumPy | 2.4.6 |
| CuPy | 14.1.1 |
| Nsight Systems | 2025.3.2 |
| Nsight Compute | 2025.2.0; CLI capture unavailable, noted below |

Raw, schema-versioned summaries are committed as:

- `benchmarks/baselines/p0-windows-rtx5070ti-shape-sweep.json`
- `benchmarks/baselines/p0-windows-rtx5070ti-nsys-summary.json`

Binary `.nsys-rep`, `.sqlite`, and future `.ncu-rep` reports remain under the
ignored `artifacts/` directory.

## Shape sweep

The complete sweep measures both penalties, both supported dtypes, and NumPy,
host-fed CUDA, and device-resident CUDA. Each number below is the median of
three warm runs. Data generation and device preload are outside the
device-resident timing.

Representative unpenalized `float32` results:

| Shape | NumPy CPU | CUDA host input | CUDA device input | Device/CPU |
|---|---:|---:|---:|---:|
| latency: 4,096 x 16 | 3.71 M samples/s | 0.44 M | 0.46 M | 0.12x |
| reference: 100,000 x 90 | 1.56 M samples/s | 4.10 M | 4.57 M | 2.93x |
| wide: 16,384 x 256 | 21.8 K samples/s | 397.7 K | 406.8 K | 18.64x |
| streaming: 1,000,000 x 32 | 4.98 M samples/s | 13.71 M | 15.99 M | 3.21x |

The result establishes three dispatch requirements:

1. Small problems belong on CPU because CUDA launch and synchronization costs
   dominate.
2. The unpenalized weighted-Gram path benefits strongly from CUDA as feature
   count grows.
3. L1 does not consistently benefit from CUDA yet. For example, reference
   `float32` L1 measured 2.31 M samples/s on CPU and 1.01 M samples/s with
   device-resident CUDA. The native CUDA loop must reduce launch and scalar
   synchronization overhead before becoming the default L1 engine.

Reproduce the complete sweep:

```powershell
python scripts/benchmarks/benchmark_shape_sweep.py `
  --profile standard --backend both --penalty both --dtype both `
  --warmup 1 --repeats 3 `
  --output artifacts/p0-shape-sweep.json
```

Use `--profile smoke` before longer runs. A named standard case can be selected
with `--case latency`, `--case reference`, `--case wide`, or
`--case streaming`. Each named case has a stable derived seed whether it is run
alone or as part of the full profile.

## Nsight Systems baseline

The timeline workload used two device-resident batches with 8,192 total rows,
32 features, unpenalized `float32`, three measured repetitions, and eight
solver iterations per repetition.

Within the three `profile/repeat-*` NVTX ranges, Nsight recorded:

| Metric | Count or duration | Per solver iteration |
|---|---:|---:|
| CUDA kernels | 2,067 | 86.1 |
| Device-to-host copies | 168 copies / 492 bytes | 7.0 copies |
| Stream wait synchronizations | 174 | 7.25 |
| CUDA kernel execution | 3.28 ms | 0.137 ms |
| Profile range window | 49.63 ms | 2.07 ms |
| `cudaMemcpyAsync` host API time | 14.29 ms | 0.60 ms |

Only 492 bytes moved from device to host, but those scalar copies created 168
separate transfers. This confirms that transfer volume is not the problem;
frequency and synchronization are. Kernel execution occupied a small fraction
of the profiled wall-time window, while scalar copies, launch dispatch, and
stream waits dominated host-side time.

The most expensive individual kernel family in this small workload was the
cuSOLVER LU factorization path. The trace contains one factorization per solver
iteration. P1/P2 should therefore:

- move the complete solver loop behind one native call;
- keep convergence and line-search scalars on device as long as possible;
- cache handles and workspaces;
- use a Cholesky fast path for positive-definite Hessians;
- retain a tested QR/least-squares fallback;
- continue using vendor GEMV/GEMM instead of custom matrix multiplication.

Capture and summarize a new Systems report:

```powershell
.\scripts\profiling\run_nsight_systems.ps1 `
  -Python python `
  -Samples 100000 -Features 90 -BatchSize 32768
```

The wrapper finds an installed Windows Nsight Systems CLI even when `nsys` is
not on `PATH`. It emits the binary report, SQLite export, workload metadata,
and a range-filtered JSON summary under `artifacts/nsight/`.

The summary can also be regenerated independently:

```powershell
python scripts/profiling/summarize_nsys_sqlite.py `
  artifacts/nsight/native-core-p0-systems.sqlite `
  --metadata artifacts/nsight/native-core-p0-systems.json `
  --output artifacts/nsight/native-core-p0-summary.json
```

## Nsight Compute status

The P0 repository includes `run_nsight_compute.ps1` with a bounded launch count
and the same NVTX-instrumented workload. On the capture machine, Nsight Compute
2025.2.0 printed its version successfully but its CLI exited with Windows code
`-1073740791` even for `--list-sets`; no trustworthy counter report was
produced. The failure occurs before the Python workload starts.

This is recorded as a tooling gap rather than fabricated occupancy or roofline
data. After repairing or upgrading the local Nsight Compute installation, run:

```powershell
.\scripts\profiling\run_nsight_compute.ps1 `
  -Python python `
  -Samples 100000 -Features 90 -BatchSize 32768 `
  -LaunchCount 50
```

Nsight Compute metrics are not a gate for starting the Rust CPU engine.
CUDA-specific kernel tuning beyond the Systems findings remains gated on a
valid Compute report.

## Correctness baseline

`tests/golden/native_core_v1.json` freezes four differential cases:

- weighted unpenalized two-batch `float64`;
- L1 two-batch `float64`;
- outlier-heavy, no-intercept `float32`;
- rank-deficient, zero-ridge least-squares fallback.

Regenerate or verify it with:

```powershell
python scripts/generate_native_golden.py --check
python -m unittest tests.test_native_golden -v
```

Golden changes require a documented algorithm correction or an RFC amendment.
Iteration counts are diagnostic; final state, portable metadata, diagnostics,
and predictions are compared with case-specific tolerances.
