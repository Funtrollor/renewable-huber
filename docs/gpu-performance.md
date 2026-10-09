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
repository for review:

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

## P7: native CUDA line-search rounds

An Nsight breakdown of the ABI 2 engine on the fixed host found launch and
synchronization latency dominating every `float32` configuration, with the
line search evaluating 1.9–3.0 candidates per Newton iteration at about 21
API calls each. The engine now evaluates up to four candidates per fused
round with one synchronization. Against the 0.7.0-era `main` it is 1.58x
faster in the median cold case and 1.69x in the median steady case,
no case is slower, and native CUDA stays faster than CuPy everywhere. The
four `float32` L1 cases change iteration counts by more than one, so the
strict A/B gates do not pass; with the iteration limit relaxed to 11 on the
maintainer's instruction, cold passes and steady fails only three
relative-MAD cases; [native-core P2](native-core-p2.md#p7-fixed-host-breakdown-c-abi-2--api-4)
has the breakdown, the per-iteration times and the rejected alternatives.
The A/B records are
`benchmarks/baselines/p7-windows-rtx5070ti-native-cuda-ab-{cold,steady}-*.json`.

## N5 fixed-host results for native CUDA

All N5 records were captured on the Windows host with an RTX 5070 Ti
(SM 12.0), driver 616.64, CUDA 12.9, CPython 3.11.0 and CuPy 14.2.0. The
[native penalty completion plan](native-penalty-completion-plan.md#n5-fixed-host-results)
has the full results, the environment and every record's SHA-256.

### `penalty="none"`: no regression across the ABI 2 change

The [GPU host runbook](gpu-host-runbook.md) §3a interleaved A/B compared
`fca7b83` (C ABI 1 / Python API 3) with `10fc363` (ABI 2 / API 4) on
2026-09-30. It used `--backend gpu --rounds 9 --allow-native-version-change
--freeze-sample-repetitions`, once cold and once steady. Both gates passed:

- all 16 native cases in each lifecycle met every criterion;
- relative MAD was at most 8.3% (cold) and 8.5% (steady);
- median iterations were identical;
- paired candidate/baseline medians were 0.937–1.056 (cold) and 0.925–1.038
  (steady);
- native/CuPy was at most 0.862.

Every difference is inside ±10%, so there is no measurable difference. The
accepted records are committed as
`benchmarks/baselines/p5-windows-rtx5070ti-native-cuda-none-ab-{cold,steady}-{baseline,candidate,gate,sample-repetitions-plan}.json`;
for example, the
[cold gate report](https://github.com/Funtrollor/renewable-huber/blob/main/benchmarks/baselines/p5-windows-rtx5070ti-native-cuda-none-ab-cold-gate.json)
and the
[steady gate report](https://github.com/Funtrollor/renewable-huber/blob/main/benchmarks/baselines/p5-windows-rtx5070ti-native-cuda-none-ab-steady-gate.json).

### L1: native CUDA against CuPy

The runbook §3b command ran three times on 2026-09-29 at `4114918`: standard
profile, `--backend gpu --penalty l1 --dtype both --lifecycle both
--operation both --warmup 3 --repeats 9`.

Each native CUDA case is compared with the CuPy case of the same shape,
dtype, lifecycle, operation and input transport, using the ratio of the
median times, native over CuPy:

- 46 of the 48 comparisons are below 0.90 in all three runs, a repeatable
  native advantage. The per-run ratios span 0.23–0.89, with a median of 0.34.
- 2 comparisons show no measurable difference: host-input streaming cold
  `fit`, float32 (1.055–1.071) and float64 (0.902–0.978).
- None is repeatably slower.

Every case converged, and the largest native relative MAD was 9.4%. The
float32 `partial_fit` iteration counts differ between the engines (the
float64 counts match). The plan shows this is float32 rounding near
`tol=1e-6`, not a defect.

The three runs are the fixed-host L1 baseline:
[run 1](https://github.com/Funtrollor/renewable-huber/blob/main/benchmarks/baselines/p5-windows-rtx5070ti-native-cuda-l1-run1.json),
[run 2](https://github.com/Funtrollor/renewable-huber/blob/main/benchmarks/baselines/p5-windows-rtx5070ti-native-cuda-l1-run2.json),
[run 3](https://github.com/Funtrollor/renewable-huber/blob/main/benchmarks/baselines/p5-windows-rtx5070ti-native-cuda-l1-run3.json).
`validate_record` accepts them, because they record native CUDA ABI 2 with
`l1` in `supported_penalties`.

### Selection

Native CUDA, L1 included, remains explicit opt-in; `backend="auto"` never
selects it. The plan gives the reasoning.

Like the P3 baselines, accepted fixed-host records are committed under
`benchmarks/baselines/`. Everything else, including the A/B round and
calibration files and the superseded attempts, stays outside Git as described
above. See also the [native performance policy](native-performance-policy.md).