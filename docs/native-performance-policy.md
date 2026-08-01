# Native performance measurement and dispatch policy

This policy prevents an attractive native number from being compared with a
different lifecycle, input transport, or solver operation. It applies to the
Rust CPU engine, the Rust/CUDA P2 solver, and its P3 DLPack transport extension
while both native engines remain opt-in.

`RenewableHuberRegressor(backend="auto")` is intentionally unchanged. The
tools below are an offline, auditable promotion mechanism; they do not silently
route users to an experimental extension.

## Measurement contract

Run `scripts/benchmarks/benchmark_shape_sweep.py` with schema version 2. A
case is comparable only when all of these agree:

- shape, seed, SHA-256 dataset fingerprint, dtype, penalty, `max_iter`, and
  `tol`;
- operation (`fit` or `partial_fit`);
- lifecycle (`cold` or `steady`);
- input location and transfer policy;
- initialization and state-reset timing policy;
- fixed CPU/GPU runner, thread settings, BLAS/LAPACK provider, and relevant
  Python, NumPy, CUDA, and CuPy versions.

The harness records every raw duration, median solver iterations, convergence,
and the timing policy in each result, including
`includes_engine_destruction=false`. Destruction is excluded because it is not
part of the public `fit`/`partial_fit` call and fitted models normally remain
alive for prediction. Early schema-v2 records omitted this field and
accidentally included temporary-model teardown; the policy parser treats an
omitted value as `true`, preventing comparison with corrected captures. It
records unsupported combinations in `skipped`; it must never omit them
silently.

On Windows, one 5--15 ms CUDA call is too short to be a reliable statistical
sample under WDDM scheduling. The harness therefore targets 100 ms of timed
work per sample by default (`--minimum-sample-seconds 0.1`). It calibrates a
fixed repetition count from the explicit warmups, then reports the arithmetic
mean of that many independent public operations. Every cold repetition still
constructs a new estimator and native engine; durations are not trimmed,
filtered, or selectively retried. Cyclic Python GC is collected before each
sample and disabled during its timed intervals, while reference-counted object
destruction remains normal and stays outside each operation interval. The JSON
records the timer, aggregation, actual repetitions, calibration runs, target
duration, and GC policy. Sampling and GC policy are part of the comparison key,
so older single-call schema-v2 captures must be recaptured rather than silently
mixed with stabilized measurements. The 10% CUDA relative-MAD ceiling is not
relaxed.

| Contract | Public work measured | Included in timer | Excluded from timer | Valid comparisons |
| --- | --- | --- | --- | --- |
| `cold` + `fit` | one full-data `fit` | estimator/backend/native-engine construction, input transfer | data generation, CUDA context creation, device preload, fitted-model destruction | same operation, host/device transport, and engine lifecycle |
| `cold` + `partial_fit` | configured stream of batches | new estimator/backend/native-engine and host-to-device transfer when applicable | data generation, CUDA context creation, device preload, fitted-model destruction | end-to-end host paths; separate CuPy and native CUDA device-input results |
| `steady` + `partial_fit` | configured stream of batches | batch computation and per-batch host-to-device transfer | one-time prime, workspace allocation, and empty-state restore | all engines use the same reusable-model reset policy |
| `steady` + `fit` | not measured | not applicable | not applicable | `fit()` calls `reset()`, so retaining an engine would not represent public semantics |

For CuPy and native CUDA, the device-input records preload the same CuPy arrays
before timing and say `input_location="device"`. Native CUDA consumes those
arrays through DLPack on its private stream and records
`includes_input_transfer=false`; its internal device-to-device workspace copy
when required, direct intercept expansion, and all solver work remain part of
the timed operation. Device and host records are not interchangeable. Native
CUDA still emits an explicit L1 skip.

```powershell
# Fair cold end-to-end stream comparison.
python scripts/benchmarks/benchmark_shape_sweep.py `
  --profile standard --backend all --penalty both --dtype both `
  --lifecycle cold --operation partial-fit --warmup 3 --repeats 9 `
  --output artifacts/shape-cold-v2.json

# Reusable-engine streaming throughput. Do not compare it with a cold result.
python scripts/benchmarks/benchmark_shape_sweep.py `
  --profile standard --backend all --penalty none --dtype both `
  --lifecycle steady --operation partial-fit --warmup 3 --repeats 9 `
  --output artifacts/shape-steady-v2.json
```

The old P1/P2 JSON files are schema v1 historical observations. In particular,
the P2 native CUDA result was steady while NumPy/CuPy constructed a new
estimator per repeat. They remain useful for investigation but cannot be used
as a schema-v2 pass/fail baseline.

The approved CPU schema-v2 baseline is
[`p3-windows-ryzen9900x-native-cpu-v2.json`](../benchmarks/baselines/p3-windows-ryzen9900x-native-cpu-v2.json).
Its strict native/reference gate passes all 32 standard combinations across
shape, penalty, dtype, and public operation using 0.25-second samples. The
approved CUDA baseline is
[`p3-windows-rtx5070ti-native-cuda-v2.json`](../benchmarks/baselines/p3-windows-rtx5070ti-native-cuda-v2.json).
It passes all 16 host-input and all 16 device-input CuPy comparisons using
0.5-second samples. Both records use three warmups and nine measured samples;
a result captured while another workload is active must not be promoted.

## Fixed-runner regression gate

Capture a v2 baseline and candidate on the same self-hosted runner. Use at
least three warmups and nine measured samples. With the default stabilized
protocol, a sample may aggregate several independent operations while retaining
per-operation units. Then run:

```powershell
python scripts/benchmarks/check_performance_regression.py `
  --baseline benchmarks/baselines/native-v2-fixed-runner.json `
  --candidate artifacts/native-v2-candidate.json `
  --output artifacts/native-v2-gate.json
```

| Engine class | Maximum median slowdown | Maximum relative MAD | Other requirements |
| --- | ---: | ---: | --- |
| Rust native CPU | 1.10x | 5% | at least 9 repeats, convergence, same runner/thread/BLAS fingerprint, median iterations within 1, and no slower than matched NumPy |
| Native CUDA | 1.15x | 10% | at least 9 repeats, convergence, same GPU/runtime fingerprint, median iterations within 1, and no slower than matched CuPy under the same host/device transport |

The checker gates `rust_native_cpu`, `native_cuda_host_input`, and
`native_cuda_device_input` by default.
Reference results remain in the record for diagnosis. A difference in GPU,
driver, toolkit, CPU, Python, NumPy, thread environment, or BLAS/LAPACK
provider makes a fixed-runner gate fail;
`--allow-different-hardware` is reporting-only and must not approve a
regression.

The paired-reference parity check is also on by default. It compares native
CPU against `numpy_cpu`, host-fed native CUDA against `cupy_cuda_host_input`,
and DLPack native CUDA against `cupy_cuda_device_input`, with the same
checksum, solver settings, lifecycle, initialization, transfer, and state-reset
policy. A diagnostic run may pass
`--no-require-competitor-parity`, but that result cannot justify dispatch
promotion.

The thresholds are guardrails, not statistical proof of a speedup. Repeated
noisy measurements should be recaptured rather than accepted by increasing a
tolerance. Correctness still comes first: the golden/differential suite must
pass independently of the timing gate.

## Native CPU thread scaling

Use the dedicated scaling harness to compare per-estimator `n_jobs` settings
without mutating `RAYON_NUM_THREADS` or another process-global environment
variable:

```powershell
python scripts/benchmarks/benchmark_native_cpu_scaling.py `
  --profile standard --case reference --dtype float64 --penalty none `
  --lifecycle steady --operation partial-fit --n-jobs 1,2,4,8,-1 `
  --warmup 3 --repeats 9 --minimum-sample-seconds 0.1 `
  --output artifacts/native-cpu-reference-f64-thread-scaling.json
```

The dataset is generated once and fingerprinted, then every thread setting
uses the shape-sweep measurement discipline: explicit warmups, one unreported
calibration operation, a fixed-size timing block, cyclic-GC control, raw
per-sample seconds, and the same cold/steady reset rules. The JSON records both
`requested_n_jobs` and `effective_threads`; `RenewableHuberRegressor.n_jobs_`
is the authoritative effective count after the first operation. A fallback
source is labeled explicitly for compatibility with an older extension and
must not be used for a release claim.

Every case reports `median_seconds`, `speedup_vs_n_jobs_1`, and
`parallel_efficiency`. The one-thread case is mandatory and always supplies
the baseline. `n_jobs=-1` is measured as its own configuration; do not replace
it with the fastest observed fixed count or compare records with different
dataset, lifecycle, operation, dtype, penalty, solver, or sampling contracts.

## Calibration-driven shape-aware dispatch

The conservative dispatch advisor reads one v2 calibration record and makes a
decision for one *exact* `(samples, features, batch_size, dtype, penalty,
input location, lifecycle, operation, max_iter, tol)` tuple. All selected
engines must also come from one dataset/checksum contract, and unconverged
measurements are never eligible:

```powershell
python scripts/benchmarks/dispatch_policy.py `
  --calibration artifacts/shape-steady-v2.json `
  --samples 16384 --features 256 --batch-size 4096 `
  --dtype float32 --penalty none --lifecycle steady --operation partial_fit `
  --max-iter 100 --tol 1e-6 `
  --cupy-available --native-cpu-available --native-cuda-available
```

| Runtime condition | Recommendation |
| --- | --- |
| Device input | Fastest exactly calibrated CuPy/native-CUDA result; native promotion requires the configured speedup margin |
| Host input, no exact calibration | NumPy; do not extrapolate a native win to a new shape |
| Host input, L1 | fastest calibrated NumPy/CuPy reference; native CUDA is ineligible |
| Host input, native CPU/CUDA has an exact result at least 10% faster than the fastest available reference | that native backend |
| Host input, native result is within 10% of or slower than the reference | NumPy or CuPy reference |

The 10% selection margin avoids flip-flopping on measurement noise and is
stricter than merely choosing the smallest one-off median. Capabilities are
inputs to the advisor: a future runtime integration must first verify that the
requested extension can be loaded. Missing calibration always prefers the
portable path.

Before integrating this advisor into a public `native_auto` selector, expand
the calibration grid beyond the four named shapes, persist the approved policy
with its hardware fingerprint, and add telemetry-free unit tests for every
fallback. `backend="auto"` must remain stable until that migration is accepted
as a separate API decision.
