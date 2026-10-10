# Native performance measurement and dispatch policy

This policy prevents an attractive native number from being compared with a
different lifecycle, input transport, or solver operation. It applies to the
Rust CPU engine, the Rust/CUDA P2 solver, and its P3 DLPack transport extension.
Native CUDA remains explicit opt-in. CPU `backend="auto"` may select native CPU
through the separate bounded runtime policy accepted in PR #29.

The tools below remain an offline, exact-record advisor for performance audits;
they are not the runtime dispatcher. Runtime CPU selection uses a small paired
probe ladder, an execution-context signature and a conservative ratio model;
see [the CPU auto-dispatch RFC](cpu-auto-dispatch-rfc.md).

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
the timed operation. Device and host records are not interchangeable. Since
C ABI 2 / Python API 4 the sweep also measures native CUDA with
`penalty="l1"` instead of skipping it. `validate_record` accepts such a case only
when the record's `native_cuda_abi` shows `abi_version` 2 or later or lists
`l1` in `supported_penalties`; an ABI 1 record that claims L1 is rejected.

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
[`p6-windows-ryzen9900x-native-cpu-v2.json`](https://github.com/Funtrollor/renewable-huber/blob/main/benchmarks/baselines/p6-windows-ryzen9900x-native-cpu-v2.json),
captured at `be8984a` with the post-0.7.0 Rust CPU engine. Its strict
native/reference gate
([`p6-windows-ryzen9900x-native-cpu-v2-gate.json`](https://github.com/Funtrollor/renewable-huber/blob/main/benchmarks/baselines/p6-windows-ryzen9900x-native-cpu-v2-gate.json))
passes all 32 standard combinations across shape, penalty, dtype, and public
operation using 0.5-second samples: NumPy/native 1.38x-7.86x, median 1.81x.
The first capture used 0.25-second samples and three pairs exceeded the 5%
relative MAD; the sweep was recaptured with longer samples, not more repeats.
It was taken on the fixed host while the desktop was in light use; the
processes using the most CPU before and after each run are recorded in the
pull request that added it. The earlier
[`p3-windows-ryzen9900x-native-cpu-v2.json`](https://github.com/Funtrollor/renewable-huber/blob/main/benchmarks/baselines/p3-windows-ryzen9900x-native-cpu-v2.json)
(1.17x-15.65x, median 1.68x, 0.25-second samples) is historical; see
[P1](native-core-p1.md#optimized-schema-v2-baseline) for why the wide
unpenalized ratios are lower now. The approved CUDA baseline is
[`p3-windows-rtx5070ti-native-cuda-v2.json`](https://github.com/Funtrollor/renewable-huber/blob/main/benchmarks/baselines/p3-windows-rtx5070ti-native-cuda-v2.json).
It passes all 16 host-input and all 16 device-input CuPy comparisons using
0.5-second samples. Both records use three warmups and nine measured samples.

A result captured while another benchmark, build or test suite runs on the
host must not be promoted. The fixed host is also the maintainer's desktop,
and light interactive use during a capture (browser, chat, launchers) is
acceptable on two conditions: the processes using the most CPU (or, for GPU
captures, the GPU clients) are recorded before and after each run and kept
with the evidence, and the record passes its relative-MAD gate unchanged. A
capture that misses the MAD gate is recaptured with longer samples, never
accepted by relaxing the limit.

Two N5 fixed-host baselines were added for native CUDA at C ABI 2 / Python
API 4 on the same RTX 5070 Ti host. No threshold changed for either. The
[native penalty completion plan](native-penalty-completion-plan.md#n5-fixed-host-results)
has the analysis and every hash.

- **`penalty="none"` interleaved A/B**, `fca7b83` against `10fc363`, captured
  cold and steady with the frozen-plan protocol described below. Both gates
  passed on all 16 native cases, and every difference is inside ±10%. The
  records are
  `p5-windows-rtx5070ti-native-cuda-none-ab-{cold,steady}-{baseline,candidate,gate,sample-repetitions-plan}.json`,
  for example the
  [cold gate report](https://github.com/Funtrollor/renewable-huber/blob/main/benchmarks/baselines/p5-windows-rtx5070ti-native-cuda-none-ab-cold-gate.json).
- **L1**: three standard-profile runs at `4114918`,
  [run 1](https://github.com/Funtrollor/renewable-huber/blob/main/benchmarks/baselines/p5-windows-rtx5070ti-native-cuda-l1-run1.json),
  [run 2](https://github.com/Funtrollor/renewable-huber/blob/main/benchmarks/baselines/p5-windows-rtx5070ti-native-cuda-l1-run2.json) and
  [run 3](https://github.com/Funtrollor/renewable-huber/blob/main/benchmarks/baselines/p5-windows-rtx5070ti-native-cuda-l1-run3.json).
  They use the shape sweep's default 0.1-second samples, three warmups and
  nine measured samples. In 46 of 48 same-transport comparisons native CUDA is
  below 0.90x CuPy in all three runs; two show no measurable difference, and
  none is repeatably slower.

`validate_record` accepts all four A/B baseline and candidate records and all
three L1 runs.

## Fixed-runner regression gate

Capture a v2 baseline and candidate on the same self-hosted runner. Use at
least three warmups and nine measured samples. With the default stabilized
protocol, a sample may aggregate several independent operations while retaining
per-operation units.

For acceptance, baseline and candidate must be captured as aligned pairs. The
runner alternates `A -> B`, then `B -> A`, so neither binary owns only the cold
or hot half of the session:

```powershell
python scripts/benchmarks/run_interleaved_benchmark.py `
  --baseline-python C:\bench\base\Scripts\python.exe `
  --baseline-repo C:\bench\base-source `
  --candidate-python C:\bench\candidate\Scripts\python.exe `
  --candidate-repo . `
  --output-dir artifacts/interleaved `
  --rounds 9 --profile smoke --backend all --penalty none --dtype both `
  --freeze-sample-repetitions
```

The runner writes every one-sample round, two merged schema-v2 records, and a
machine-readable gate report. It requires the existing fixed-runner checks and
also gates the median of the nine aligned `candidate / baseline` ratios.
GPU capture is local-only: build isolated baseline and candidate CPU/CUDA
extensions on the fixed Ryzen/RTX host and run this command there. Record the
tested commit and environment alongside the JSON report. Do not commit round or
calibration files; accepted fixed-host records go to `benchmarks/baselines/`
as the [GPU host runbook](gpu-host-runbook.md) describes.

Three options exist only in the interleaved runner:

- `--freeze-sample-repetitions` calibrates every case once for both variants
  before the first timed round. It writes `sample-repetitions-plan.json`
  (each case gets the larger of the two calibrated block sizes, still capped
  by `--max-sample-repetitions`) and applies that plan to every round of both
  variants. Without it, each round sizes its own blocks, and
  `merge_round_records` refuses to merge rounds whose `sample_repetitions`
  differ. The only other way through was `--max-sample-repetitions 1`, which
  on a desktop GPU in use pushed relative MAD past the gate. In this mode both
  variants run the candidate's sweep harness against their own source tree
  (`RENEWABLE_HUBER_BENCHMARK_SOURCE_ROOT`). Records keep `git_revision` (the
  measured tree) apart from `benchmark_harness_git_revision`. `gate.json` and
  each merged record's `interleaved_capture` state
  `sample_repetitions: {policy, plan_sha256, harness}`; `policy` is
  `frozen_plan` for this mode and `per_round_calibration` otherwise. The gate
  refuses variants whose policies differ.
- `--allow-native-version-change` permits an A/B across a native interface
  change. It drops only `abi_version` and `python_api_version` from the
  hardware and runtime fingerprint; driver and runtime versions, GPU, CuPy,
  Python, CPU, BLAS and threading must still match.
- `--iteration-policy` (default `relative`) sets how float32 cases' solver
  iterations are gated. float32 sits at its rounding floor at `tol=1e-6`, so
  any change to a summation order moves its iteration count by more than one
  without the solver getting worse: the 0.7.1 Rust CPU and CUDA changes moved
  float32 L1 streams by 2-11 iterations. Under `relative`, a float32 case whose
  median iterations moved by more than one still passes only if all of these
  hold:
  - the move is at most 25% of the baseline count;
  - each iteration is no slower than the slowdown limit (candidate seconds per
    iteration over the baseline's), so fewer iterations cannot hide slower
    ones;
  - both records carry `median_final_objective`, and the two differ by at most
    1e-4 relative. That bound is measured: the released 0.7.0 -> 0.7.1 Rust
    CPU change moved float32 final objectives by at most 6.1e-5 (streaming L1)
    and float64 ones by under 1e-15.

  float64 cases always keep the one-iteration limit. `absolute` restores it for
  float32 too. A whole-gate recomputation with a larger
  `max_iteration_delta`, as the P7 A/B needed, is no longer the way through.

`gate.json` schema version 2 added `allow_native_version_change` and, for each
gated native family, both sides' `abi_version`/`python_api_version` and
whether they `changed`. Schema version 3 adds `iteration_policy` and
`iterations_changed`, the keys of every float32 case that passed with a moved
iteration count. Each check also reports `baseline_iterations`,
`candidate_iterations`, `per_iteration_ratio` and
`final_objective_relative_difference`. Shape-sweep records now carry
`final_objectives` (the last update's diagnostic objective, per sample) and
`median_final_objective`; a record from an older harness simply lacks them,
and a float32 case that needs them then fails. `check_performance_regression.py`
has neither option: it keeps rejecting an interface change, and any iteration
move beyond one, against a stored baseline.

For a diagnostic record that was not captured by the interleaved runner, the
older non-paired checker remains available:

```powershell
python scripts/benchmarks/check_performance_regression.py `
  --baseline benchmarks/baselines/native-v2-fixed-runner.json `
  --candidate artifacts/native-v2-candidate.json `
  --output artifacts/native-v2-gate.json
```

| Engine class | Maximum median slowdown | Maximum relative MAD | Other requirements |
| --- | ---: | ---: | --- |
| Rust native CPU | 1.10x | 5% | at least 9 repeats, convergence, same runner/thread/BLAS fingerprint, median iterations within 1 (float32 in an interleaved A/B: the relative policy above), and no slower than matched NumPy |
| Native CUDA | 1.15x | 10% | at least 9 repeats, convergence, same GPU/runtime fingerprint, median iterations within 1 (float32 in an interleaved A/B: the relative policy above), and no slower than matched CuPy under the same host/device transport |

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

The thresholds are guardrails, not statistical proof of a speedup. A paired
GPU change below roughly 10% remains indistinguishable from this host's normal
noise even when it passes. Repeated noisy measurements should be recaptured
rather than accepted by increasing a tolerance. Correctness still comes first:
the golden/differential suite must pass independently of the timing gate.

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

Do not turn this exact-shape advisor into a persisted runtime crossover map.
It is intentionally hardware-fingerprinted evidence for a recorded workload;
the public CPU dispatcher instead recalibrates within a fixed cost bound and
never writes host-specific state to disk. Native CUDA remains explicit.
