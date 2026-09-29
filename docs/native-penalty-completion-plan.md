# Native penalty completion plan

- Status: N0–N4 implemented and passed on the fixed GPU host (stage 1,
  2026-09-29, `4114918`). N5: the native CUDA L1 baseline is recorded; the
  `penalty="none"` A/B is **not yet accepted** (see "N5 fixed-host results"
  and "Still open" at the end)
- Baseline: `v0.6.1` / CUDA C ABI 1 / CUDA Python API 3
- Scope: complete the existing public penalties, `none` and `l1`
- Out of scope: adding L2, elastic-net, reduced precision, or automatic CUDA selection

## Outcome

The public estimator already defines `penalty` as `none | l1`. Native CPU
implements both paths. Native CUDA implements only `none` and correctly rejects
`l1` before entering the extension. This work closes that CUDA gap, strengthens
the native CPU oracle coverage, and makes penalty support discoverable before an
update starts.

Completion means all three engines can resume the same L1 checkpoint and process
the next batch with equivalent state, diagnostics, and predictions:

```text
NumPy reference <-> Rust CPU <-> Rust/CUDA
        checkpoint format v2 remains portable
```

No phase may change the paper equations, batch-order semantics, frequency-weight
semantics, intercept exclusion, strict dtype behavior, or public estimator API.

## Frozen decisions

1. Keep `tests/golden/native_core_v1.json` byte-identical. Add a v2 corpus; do
   not regenerate v1.
2. Generate and review the v2 expected values from the NumPy reference at the
   baseline commit before using any candidate native engine as an oracle.
3. Keep checkpoint format 2. It already stores the config, `previous_lambda`,
   `weight_sum`, coefficients, and information matrix required by L1.
4. Raise the CUDA C ABI from 1 to 2 and the CUDA Python payload API from 3 to 4.
   New and old components must fail closed instead of silently treating L1 as
   `none`.
5. Preserve exactly 17 exported `rh_cuda_*` symbols. Evolve the existing update
   contract; do not add a second versioned update symbol.
6. Change `native/contracts/rh_cuda_contract.json` first, then update every
   mirror named by that manifest. Preserve all parser-count assertions.
7. Native CUDA remains explicit opt-in. Functional L1 support does not authorize
   `backend="auto"` to select CUDA.
8. GPU correctness and performance run on the fixed local GPU host, not GitHub
   Actions. Portable ABI, Rust, Python, build, and CPU tests remain in CI.

## Numerical contract

For total effective weight `N`, feature count `p`, Huber threshold `tau`, and
configured scale `lambda_scale`:

```text
lambda = lambda_scale * tau * sqrt(log(max(p, 2)) / N)
```

The CUDA L1 transition must match the reference and Rust CPU behavior:

- include the historical quadratic term from the information matrix;
- include the historical subgradient correction using the previous state's
  `weight_sum`, `previous_lambda`, and `sign(coefficients)`;
- apply soft thresholding to every coefficient except the trailing intercept;
- use the existing 40-attempt LAMM backtracking rule, `phi` update, tolerance,
  and convergence definition;
- report the final penalized diagnostic objective and current lambda;
- commit the current lambda to `previous_lambda` only after a successful,
  transactional update;
- form final renewable information at the accepted coefficients;
- leave active device state unchanged after any validation, CUDA, or solver
  failure.

Strict `float32` and `float64`, host input and zero-copy CUDA DLPack input,
weighted and unweighted batches, and intercept/no-intercept configurations are
all part of the contract.

## Work stages

The names `N0`-`N5` avoid the repository's two existing P0-P3 numbering schemes.

### N0 — Freeze oracle and compatibility boundaries

Deliver:

- `native_core_v2.json`, generated from baseline NumPy, with at least:
  - weighted three-batch L1 float64, including zero and non-unit weights;
  - multi-batch L1 float32 without an intercept and with true zero coefficients;
  - sparse L1 float64 with an intercept that must remain unpenalized;
  - streaming L1 with `lambda_scale=0`;
- tests that replay v1 and v2 independently;
- an ABI 2 contract proposal covering a fixed-width penalty enum and
  `lambda_scale`;
- `generate_native_golden.py --check` gates for both corpora.

Accept when v1 is byte-identical, v2 is deterministic, and NumPy can replay
every case twice with identical serialized output.

### N1 — Shared contract, capabilities, and fail-closed ABI 2

Deliver:

- CUDA update config generalized from unpenalized-only to `none | l1`;
- fixed-width `NONE=0`, `L1=1` C ABI values and `lambda_scale`;
- manifest, C header, C++ assertions, Rust layout, PyO3 boundary, and version
  metadata updated together;
- CUDA Python API 4 metadata with `supported_penalties`;
- backend capability field `native_update_penalties` populated only through
  `capabilities_of()`:
  - native CPU: `{none, l1}`;
  - old CUDA API: `{none}`;
  - new CUDA API: `{none, l1}`;
- explicit requests for an unsupported penalty fail before native execution;
- `rh-cuda-ffi` reuses `rh-core` penalty/config validation where it reduces
  duplicated policy without coupling CUDA buffers to CPU implementation types.

Accept when malformed penalties, ABI 1/API 3 mismatches, and partial struct
layouts all fail closed; the library still exports exactly 17 symbols; all
no-CUDA ABI/layout tests run in CI.

### N2 — CUDA L1 transition

Deliver the L1 solver behind the existing whole-batch native update boundary.
The implementation may reuse or extend the current trial, candidate, gradient,
delta, reduction, and information workspaces. Its internal kernel split is left
to the implementer, subject to the numerical and transactional contracts above.

The first correct implementation may use ordinary stream launches and decline
CUDA Graph capture for L1. Graph specialization is an optimization, not a
functional acceptance requirement.

Accept when direct C ABI smoke tests cover host/device input, weighted streaming,
intercept masking, `lambda_scale=0`, invalid config, restored asymmetric
information, non-convergence, and failure rollback for both dtypes.

### N3 — Bindings and cross-backend resume

Deliver:

- PyO3 host and DLPack updates that pass penalty and lambda policy explicitly;
- removal of the Python native-CUDA L1 rejection only after API 4 support is
  verified;
- native CUDA replay of every v2 L1 case;
- NumPy -> CPU -> CUDA and CUDA -> NumPy/CPU checkpoint migration tests;
- immediate prediction and next-batch `partial_fit` after restore;
- no checkpoint format or public estimator API change.

Accept when a checkpoint taken after any v2 batch can resume on each other
engine and match the uninterrupted oracle within the existing dtype-specific
tolerances for state, diagnostics, and predictions.

### N4 — Full correctness and regression matrix

Required gates:

- Python `core`, `native-cpu`, `cuda`, and `all` profiles;
- all Rust formatting, clippy, check, and scoped tests;
- C++ ABI syntax check, static CMake build, and CTest;
- CUDA C ABI status-survival test across translation units;
- exact 17-symbol export check and all contract parser counts;
- v1 `none` results unchanged and v1/v2 three-engine differential replay;
- host and DLPack L1, float32/64, weighted/unweighted,
  intercept/no-intercept, checkpoint resume, NaN/Inf, and rollback tests.

No test may convert a missing native dependency or GPU into success in a
required profile.

### N5 — Benchmark, documentation, and release readiness

Deliver:

- shape-sweep native CUDA runner accepts L1 instead of hard-coding `none`;
- removal of the explicit CUDA-L1 benchmark skip;
- fixed-host benchmark records separated by host/device transport, dtype,
  lifecycle, operation, shape, and penalty;
- interleaved A/B protection showing the existing CUDA `none` path has not
  regressed;
- new L1 comparisons against CuPy under the same input-transport contract;
- support matrix, architecture, P2 history, CUDA package README, performance
  policy, changelog, and release checklist updated together.

For existing `none`, use at least nine aligned paired samples, require all cases
to converge, median iteration difference at most one, relative MAD at most 10%,
and candidate slowdown at most 1.15x. Do not claim differences inside the
host's documented approximately 10% noise band.

There is no previous native CUDA L1 baseline. Initial L1 acceptance therefore
requires correctness plus stable measurements. Keep it explicit opt-in unless
the same-transport native result demonstrates a repeatable advantage over CuPy;
freeze that accepted result as the baseline for later releases.

## Implementation status

Implemented on `claude/nifty-goldberg-0lm77u` (base `fca7b83`):

- **N0.** `tests/golden/native_core_v2.json` holds the four required L1 cases,
  generated by `scripts/generate_native_golden.py --corpus v2` from the NumPy
  reference and checked twice for byte-identical output. Every v2 case
  converges (`tol=1e-6`; tighter tolerances make LAMM stall at `max_iter`,
  which is a poor cross-engine oracle). v1 is byte-identical to `v0.6.1`;
  CI's `--check` of both corpora is what keeps them that way.
- **N1.** C ABI 2 / Python API 4. `RhCudaUnpenalizedConfig` became
  `RhCudaUpdateConfig` (72 bytes: `penalty`, `reserved0`, `lambda_scale`
  appended); `RH_CUDA_PENALTY_NONE=0`, `RH_CUDA_PENALTY_L1=1`. Unknown
  penalties, a non-zero `reserved0` and a negative or non-finite
  `lambda_scale` fail before device work. `rh-cuda-ffi` validates through
  `rh_core::UpdateConfig::validate`. `native_update_penalties` is resolved
  only by `capabilities_of()`: NumPy/portable `None`, native CPU `{none, l1}`,
  native CUDA whatever `version()["supported_penalties"]` advertises, and
  `{none}` when it advertises nothing. The core refuses an unadvertised
  penalty with `ValidationError` before the engine is called. The library
  still exports 17 symbols; `rh_cuda_engine_predict_host` was renamed
  `rh_cuda_engine_predict` as part of the ABI 2 break (see below).
- **N2.** `solve_l1` in `native/cuda/src/pipeline.cu` mirrors the Rust CPU
  engine step for step. The historical subgradient term and the majorization
  inner product ride the existing single objective transfer (reduction slots
  4 and 5); the final L1 norm rides the update's one transactional sync.
  CUDA Graph capture is supported for the L1 candidate objective as well.
  `previous_lambda` is committed only after the transactional swap.
- **N3.** PyO3 `update`/`update_device` take `penalty` and `lambda_scale`; the
  Python L1 rejection is gone; `tests/test_native_cuda_backend.py` replays v1
  and v2 and resumes an L1 checkpoint NumPy -> CUDA -> CPU.
- **N4.** Portable gates (contract referee, Rust layout, fakes, NumPy and
  native CPU v2 replay) run in CI. The CUDA smoke test gained five cases
  (reference L1 stream, unpenalized intercept, rejection rollback, device vs
  host L1, device prediction).

Beyond the plan, the same ABI break carries device-resident prediction:
`RhCudaHostPrediction` became `RhCudaPrediction` with `n_features_in` and an
`input_location` (`RH_CUDA_MEMORY_HOST`/`RH_CUDA_MEMORY_DEVICE`), so a CUDA
DLPack tensor is read in place and the raw feature matrix may be widened on
device, exactly as for updates.

## N5 fixed-host results

Stage 1 and stage 2 of the [GPU host runbook](gpu-host-runbook.md) ran on
2026-09-29 at `4114918`, the first time this implementation executed on a
GPU. Section numbers such as §3a refer to the runbook.

| Component | Version |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Ti, compute capability 12.0, WDDM |
| Driver | 616.64 (CUDA UMD 13.4) |
| CUDA toolkit | 12.9 (`nvcc` 12.9.41) |
| Host compiler | MSVC 19.43.34809 (VS 2022 Build Tools 17.13) |
| Build tools | CMake 3.29.2, Ninja 1.12.0, rustc 1.97.1 |
| Python stack | CPython 3.11.0, NumPy 2.4.6, SciPy 1.17.1, `cupy-cuda12x` 14.2.0 |
| Native CUDA `version()` | ABI 2, Python API 4, runtime 12090, driver 13040, `supported_penalties` `[none, l1]` |

### Stage 1: correctness on the device

Every stage 1 command passed:

| Command | Exit | Result |
|---|---:|---|
| `run_test_profile.py --check` | 0 | 6 profiles cover 25 test modules |
| `run_test_profile.py core --verbose` | 0 | Ran 326, OK (skipped=2) |
| `run_test_profile.py native-cpu --verbose` | 0 | Ran 20, OK |
| `run_test_profile.py cuda --verbose` | 0 | Ran 40, OK (skipped=2) |
| `run_test_profile.py performance --verbose` | 0 | Ran 54, OK |
| `generate_native_golden.py --check` and `--corpus v2 --check` | 0, 0 | "Golden corpus matches 4 generated cases" for each |
| static CMake build and `ctest` | 0 | 1/1 passed; `ctest --verbose` shows `rh_cuda_smoke passed`, which the binary prints only after all 13 named cases, including the four `l1_*` cases and `prediction_reads_device_input` |
| shared build and `dumpbin /exports` | 0 | exactly 17 `rh_cuda_*` exports |
| `python -m build --wheel` | 0 | base wheel built |
| `smoke_test_cuda_wheels.py` | 0 | "Clean wheel smoke passed" in a clean venv with the pip `nvidia-*-cu12` 12.9 runtime wheels |

The two skips in `core` and in `cuda` are the PyTorch and TensorFlow CUDA
DLPack integration tests in `tests/test_dlpack_adapters.py`. Neither framework
is installed by the runbook's extras, so those two device paths were not
exercised.

### Stage 2 §3a: `penalty="none"` A/B, not accepted

Baseline `fca7b83` (ABI 1 / API 3) against candidate `4114918` (ABI 2 /
API 4). Both sides used CPython 3.11.0 with the same NumPy, SciPy and CuPy
versions, and both native CUDA extensions were Release builds for SM 120. The
official runs used the runbook command with `--backend gpu` and
`--max-sample-repetitions 1` added (the runbook now explains why), nine
rounds, the standard profile, and a separate `--output-dir` for the steady
run. Each lifecycle gates 16 native cases: two transports, four shapes, two
dtypes.

Both `gate.json` reports say `passed=false` with 16 checked cases. The
reasons, verbatim, with the number of cases that report each one:

| Reason | cold | steady |
|---|---:|---:|
| `hardware or runtime fingerprint differs` | 16 | 16 |
| `baseline relative MAD 10.4% exceeds 10.0%` | 1 | 0 |
| `baseline relative MAD 10.9% exceeds 10.0%` | 0 | 1 |
| `candidate relative MAD 19.1% exceeds 10.0%` | 0 | 1 |
| `candidate relative MAD 10.4% exceeds 10.0%` | 0 | 1 |

The only fingerprint fields that differ are `native_cuda_abi.abi_version`
(1 against 2) and `native_cuda_abi.python_api_version` (3 against 4); the CuPy
fingerprints are identical. `hardware_fingerprint()` in
`scripts/benchmarks/performance_policy.py` includes both fields for native
CUDA engines, and the interleaved gate hard-codes `require_same_hardware=True`,
so on `main` an A/B across the ABI 1 → ABI 2 change cannot pass this gate,
whatever the timings.

The plan's own five criteria, evaluated per case from the same records:

| Criterion | cold | steady |
|---|---|---|
| at least nine aligned pairs | 16/16 (9 each) | 16/16 (9 each) |
| every case converges | 16/16 | 16/16 |
| median iteration difference ≤ 1 | 16/16 (all 0) | 16/16 (all 0) |
| relative MAD ≤ 10% on both sides | 15/16 | 14/16 |
| candidate slowdown ≤ 1.15x | 16/16 | 16/16 |
| **all five** | **15/16** | **14/16** |
| paired median, candidate / baseline | 0.921–1.045 | 0.986–1.111 |
| native / CuPy, same transport (gate limit 1.0) | 16/16, 0.36–0.82 | 16/16, 0.27–0.78 |

The MAD failures are all device-input float32: cold streaming (baseline
10.4%), steady latency (baseline 10.9%, candidate 19.1%) and steady streaming
(candidate 10.4%). Every paired median but one lies within ±10%, which is no
measurable difference on this host. The exception is device-input reference
float32 in the steady run: paired median 1.111 with 8 of 9 pairs slower, just
outside the ±10% band and below the 1.15x limit. The same case measured
1.026 in the cold run. This is a single-run observation, not a regression
claim; the recapture will show whether it repeats. The median-of-medians
ratios, which the fixed-runner part of the gate uses, stay below 1.15x as
well; the highest are 1.144 (steady device-input latency float32, the case
with 19.1% candidate MAD), 1.119 (steady host-input latency float32) and
1.111 (the steady reference case above).

**Verdict: §3a is not accepted,** for two independent reasons: the gate fails
every case on the ABI fields of the fingerprint, and the plan's MAD criterion
fails in one cold and two steady cases.

Environment during the runs: Wallpaper Engine was closed before the official
runs. OpenAI Codex (`ChatGPT.exe`, `codex-computer-use-swift.exe`) appeared as
a GPU client from 16:40, that is, during the steady run and all three L1 runs.
Chrome, Discord, NVIDIA App and NVIDIA Overlay were present throughout. The
first attempts with the default repetition cap stopped in
`merge_round_records` ("sample repetition calibration changed between
interleaved rounds") and wrote no gate report; they are diagnostics only.

The records are not committed because they are not accepted. They stay on the
host under `artifacts/`, per the retention rule in
[CUDA performance paths](gpu-performance.md):

| Record | SHA-256 |
|---|---|
| `artifacts/n5-none-ab/baseline.json` | `ebf14fce8545f8323ce10e97ad238e13129f6bbb2067c1b2b3fba95bfeb902b3` |
| `artifacts/n5-none-ab/candidate.json` | `9e73bc182935ecf0b787e404063551a3af6362030e19a1327030a30774324dea` |
| `artifacts/n5-none-ab/gate.json` | `b8092da595696f3f6c7188479fc3fc637c43e6369c2e0251f88fe4e6bb9041ff` |
| `artifacts/n5-none-ab-steady/baseline.json` | `b270a2e7d8fb35db8e8ec9ac4bf48e3d26a62b9cc7e9e470a7d80e234d945b84` |
| `artifacts/n5-none-ab-steady/candidate.json` | `ad1c85d0aed1a9921ab73cb9cdb31890568d2ce483bf93312d66b86dc8191207` |
| `artifacts/n5-none-ab-steady/gate.json` | `707942a96d594ad4e07da539e74342218b33f522d114c8dcc3509fbde8170bd0` |

### Stage 2 §3b: L1, native CUDA against CuPy

The runbook command ran unchanged three times, back to back. Each run holds
96 cases: native CUDA and CuPy, each with host and device input, over four
shapes, two dtypes, and cold `fit`, cold `partial_fit` and steady
`partial_fit`. Every case converged. Each run also lists 24 skips, all
"fit resets the estimator; steady state is partial_fit only".

That gives 48 native-against-CuPy comparisons with shape, dtype, lifecycle,
operation and transport held equal. Classified by the ratio of the native
median to the CuPy median in each of the three runs:

| Category | Comparisons |
|---|---:|
| ratio < 0.90 in all three runs (repeatable native advantage) | 46 |
| ratios within 0.90–1.10 (no measurable difference) | 2 |
| ratio > 1.10 in all three runs (repeatable native disadvantage) | 0 |

Across the 46, the per-run ratios run from 0.23 to 0.89 with a median of
0.34; 38 of the 46 are below 0.5 in every run. The advantage is smallest for
cold `fit` of the streaming shape and of reference float64. The two comparisons with no
measurable difference are both host-input streaming cold `fit`: float32 at
1.055–1.071 and float64 at 0.902–0.978.

The largest native relative MAD in any run is 9.4%. On the CuPy side, three
rows exceed 10% (24.1%, 10.3%, 10.5%). The native run-to-run spread, the
largest over the smallest of the three native medians, is 1.04–1.49. It
exceeds 1.2 only for reference float32 `partial_fit` (all four
lifecycle/transport combinations, 1.22–1.49) and host-input latency float64
cold `partial_fit` (1.26).

The float32 L1 `partial_fit` iteration counts (summed over the batches of the
stream) differ between native CUDA and CuPy; each engine's count is the same
in all three runs. The float64 counts match everywhere:

| Shape (float32 `partial_fit`) | native CUDA | CuPy |
|---|---:|---:|
| reference | 29 | 26 |
| streaming | 52 | 63 |
| wide | 49 | 51 |

(Wide float32 cold `fit` differs by exactly one, 14 against 15.) A follow-up
diagnostic replayed the same shape-sweep datasets through all four engines
with `penalty="l1"`, `tol=1e-6`, and compared each float32 result with a
float64 NumPy fit of the same float32-valued data. The float32 iteration
counts differ across all four engines, NumPy included (streaming: NumPy 55,
native CPU 56, CuPy 63, native CUDA 52). Every engine converged, and every
float32 result lies within 8.2e-7 to 4.4e-5 of the float64 reference (largest
coefficient difference relative to the largest reference coefficient). Native
CUDA is among the closest (8.2e-7 reference, 2.8e-6 streaming, 7.1e-6 wide),
and its host-input and DLPack results are identical. The differing counts are
float32 rounding near `tol=1e-6`, not an engine defect.

The three runs are committed as the fixed-host L1 baseline.
`benchmark_shape_sweep.py` writes them with Windows CRLF line endings;
`.gitattributes` stores them with LF, which is the only difference from the
files on the host:

| Record | SHA-256 on the host (CRLF) | SHA-256 committed (LF) |
|---|---|---|
| [`p5-windows-rtx5070ti-native-cuda-l1-run1.json`](https://github.com/Funtrollor/renewable-huber/blob/main/benchmarks/baselines/p5-windows-rtx5070ti-native-cuda-l1-run1.json) | `41ff3e17360558251c8eeb37649dfd39c3fbf6c3ec1f8a919fc5c5b415f0f1ee` | `ae1b52a0b2863e248d305d56df7ca79f82dac18ae8efaf711ed86ce45ce9e48b` |
| [`p5-windows-rtx5070ti-native-cuda-l1-run2.json`](https://github.com/Funtrollor/renewable-huber/blob/main/benchmarks/baselines/p5-windows-rtx5070ti-native-cuda-l1-run2.json) | `de61ce2eedfee13a2a0770fc1aa21e77e4e165c6b6000d46248bf001f5c4cf26` | `e18537cbf99c03795c9bca74da3dc2483b7511678d8961805559dbe4838eed36` |
| [`p5-windows-rtx5070ti-native-cuda-l1-run3.json`](https://github.com/Funtrollor/renewable-huber/blob/main/benchmarks/baselines/p5-windows-rtx5070ti-native-cuda-l1-run3.json) | `60b9d45911c04a66885f13918446cd545642b5c92320ce4bc7e0d0d61939b333` | `08ae8e9097af45701e0ac5d2e3392adb49d5c22fac9909228eea41faa85d811b` |

### Decision: native CUDA L1 stays explicit opt-in

**Native CUDA L1 stays explicit opt-in; `backend="auto"` never selects it.**

1. The condition this plan and the runbook set for considering more than
   opt-in, a repeatable same-transport advantage over CuPy, is met on this
   host: 46 of 48 comparisons, two with no measurable difference, none worse.
   That is recorded here as evidence.
2. It still does not change selection. Frozen decision 7 and this plan's
   scope exclude automatic CUDA selection, and `auto` does not select native
   CUDA even for `penalty="none"`, where native CUDA also beats CuPy (0.27–0.82
   in the §3a runs). Automatic CUDA selection is a question independent of
   the penalty. It needs its own RFC, as the CPU path had in the
   [CPU auto-dispatch RFC](cpu-auto-dispatch-rfc.md), with evidence from more
   than one host.
3. The `penalty="none"` A/B is not yet accepted.
4. The float32 L1 trajectories differ between engines: the results are
   numerically equivalent, but the iteration counts are not. Switching engines
   silently would change the diagnostics users see.

## Still open

- **§3a acceptance.** The maintainer has decided how to proceed. A separate
  pull request, not yet merged, adds an option to the interleaved runner only,
  `--allow-native-version-change`. It excludes only `abi_version` and
  `python_api_version` from the same-hardware fingerprint, and it records the
  option and both sides' ABI and API versions in `gate.json`.
  `check_performance_regression.py` keeps rejecting ABI differences by
  default. After that pull request merges, §3a is recaptured on a quiet GPU.
  If a relative MAD still exceeds 10%, the thresholds stay and the recapture
  raises `--rounds` or `--minimum-sample-seconds` instead.
- **Consuming the L1 baseline.** `performance_policy._validate_timing_contract`
  on `main` still raises "native CUDA benchmark records may not claim L1
  support" for any native CUDA case whose penalty is not `none`. Until that
  changes, `validate_record` and `check_performance_regression.py` cannot load
  the committed L1 runs. The same pending pull request accepts native CUDA L1
  records only when the recorded `native_cuda_abi` is 2 or later, or when its
  `supported_penalties` contains `l1`.
- **PyTorch and TensorFlow CUDA DLPack.** Their integration tests skipped in
  stage 1 because neither framework was installed, so neither path has run on
  a device yet.
