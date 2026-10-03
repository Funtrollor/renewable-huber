# Native penalty completion plan

- Status: N0–N4 implemented and passed on the fixed GPU host (stage 1,
  2026-09-29, `4114918`). N5: the `penalty="none"` A/B is **accepted**
  (frozen-plan recapture at `10fc363`, 2026-09-30) and the native CUDA L1
  baseline is recorded (see "N5 fixed-host results" and "Still open" at the
  end)
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
GPU. The §3a A/B was recaptured on 2026-09-30 at `10fc363`, after #50 and #51
made an A/B across the ABI change measurable; that recapture is the accepted
result. Section numbers such as §3a refer to the runbook.

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

### Stage 2 §3a: `penalty="none"` A/B, accepted

**Verdict: §3a is accepted.** `fca7b83` and `10fc363` show no measurable
difference for `penalty="none"`. The official result is the frozen-plan
recapture of 2026-09-30. The two earlier attempts of 2026-09-29 are
superseded and listed at the end of this section.

Baseline `fca7b83` (ABI 1 / API 3) was compared with candidate `10fc363`
(ABI 2 / API 4). With `--freeze-sample-repetitions`, both sides run the
candidate's measurement harness against their own source tree. The baseline
records therefore carry `git_revision` `fca7b83` and
`benchmark_harness_git_revision` `10fc363`. The candidate records have no
harness field, because the candidate measured its own tree.

Both venvs were the ones from 2026-09-29: CPython 3.11.0 with the same NumPy,
SciPy and CuPy versions, and Release native CUDA builds for SM 120. Nothing was
rebuilt. No native library code changed between `4114918` and `10fc363`, so
the candidate extension built earlier is the one that was measured.

The command was the runbook §3a command: `--profile standard --backend gpu
--penalty none --dtype both --lifecycle cold --operation partial-fit
--rounds 9 --allow-native-version-change --freeze-sample-repetitions`, with
every other argument at its default. The steady run was identical apart from
`--lifecycle steady --output-dir artifacts/n5-none-ab-steady`. Each lifecycle
gates 16 native cases: two transports, four shapes and two dtypes.

| | cold | steady |
|---|---|---|
| window (+08:00) | 16:37:44–16:41:41 | 16:42:07–16:46:34 |
| wall time, exit code | 236.3 s, 0 | 266.9 s, 0 |
| `gate.json` `passed` | `true` | `true` |
| `schema_version`, `checked_cases` | 2, 16 | 2, 16 |
| failure reasons | none | none |
| `allow_native_version_change` | `true` | `true` |
| `native_versions.native_cuda_abi` | ABI 1 / API 3 → ABI 2 / API 4, `changed: true` | same |
| `sample_repetitions.policy`, `.harness` | `frozen_plan`, `candidate` | `frozen_plan`, `candidate` |
| `sample_repetitions.plan_sha256` | `d6a026f2405b089bc8739d2771337d00c02b4305db711bad1cd92f3f472a04f3` | `02a4d3dfb5961c412dc0d2c24d7ad08677e1f0ca6601f7cece83b6a02cf74405` |
| plan block sizes (32 cases) | 1–58 | 3–64 (four cases at the default cap of 64) |

**Checks on the frozen plan:**

- Each `plan_sha256` equals the SHA-256 of its plan file.
- The plan is the per-case maximum of the two calibration sweeps.
- All 18 round records of each run (nine per variant) used the plan, and every
  case kept the same repetition count in every round.
- The merged records' repetitions equal the plan.

**Recomputing the gate:** `compare_interleaved_records` and `report` on
`main` were run on the committed baseline and candidate records with
`allow_native_version_change=True`. They reproduce the committed `gate.json`
exactly. Without the option, every case reports only
`hardware or runtime fingerprint differs`.

The plan's five criteria, evaluated per case:

| Criterion | cold | steady |
|---|---|---|
| at least nine aligned pairs | 16/16 (9 each) | 16/16 (9 each) |
| every case converges | 16/16 | 16/16 |
| median iteration difference ≤ 1 | 16/16 (all 0) | 16/16 (all 0) |
| relative MAD ≤ 10% on both sides | 16/16 (max 8.3%) | 16/16 (max 8.5%) |
| candidate slowdown ≤ 1.15x | 16/16 | 16/16 |
| **all five** | **16/16** | **16/16** |
| paired median, candidate / baseline | 0.937–1.056 | 0.925–1.038 |
| median of medians, candidate / baseline | 0.947–1.055 | 0.982–1.081 |
| native / CuPy, same transport (gate limit 1.0) | 16/16, 0.481–0.862 | 16/16, 0.349–0.838 |

**Reading the results:**

- Every difference is inside ±10%, so there is no measurable difference
  between the two engines for `penalty="none"`.
- No threshold was changed, and none needed to be.
- The steady device-input reference float32 case, 1.111 in the superseded
  `--max-sample-repetitions 1` run, measured 1.015 here. That earlier
  observation did not repeat.
- The ungated CuPy control runs the same CuPy code on both sides. Its relative
  MAD reached 13.1% (cold) and 14.0% (steady), and its paired ratios spanned
  0.883–1.129. The gate applies the MAD limit to the gated native engines
  only.

**Environment:** `nvidia-smi --query-compute-apps` listed the same GPU
clients, with the same process IDs, before and after both runs:

- Wallpaper Engine, Chrome, Discord, NVIDIA Overlay, the Claude desktop app
  (`claude.exe`) and Microsoft Edge WebView2;
- Riot Vanguard (`vgtray.exe`), AMD Radeon Software (`AMDRSSrcExt.exe`) and
  Phone Link;
- Windows shell processes.

They were not closed, because the maintainer uses this computer during
measurements. No other Python, Cargo, CMake or CUDA build process was running.

**Committed records.** `benchmark_shape_sweep.py`/`run_interleaved_benchmark.py`
write records and `gate.json` with CRLF on Windows, and `.gitattributes`
stores them with LF; that is the only difference. The plan files are written
as bytes with LF, so their hash is unchanged and still equals `plan_sha256`.

| Record | SHA-256 on the host | SHA-256 committed |
|---|---|---|
| [`p5-windows-rtx5070ti-native-cuda-none-ab-cold-baseline.json`](https://github.com/Funtrollor/renewable-huber/blob/main/benchmarks/baselines/p5-windows-rtx5070ti-native-cuda-none-ab-cold-baseline.json) | `267304f0eb657a4787aeabcdf98ae6e3d89789598071f50217cd9cbad56c0963` | `2349b80df16331a507a64671c6e231dbfb027f44dc9e0e2f8beafc828435f62c` |
| [`p5-windows-rtx5070ti-native-cuda-none-ab-cold-candidate.json`](https://github.com/Funtrollor/renewable-huber/blob/main/benchmarks/baselines/p5-windows-rtx5070ti-native-cuda-none-ab-cold-candidate.json) | `9ed4a927cfdd484f6230694ddd44fc863b9264aecc392e390b832e867d3cd9f1` | `58ec01945db8d08bd6131443cdd2faafd6fd039dd030aea41e24028c8b8e6cad` |
| [`p5-windows-rtx5070ti-native-cuda-none-ab-cold-gate.json`](https://github.com/Funtrollor/renewable-huber/blob/main/benchmarks/baselines/p5-windows-rtx5070ti-native-cuda-none-ab-cold-gate.json) | `800abda85341d999a53bb187ed64618a3331daae71f90ec873c8972930fc382c` | `c9a706652ffa6b5921837f8f7cedd250e4be315b41d23cf5ddbd6fb5e4075b3f` |
| [`p5-windows-rtx5070ti-native-cuda-none-ab-cold-sample-repetitions-plan.json`](https://github.com/Funtrollor/renewable-huber/blob/main/benchmarks/baselines/p5-windows-rtx5070ti-native-cuda-none-ab-cold-sample-repetitions-plan.json) | `d6a026f2405b089bc8739d2771337d00c02b4305db711bad1cd92f3f472a04f3` | same |
| [`p5-windows-rtx5070ti-native-cuda-none-ab-steady-baseline.json`](https://github.com/Funtrollor/renewable-huber/blob/main/benchmarks/baselines/p5-windows-rtx5070ti-native-cuda-none-ab-steady-baseline.json) | `7519364df6d7659b847fde48ee26fd446d9c6f4dc700459edd5c656ab0a4ee49` | `234cd3d807b07e8bb8b2079a6b94cbdc59c4cb23d2455f5e2f82c1c6987b7e4c` |
| [`p5-windows-rtx5070ti-native-cuda-none-ab-steady-candidate.json`](https://github.com/Funtrollor/renewable-huber/blob/main/benchmarks/baselines/p5-windows-rtx5070ti-native-cuda-none-ab-steady-candidate.json) | `097c33305224205ca4ecbc7ce18dc7a78f85668a39bbd90a3edc9205416c7bbd` | `a029daec76b8d8cbd7ad45de6be572df6554135b44df5e179cf76c3b9d2045a9` |
| [`p5-windows-rtx5070ti-native-cuda-none-ab-steady-gate.json`](https://github.com/Funtrollor/renewable-huber/blob/main/benchmarks/baselines/p5-windows-rtx5070ti-native-cuda-none-ab-steady-gate.json) | `41583d00ea2f6cf6be5a0b2c57bf1aab4c679c8efcc5ed0b4255e009ba75b054` | `a845e0c8877e5e6e1edd562a86ee9c6156e2ea9aee184b0ea2d427df84d1b55e` |
| [`p5-windows-rtx5070ti-native-cuda-none-ab-steady-sample-repetitions-plan.json`](https://github.com/Funtrollor/renewable-huber/blob/main/benchmarks/baselines/p5-windows-rtx5070ti-native-cuda-none-ab-steady-sample-repetitions-plan.json) | `02a4d3dfb5961c412dc0d2c24d7ad08677e1f0ca6601f7cece83b6a02cf74405` | same |

`validate_record` on `main` accepts all four committed baseline and candidate
records.

**Kept on the host, not committed**, under `artifacts/n5-none-ab/` and
`artifacts/n5-none-ab-steady/`:

- the two calibration sweeps of each run. Their SHA-256 are: cold baseline
  `91809d09b96539f268c551f6c7f4766f4932c0cfb74c5b80f9d589b1cc92d6ac`, cold
  candidate `965fda24136ebb7b8d062e8def49af56c9f4b0b4fcdf1e4e9b6eea0fb5607227`,
  steady baseline
  `e188cf3312670abad7af521134e2033064053aa476727cdc39ff5c701cf506d3`, steady
  candidate `c5e6c74fee30b919ff33ac8a63cdd7d79ea766e13506d2d4fd18706f0ea832ab`;
- the 18 round records of each run, in `rounds/`. Their SHA-256 are in the
  description of pull request #49.

#### Superseded attempts (not accepted)

These are kept on the host for the record. None of them is evidence for or
against a regression.

1. **Default repetition cap, 2026-09-29 at `4114918`.** Each round calibrated
   its own sample blocks. After nine rounds `merge_round_records` refused to
   merge them ("sample repetition calibration changed between interleaved
   rounds"), so no gate report was written. The round records are in
   `artifacts/n5-none-ab.attempt1-default-reps/` and
   `artifacts/n5-none-ab-steady.attempt1-default-reps/`. Wallpaper Engine was
   running for the cold attempt and closed during the steady one.
2. **`--max-sample-repetitions 1`, 2026-09-29 at `4114918`, without
   `--allow-native-version-change`.** The records merged, but both gates said
   `passed=false`, for two reasons:
   - Every case reported `hardware or runtime fingerprint differs`. The only
     differing fields were `native_cuda_abi.abi_version` and
     `python_api_version`, which is what #50 addresses.
   - With single-operation samples, relative MAD exceeded 10% in one cold case
     and two steady cases, all device-input float32: cold streaming (baseline
     10.4%), steady latency (baseline 10.9%, candidate 19.1%) and steady
     streaming (candidate 10.4%).

   Otherwise the plan's criteria held. Paired medians were 0.921–1.045 cold
   and 0.986–1.111 steady; the 1.111 was steady device-input reference
   float32. Native/CuPy was 0.27–0.82.

   OpenAI Codex (`ChatGPT.exe`, `codex-computer-use-swift.exe`) appeared as a
   GPU client from 16:40, during the steady run.

   The records have moved to `artifacts/n5-none-ab.reps1/` and
   `artifacts/n5-none-ab-steady.reps1/`, with unchanged hashes:

   | Record | SHA-256 |
   |---|---|
   | `n5-none-ab.reps1/baseline.json` | `ebf14fce8545f8323ce10e97ad238e13129f6bbb2067c1b2b3fba95bfeb902b3` |
   | `n5-none-ab.reps1/candidate.json` | `9e73bc182935ecf0b787e404063551a3af6362030e19a1327030a30774324dea` |
   | `n5-none-ab.reps1/gate.json` | `b8092da595696f3f6c7188479fc3fc637c43e6369c2e0251f88fe4e6bb9041ff` |
   | `n5-none-ab-steady.reps1/baseline.json` | `b270a2e7d8fb35db8e8ec9ac4bf48e3d26a62b9cc7e9e470a7d80e234d945b84` |
   | `n5-none-ab-steady.reps1/candidate.json` | `ad1c85d0aed1a9921ab73cb9cdb31890568d2ce483bf93312d66b86dc8191207` |
   | `n5-none-ab-steady.reps1/gate.json` | `707942a96d594ad4e07da539e74342218b33f522d114c8dcc3509fbde8170bd0` |

### Stage 2 §3b: L1, native CUDA against CuPy

The runbook command ran unchanged three times, back to back. Each run holds
96 cases: native CUDA and CuPy, each with host and device input, over four
shapes, two dtypes, and cold `fit`, cold `partial_fit` and steady
`partial_fit`. Every case converged. Each run also lists 24 skips, all
"fit resets the estimator; steady state is partial_fit only". The runs took
place 16:42–16:50 (+08:00) on 2026-09-29, with Wallpaper Engine closed and
OpenAI Codex (`ChatGPT.exe`, `codex-computer-use-swift.exe`), Chrome,
Discord, Obsidian, NVIDIA App and NVIDIA Overlay running.

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
cold `fit` of the streaming shape and of reference float64. The two
comparisons with no measurable difference are both host-input streaming cold
`fit`: float32 at 1.055–1.071 and float64 at 0.902–0.978.

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

Since #50, `validate_record` accepts native CUDA L1 records whose recorded
`native_cuda_abi` shows ABI 2 or later or lists `l1` in
`supported_penalties`. It accepts all three committed runs (ABI 2 / API 4).

### Decision: native CUDA L1 stays explicit opt-in

**Native CUDA L1 stays explicit opt-in; `backend="auto"` never selects it.**

1. The condition this plan and the runbook set for considering more than
   opt-in, a repeatable same-transport advantage over CuPy, is met on this
   host: 46 of 48 comparisons, two with no measurable difference, none worse.
   That is recorded here as evidence.
2. It still does not change selection. Frozen decision 7 and this plan's
   scope exclude automatic CUDA selection, and `auto` does not select native
   CUDA even for `penalty="none"`, where native CUDA also beats CuPy (0.35–0.86
   in the accepted §3a recapture). Automatic CUDA selection is a question independent of
   the penalty. It needs its own RFC, as the CPU path had in the
   [CPU auto-dispatch RFC](cpu-auto-dispatch-rfc.md), with evidence from more
   than one host.
3. The `penalty="none"` A/B is now accepted (no measurable difference
   between `fca7b83` and `10fc363`). That removes one earlier reason for
   caution, but not reasons 2 and 4, so the decision stands.
4. The float32 L1 trajectories differ between engines: the results are
   numerically equivalent, but the iteration counts are not. Switching engines
   silently would change the diagnostics users see.

## Still open

Resolved since the first run:

- the §3a acceptance, by #50 (`--allow-native-version-change`), #51
  (`--freeze-sample-repetitions`) and the accepted recapture above;
- consuming the L1 baseline, by #50's conditional native CUDA L1 rule in
  `validate_record`.

Still open:

- **PyTorch CUDA DLPack** must pass on the fixed host before 0.7.0: install a
  CUDA 12.9 PyTorch build into the venv and rerun the `cuda` profile
  (`docs/gpu-host-runbook.md`, stage 1). Stage 1 skipped it because PyTorch
  was not installed.
- **TensorFlow CUDA DLPack** is released as unverified on a device. TensorFlow
  has no GPU support on native Windows since 2.11, so the Windows host cannot
  run it; it needs a WSL2 or Linux GPU host. The support matrix says so.

Resolved after the evidence PR: the offline dispatch advisor
(`scripts/benchmarks/dispatch_policy.py`) still recommends native CUDA only
for `penalty == "none"`, which matches the opt-in decision above, and its
comment now gives that reason instead of the pre-ABI 2 "does not implement
L1"; the shape-sweep CLI docstring no longer names a native CUDA L1 skip that
ABI 2 removed.
