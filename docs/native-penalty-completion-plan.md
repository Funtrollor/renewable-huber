# Native penalty completion plan

- Status: N0–N4 implemented; GPU-host acceptance and N5 measurements pending
  (see "Implementation status" at the end)
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

Still open, because they need the fixed GPU host:

- run the `cuda` profile and `ctest`; nothing in this implementation has
  executed on a GPU yet. It has only been compiled with CUDA 12.9 (locally for
  SM 120; the PR CI `native-cuda-linux-wheel` job builds SM 75 and 120);
- N5 measurements: the shape sweep and profiler now accept `penalty="l1"` for
  native CUDA, but no fixed-host L1 record or `none` regression A/B exists.
