# Native penalty completion plan

- Status: Ready for implementation
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
- hashes for v1 and v2 in the hand-off evidence.

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

## Required implementation hand-off

Claude Code should work in a dedicated worktree below
`/renewable-huber/build/workspaces/`, see the whole repository, and modify only
that worktree. It must not commit, push, tag, publish, alter GitHub settings, or
change the `v0.6.1` release. The hand-off must include:

- base SHA and complete changed-file list;
- contract/ABI/API migration explanation;
- exact verification commands and exit results;
- v1/v2 corpus hashes;
- machine-readable local GPU benchmark and environment artifacts;
- explicit unresolved risks and any acceptance criterion not met.

Codex retains architecture review, code acceptance, commits, pushes, pull
requests, and release decisions.
