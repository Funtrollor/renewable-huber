# Claude Code / Codex hand-off

This is the durable, repository-local message channel between the two agents.
`AGENTS.md` contains invariants; this file contains changing work state.

## Ownership

| Responsibility | Owner |
|---|---|
| Architecture, acceptance criteria, code review and engineering plan | Codex |
| Implementation from the accepted plan | Claude Code |
| Commit, push, pull request and release mutations | Codex only |
| Final correctness/performance acceptance | Codex |

Claude Code must leave its work uncommitted, or on a local branch/worktree that
Codex can review. It must never push. Codex must not publish a change it has not
reviewed and verified.

## Required entry format

Append newest entries at the top of **Log**:

```text
### YYYY-MM-DD HH:MM TZ — Agent — short topic
- Base SHA / branch:
- Scope and decisions:
- Files changed:
- Verification run:
- Known risks or unresolved questions:
- Requested next action / owner:
```

Before acting on an entry, compare its base SHA with `git rev-parse HEAD` and
inspect intervening commits. Agents must use separate worktrees when operating
concurrently; a message in this file is not a lock.

## Log

### 2026-08-09 — Codex — hosted release runner compatibility

- Base SHA / branch: `06edd37cb5fc51b497b883006ea8a0011c98a5d0` on
  `agent/release-cuda-vs2022`.
- Scope and decisions: the first hosted CUDA build-only rehearsal proved that
  CUDA 12.9 installed correctly, then failed all three wheels because
  `windows-latest` had migrated to Visual Studio 2026. Pin CUDA compilation to
  `windows-2022`/Visual Studio 2022 instead of overriding nvcc's supported-host
  guard. Replace retired `macos-13` with `macos-15-intel`, and use `macos-15`
  for Apple Silicon; the 15-wheel CPU matrix is unchanged.
- Files changed: `.github/workflows/release.yml`, `CHANGELOG.md`,
  `docs/release-process.md`, this hand-off, and release workflow contract tests.
- Verification run: release metadata tests **6/6**, required core profile
  **297 / 3 skips**, Ruff check/format and `git diff --check` pass. A complete
  GitHub build-only rehearsal from merged `main` remains required.
- Known risks or unresolved questions: hosted CUDA jobs compile and inspect
  wheels without a GPU; local RTX correctness/performance evidence remains the
  runtime release gate.
- Requested next action / owner: Codex validates, publishes the small runner-fix
  PR, reruns all 20 release artifacts, and only then creates `v0.6.1`.

### 2026-08-09 — Codex — 0.6.1 release candidate and native dependency hardening

- Base SHA / branch: `b29b394d46de2c6f4b4382e898978ddddc6a12f0` on
  `agent/release-0.6.1`; release worktree is
  `build/workspaces/codex-release-0.6.1`.
- Scope and decisions: release as `0.6.1` because the immutable `v0.6.0` tag
  belongs to an incomplete historical run. Base, Rust workspace, CPU plugin
  and CUDA plugin now share 0.6.1. PyO3/rust-numpy move to 0.29, MSRV to Rust
  1.83, deprecated GIL release calls to `Python::detach`, and both modules
  explicitly declare `gil_used=true`; CUDA C ABI 1, Python APIs 2/3,
  algorithms, streams and checkpoint v2 are unchanged. Release workflow gains
  required profiles, base sdist smoke, exact-main enforcement and a manual
  full-matrix build-only mode; publishing remains tag-only. Native CPU/CUDA
  GitHub environments were created with the Funtrollor reviewer and `v*` tag
  policy. Public documentation, security/citation metadata, community templates
  and dependency monitoring were synchronized for 0.6.1.
- Files changed: version and native dependency metadata; two PyO3 bindings;
  release/TestPyPI/CI workflows; release artifact validator and tests; all
  current public API, architecture, support, performance, release, governance
  and native distribution documentation. Historical hand-off entries and raw
  benchmark data were not rewritten.
- Verification run: Python discover **410 tests / 43 skips**, required core
  **297 / 3**, performance **54**, native-cpu **19**, and local RTX 5070 Ti
  CUDA **35 / 2** all pass. Ruff check/format, profile membership, four-case
  golden corpus, source metadata, C++ ABI syntax, Rust fmt/clippy/check and 14
  scoped Rust tests pass. CUDA 12.9 CTest is **1/1**. Base wheel/sdist and local
  CPython 3.12 CPU/CUDA candidate wheels pass Twine, clean install, `pip check`
  and runtime smoke; engine versions report 0.6.1. Local environment: Python
  3.12.3, NumPy 2.5.1, RTX 5070 Ti, driver 596.49, CUDA 12.9. Candidate hashes:
  base wheel `57d9d219...efdd430`, sdist `98d5a74d...374446`, CPU wheel
  `3ce0cf0e...8b163`, CUDA wheel `70cfdd99...ac2cf6`.
- Known risks or unresolved questions: local native wheels are WSL development
  artifacts, not the manylinux/Windows release set. The manual `release.yml`
  build-only run must produce and validate 1 base wheel, 1 sdist, 15 CPU wheels
  and 3 Windows CUDA wheels from merged `main`. PyPI pending-publisher settings
  are account-side and become observable only when OIDC publish runs; do not
  move the old `v0.6.0` tag or retry already-uploaded distributions.
- Requested next action / owner: Codex commits, pushes and opens the release PR;
  after CI/merge, Codex runs the full build-only workflow on exact `main`, tags
  `v0.6.1`, validates release artifacts, and completes staged PyPI approvals.

### 2026-08-09 — Codex — CPU auto-dispatch acceptance

- Base SHA / branch: `a6e8a250bc103996a8d4c634261ea4540517a1e9`, reviewed in
  `build/workspaces/claude-cpu-auto-dispatch` on `claude/cpu-auto-dispatch`.
- Scope and decisions: **accepted after correction.** CPU `auto` uses a
  process-local measured ratio surface rather than a CPU/shape lookup table.
  Codex removed cross-estimator hysteresis, corrected the soft-deadline and
  uncertainty claims, made fork/context invalidation conservative, added full
  affinity plus optional effective BLAS/OpenMP thread-pool signatures, widened
  auto-selected native failure fallback to ordinary exceptions, and corrected
  the benchmark's data-volume, ordering, pairing and cold/steady contracts.
  Explicit CPU backends, CUDA paths, algorithms and checkpoint format are
  unchanged. `auto_dispatch_` is accepted as fitted observability state.
- Files changed: four new files (`cpu_dispatch.py`, its RFC, 102 portable tests,
  and the auto benchmark) plus estimator integration, benchmark contract tests,
  unittest profile membership and public/maintainer documentation. Claude Code
  wrote the initial implementation and made no Git mutation; Codex owns the
  reviewed corrections and publication.
- Verification run: Python discover/all **409 tests, 43 skips**; core **296/3**;
  performance **54**; native-cpu **19**; ruff check/format, four-case golden
  corpus, release metadata and C++ ABI all pass. Rust fmt/clippy/check and 14
  Rust tests passed earlier in the same reviewed tree; no Rust file changed
  afterward. Local CPU benchmark artifact
  `artifacts/auto-dispatch/auto-dispatch-cpu-final.json` (gitignored), SHA-256
  `67960356bf1cb11b75cc7b632a8ec9beb82054f2ffd0704fe9af56d19883b28b`.
- Known risks or unresolved questions: this WSL host never selected native via
  auto. It safely declined slower native results, but also declined a measured
  ~1.14x native win for the one-million-row fit because it did not clear the
  15% margin and was far outside the probe row range. Small streaming batches
  do not accumulate toward calibration; both are documented conservative false
  negatives and future tuning work, not wrong slower-engine selections.
- Requested next action / owner: GitHub CLI authentication has been restored.
  Codex stages the reviewed scope, commits, pushes and opens the draft PR;
  merge remains the user's decision. PyPI remains out of scope.

### 2026-08-09 — Claude Code — generalized CPU crossover and runtime auto-dispatch

- Base SHA / branch: `a6e8a250bc103996a8d4c634261ea4540517a1e9`, worktree
  `build/workspaces/claude-cpu-auto-dispatch` on `claude/cpu-auto-dispatch`.
  The tree was clean at that SHA before editing and **nothing is committed or
  pushed**; HEAD is unchanged, the index is clean, `git diff --check` is clean,
  and no remote, release, tag or GPU workflow was touched. This closes the P3
  follow-up "tune or dispatch around the 8192x32 native-CPU crossover".

#### Scope and decisions

`backend="auto"` on CPU may now select `native_cpu`. The accepted design is
[`docs/cpu-auto-dispatch-rfc.md`](cpu-auto-dispatch-rfc.md).

**1. The decision is a fitted ratio surface with a conservative uncertainty
allowance, not a table.** `src/renewable_huber/backends/cpu_dispatch.py` fits
`log(native/NumPy) = b0 + b1 log n + b2 log p` over five probe shapes, centred
on the probe design, and dispatches on `predicted + 2 * se` where
`se = s * sqrt(1 + leverage)`. Fitting the ratio rather than two cost curves
makes everything common to both engines cancel and makes the coefficients
dimensionless; judging on the upper estimate makes noisy probes, a
mis-specified surface and a distant shape all push the same way, towards NumPy.
The leverage term is the whole extrapolation policy: a uniform 0.70 ratio
clears the 15% entry margin next to the probes and stops clearing it 1,000x
further out, on the same host.

`2 * se` is a **blunt conservative allowance and is not described as a
confidence interval anywhere**. With three coefficients fitted from five
probes there are two residual degrees of freedom, at which a normal quantile
means nothing (the one-sided Student-t 95% point is 2.92, so the "95% upper
bound" an earlier draft of this entry claimed was both unjustified and *less*
conservative than the claim), and the residual-scale floor usually dominates
`s` anyway. The policy field is `uncertainty_multiplier`.

Probe timing is a **paired interleaved comparison**: each round runs both
engines and the rounds alternate which goes first. A fixed order hands the
first-touch and pool-spin-up cost to whichever engine leads, on every probe;
NumPy-first — what the first draft did — therefore biased every ratio towards
native.

No CPU brand or model string is read, and nothing is written to disk.
`PolicyInputTests` parses the module and fails on `platform.`, `uname`,
`cpuinfo`, a brand string, an `open(`, an import of
`json`/`pathlib`/`pickle`/`shelve`/`sqlite3`/`tempfile`/`sys`, or any use of
`os` outside `{register_at_fork, sched_getaffinity, cpu_count, environ}`,
reached by a dot or by `getattr`.

**2. Bounded cost.** Probes never touch the caller's data: they are a fixed
ladder of five small shapes (1,024-8,192 rows, 7-33 parameters) run at
`max_iter=3`, one untimed warmup and two timed rounds each, minimum taken.

The **hard bound is the ladder** — 2.01e8 work units, a product of policy
constants fixed before anything is measured, which no host can exceed.
`probe_start_deadline_seconds = 0.25` is a **soft** deadline and is now named
as one: the clock is checked only *between* probes, so a probe that has started
always runs to completion and a calibration can overrun. The first draft of
this entry called it a hard deadline, which the code never implemented.

Two work gates decide when the ladder is spent:

- `minimum_work_units = 5.0e4` (the smallest probe) -- below it, NumPy without
  probing or even importing the extension;
- `calibration_work_units = 1.5e8` -- a *first* calibration is only started for
  a batch at least this large, so the ladder costs at most **1.34
  single-iteration equivalents of the batch that triggers it**.

They are separate on purpose: acquiring evidence needs a batch big enough to
amortise it, using evidence already cached does not, so a process that has
calibrated dispatches batches 3,000x smaller than the one that paid.

**2b. A measurement is valid only for its execution context.** The cache key
carries a `RuntimeSignature`: the sorted CPU affinity mask itself where
`sched_getaffinity` exists, the usable CPU count (`cpu_count` fallback, or a
recorded `"unavailable"`), the seven `*_NUM_THREADS` variables, and effective
BLAS/OpenMP pool sizes when optional `threadpoolctl` is available. The latter
catches scikit-learn/joblib limits that do not modify the environment. The mask
rather than its length, so re-pinning to a *different set of the same size* —
another socket, with different cache sharing and memory locality — invalidates
rather than silently reuses. A changed signature **deletes** the stale entry;
`fork` empties the whole map in the child as well as rebuilding the locks,
because a joblib worker is routinely pinned to a subset of the parent's cores.

The extension's own `parallel_threads` is deliberately **not** in the
signature. It is not independent — `n_threads` already carries the requested
pool and the mask plus `RAYON_NUM_THREADS` carry what it gets — and it is
unknown until something imports the extension, which is what calibration does:
including it made the first calibration invalidate itself and be paid twice.

**3. The dispatch point is the estimator, not `resolve_backend`.**
`resolve_backend("auto")` has no workload and is unchanged; the estimator
resolves NumPy, validates and prepares the batch, then may swap in the native
engine. **That swap is only sound because both CPU engines see identical
arrays**, so `BackendSwapSafetyTests` pins that `NativeCpuBackend` still
inherits `asarray`/`copy`/`reshape`/`to_numpy`/`scalar`/`xp` from
`NumPyBackend` and declares no `native_design_matrix`. It is a new entry in
`AGENTS.md`'s silent-breakage list.

**4. Once per stream, including after a restore.** The decision is made on the
first batch, or the first batch after a checkpoint restore, and never
mid-stream. `load()` itself never calibrates -- there is no shape yet -- and
the checkpoint format is untouched: `backend="auto"` is what is stored, so a
restored model re-decides on its own host rather than inheriting the saving
host's answer.

**5. A backend the policy chose may be withdrawn on any ordinary exception; one
the caller named may not.** If the native engine cannot be constructed after
calibration, or raises **anything under `Exception`** on its first update,
`partial_fit` falls back to NumPy, names the exception type in
`auto_dispatch_["reason"]` and produces the same coefficients a NumPy fit
would. Catching only `BackendUnavailableError`, as the first draft did, let a
Rust panic surfacing as `RuntimeError`, a `MemoryError`, an `OSError` from a
missing `libgomp` or a PyO3 `TypeError` reach a caller who asked for `auto`.
`KeyboardInterrupt` and `SystemExit` still propagate, and an explicit
`backend="native_cpu"` failure still raises whatever it raised. Both halves are
exercised over the same eight exception types.

**5b. Decisions are independent.** `select_cpu_backend` is a pure function of
shape, policy and cached measurement. The first draft kept a per-key
`sticky_native` flag and judged later decisions against a looser hold band, so
one estimator's answer depended on what an unrelated estimator had asked
earlier in the process — invisible to callers and a real hazard under
`GridSearchCV`. **The hysteresis band is gone**; every native selection clears
`1 - enter_margin` on its own evidence, and permuting the order of questions
across a shared cache provably changes nothing.

**6. New fitted attribute `auto_dispatch_`** (JSON-compatible: chosen backend,
reason in words, work units, whether a host model was used, predicted ratio and
its upper bound, calibration seconds, probe count). Present only when the
policy ran, the same convention `cuda_features_` already uses; cleared by
`reset()` and `set_params()`. This is the one public-surface addition and is
yours to accept or reject. `get_params()["backend"]` stays `"auto"`.

**7. `scripts/benchmarks/dispatch_policy.py` is unchanged.** It answers a
different question -- audit a *recorded* measurement against one exact workload
-- and sharing an abstraction would have meant giving the runtime policy a
schema-validated record format it has no use for. If a third consumer appears,
the thing to extract is "compare two timings under a stated margin", not the
record schema.

#### Files changed

Added (4): `docs/cpu-auto-dispatch-rfc.md`,
`src/renewable_huber/backends/cpu_dispatch.py`,
`tests/test_cpu_auto_dispatch.py`,
`scripts/benchmarks/benchmark_auto_dispatch.py`.

Modified (11): `AGENTS.md`, `CHANGELOG.md`, `README.md`, `docs/api.md`,
`docs/architecture.md`, `docs/support-matrix.md`, `scripts/run_test_profile.py`,
`src/renewable_huber/backends/__init__.py`, `src/renewable_huber/cli.py`,
`src/renewable_huber/estimator.py`, `tests/test_benchmark_performance_policy.py`.

No numerical algorithm, native kernel, Rust crate, GPU path, checkpoint format,
benchmark schema-v2 record or public constructor parameter was changed. No GPU
CI job was added or altered.

#### Tests

`tests/test_cpu_auto_dispatch.py`, **102 tests**, in the `core` profile and in
`PORTABLE_NATIVE_MODULES` so `validate_profiles` fails if anyone moves it out of
CPU CI. Everything is driven through an injected clock and probe runner, so it
needs no Rust extension and is deterministic on any machine:

- **simulated hosts** -- uniform 0.25/0.70/0.95/2.5 ratios; a slow-NumPy and a
  fast-NumPy host 10,000x apart in absolute speed decide identically, because
  only the ratio is used; a host whose crossover sits inside the probe range
  answers `native` tall-and-narrow and `numpy` short-and-wide from one
  calibration;
- **generalisation** -- exact power-law hosts are reproduced to 6 decimal
  places in log space at four shapes no probe visited; the cache is asserted to
  contain the five probe shapes and *not* the queried shape;
- **uncertainty** -- standard error grows monotonically with distance; far
  extrapolation declines a win it accepts near the probes; a non-log-linear host
  widens the band instead of predicting confidently;
- **decision independence** -- every native choice clears the same entry
  margin; repeated and permuted shape queries cannot change another estimator's
  answer, and no selection state exists in the cache;
- **budget** -- exhausted budget refuses to dispatch; a four-probe partial
  ladder still fits; calibration seconds are reported;
- **failure** -- unavailable extension, raising setup, raising probe, a clock
  too coarse to resolve a probe, a degenerate ladder, and that a failure is
  cached so it is paid for once;
- **cache/context** -- 8 threads on one key calibrate exactly once; dtype,
  penalty and native thread counts are separate; affinity, environment and
  effective BLAS thread-pool changes invalidate; a fork clears measurements
  and rebuilds locks;
- **estimator** -- explicit backends and `device="cuda"` never consult the
  policy; the policy is consulted once per stream; it sees the design width, not
  the feature count; a native selection is applied; construction failure and
  any ordinary first-update exception both fall back and match NumPy exactly
  while an explicit native failure still raises;
- **checkpoint** -- `load()` never calibrates, a restored stream dispatches on
  its next batch, an explicit override on load is not governed, and the archive
  still records `backend="auto"` with no dispatch data in it.

A real `os.fork()` under a thread contending on the cache lock was also
exercised by hand: the child completed a fit and exited 0.

#### Benchmark evidence

`scripts/benchmarks/benchmark_auto_dispatch.py` measures `numpy`,
`native_cpu`, `auto_cold` and `auto_warm`. A `fit` consumes the full dataset;
`stream` consumes the declared batches. Engine order alternates forward/reverse
and repeat counts must be even. Regret is the median aligned per-round ratio to
the best explicit engine, not a ratio of independent medians. Twelve portable
contract tests pin the data volume, deciding work units, order and statistic.

```bash
.venv/bin/python scripts/benchmarks/benchmark_auto_dispatch.py \
  --profile both --operation both --repeats 8 --warmup 1 \
  --output artifacts/auto-dispatch/auto-dispatch-cpu-final.json
```

| SHA-256 | File |
|---|---|
| `67960356bf1cb11b75cc7b632a8ec9beb82054f2ffd0704fe9af56d19883b28b` | `artifacts/auto-dispatch/auto-dispatch-cpu-final.json` |

`artifacts/` is gitignored; delete the directory after acceptance. Medians in
ms, float64, `penalty="none"`, this WSL2 host:

| case | numpy | native | auto (warm) | chose | regret | calibration |
|---|---:|---:|---:|---|---:|---:|
| latency-smoke 2,048x8 fit | 0.365 | **0.216** | 0.305 | numpy | 1.391 | -- |
| latency-smoke 2,048x8 stream | 0.388 | **0.183** | 0.421 | numpy | 2.177 | -- |
| reference-smoke 8,192x32 fit | **2.678** | 3.360 | 2.646 | numpy | 1.005 | -- |
| reference-smoke 8,192x32 stream | **2.062** | 2.653 | 2.145 | numpy | 1.028 | -- |
| latency 4,096x16 fit | 1.516 | **1.331** | 1.545 | numpy | 1.158 | -- |
| latency 4,096x16 stream | 1.535 | **1.501** | 1.521 | numpy | 1.021 | -- |
| reference 100,000x90 fit | **163.8** | 255.5 | 164.2 | numpy | 1.003 | 34.2 ms |
| reference 100,000x90 stream | **103.6** | 232.9 | 97.2 | numpy | 0.929 | 38.9 ms |
| wide 16,384x256 fit | **129.8** | 212.5 | 114.8 | numpy | 0.948 | 51.5 ms |
| wide 16,384x256 stream | **100.4** | 256.9 | 93.5 | numpy | 1.036 | 43.4 ms |
| streaming 1,000,000x32 fit | 557.7 | **489.7** | 541.3 | numpy | 1.122 | 32.9 ms |
| streaming 1,000,000x32 stream | **285.5** | 475.8 | 265.5 | numpy | 0.928 | -- |

No slower native engine was selected. The conservative policy intentionally
missed the one-million-row native win (about 1.14x, below the 15% entry margin
and far outside the probe row range) and the small wins below the calibration
gate. Calibration was 32.9-51.5 ms. Cold calibration is measured in a separate
phase; same-backend aligned ratios on standard cases are 0.928-1.036, inside
the retained WSL noise band.

#### Verification run

WSL2 Ubuntu 24.04, Python 3.12.3, NumPy 2.5.1, shared venv
`/home/untrollor/renewable-huber/.venv` used read-only with
`PYTHONPATH=<worktree>/src` so the worktree source shadows the editable install.

| Gate | Result |
|---|---|
| `python -m unittest discover -s tests` | **409 tests, 43 skips, OK** |
| `run_test_profile.py --check` | 6 profiles cover 24 modules |
| `core` | **296 tests, 3 skips, exit 0** |
| `performance` | 54 tests, exit 0 |
| `native-cpu` | 19 tests, exit 0 |
| `all` | 409 tests, 43 skips, exit 0 |
| `optional-cpu` | **exit 2** — pandas/SciPy/scikit-learn/PyTorch/TensorFlow are not in this venv |
| `cuda` | **exit 2** — CuPy is not in this venv |
| `ruff check src tests scripts` | pass |
| `ruff format --check src tests scripts` | pass, 78 files |
| `generate_native_golden.py --check` | 4 cases match |
| `validate_release_artifacts.py --source-only` | consistent for 0.6.0 |
| `g++ -fsyntax-only ... abi_contract.cpp` | pass |
| `cargo fmt --all -- --check` | pass |
| `cargo clippy --locked --workspace --all-targets -- -D warnings` | 0 diagnostics |
| `cargo check --locked --workspace --all-targets` | pass |
| `cargo test --locked -p rh-core -p rh-cpu -p rh-cuda-ffi --all-targets` | 14 tests (2 + 7 + 5) |
| `python -m renewable_huber.cli info` | prints the new CPU policy line |

#### Known risks and unresolved questions

1. **`optional-cpu` and `cuda` could not be executed here.** The shared venv has
   neither the optional CPU stack nor CuPy, so both required profiles correctly
   exit 2. My changes touch no Torch, TensorFlow, CuPy or CUDA path — the CUDA
   device branch is excluded by `auto_cpu_dispatch_applies` and pinned by a test
   — but they have not been *run* against a device. Please run them on the fixed
   GPU host.
2. **No host in reach selects native through auto.** Every capture above chose NumPy, so the
   native-selection path is proven only by simulated hosts and stubs, not by an
   end-to-end run that actually reaches the Rust engine through `auto`. The
   fixed Ryzen runner should exercise it; that is the acceptance evidence I
   cannot produce.
3. **The thresholds are judgement calls.** `calibration_work_units = 1.5e8`,
   `enter_margin = 0.15`, and `uncertainty_multiplier = 2.0` are conservative
   round numbers. The accepted host run exposes their intended trade-off: auto
   never selected a slower native engine, but declined a measured 1.14x native
   win on the one-million-row fit.
4. **A stream of small batches never accumulates its way into a calibration.**
   The decision is made on the first batch, so a workload that would amortise a
   ladder easily over 10,000 batches of 2,048 rows never measures. Re-evaluating
   on cumulative stream work is the obvious extension and was left out because
   it means changing engines mid-stream — a larger contract change than this
   work, and your call.
5. **`auto_dispatch_` is a new fitted attribute** (item 6 above). If you would
   rather keep the public surface identical to `a6e8a25`, it can become
   `_auto_dispatch`; the tests reference it in one place each.
6. **The probe ladder is fixed at 1,024-8,192 rows and 7-33 parameters.** A host
   whose crossover lies far outside that box is answered by extrapolation with a
   wide band. Widening the ladder costs calibration time; it is a tuning
   decision, not a design one.

#### Requested next action / owner

Codex accepted `auto_dispatch_`, completed the architecture corrections and
local CPU benchmark, and owns final verification, commit, push and PR. GPU code
and CI are outside this CPU-only change; Claude Code performed no Git mutation.

### 2026-08-09 — Codex — P3 final acceptance and close-out

- Base SHA / branch: `14f9e72aae0f75c4f84923d61952438861ab1777`,
  reviewed from `claude/maintainability-p3`; Claude Code made no Git mutation.
- Scope and decisions: **P3 accepted.** The codec/estimator boundary, required
  unittest profiles, portable native-test placement and shape-sweep split meet
  the agreed architecture. `BackendContractError` remains an internal deep
  import rather than a new top-level export. Removing the undocumented
  `serialization.save_model` / `load_model` helpers is accepted and recorded in
  `CHANGELOG.md`; estimator `save()` / `load()` and checkpoint v2 are unchanged.
- Codex close-out changes: replaced text-based portable-test guards with AST
  semantic checks, allowed future helper classes, and added a GPU-hidden run of
  all nine CUDA selection test IDs that requires zero skips/errors. Corrected
  the benchmark evidence wording (28 fields present, 22 non-volatile fields
  compared), removed an unsupported reference to unretained self-paired data,
  and synchronized `AGENTS.md`, this report and the maintainability report to
  the accepted P3 state.
- Files changed: **28 files, 4603 insertions, 1652 deletions** from
  the base, including all nine added files; measured with a temporary index.
- Verification run: `discover` **295 tests / 26 skips**, `core` **194 / 3
  skips**, `performance` **42**, `native-cpu` **19**, `optional-cpu` **19 / 2
  skips**, and `all` **295** all exit 0. Ruff check/format, four-case golden
  corpus, release metadata, C++ ABI syntax, Cargo fmt/clippy/check and all 14
  scoped Rust tests pass. In this sandbox the required `cuda` profile correctly
  exits 2 because the driver is unavailable; Claude's final real-device run on
  the RTX 5070 Ti passed all **35** CUDA tests and CTest 1/1. The later Codex
  changes touch only CPU test guards and documentation, not a GPU path.
- Performance acceptance: deterministic pre/post schema comparison passes for
  120 cases, 52 skips, four dataset checksums and all 22 non-volatile result
  fields. Nine-round aligned candidate/baseline medians are 0.945 / 1.025 /
  0.997 / 0.927, all within the 1.10 structural-refactor limit. The fixed gate
  remains false only because sub-millisecond cold measurements exceed the 5%
  MAD ceiling and Rust is slower than NumPy at 8192x32; the baseline side
  already exhibits both, so neither is attributed to the module split.
- Known follow-up: isolate Torch and TensorFlow GPU validation into separate
  processes; tune or dispatch around the 8192x32 native-CPU crossover; decide
  whether the portable native-CPU selection tests merit a symmetric module
  split; simplify the still-large shape-sweep CLI orchestrator.
- Requested next action / owner: Codex commits, pushes and opens the P3 pull
  request. Generated `artifacts/p3-review/` remains ignored and may be removed
  after publication.

### 2026-08-09 — Claude Code — P3 review round 2: CI coverage, re-export guard, statistics

- Base SHA / branch: still `14f9e72aae0f75c4f84923d61952438861ab1777`,
  `claude/maintainability-p3`. **Nothing committed or pushed; the index is
  clean.** No algorithm, kernel, stream, checkpoint format or public API was
  touched, no GPU CI job was added, `BackendContractError` is still absent from
  `renewable_huber.__all__`, and the profiles were not extended to Torch or
  TensorFlow GPU.

#### 1. Nine portable CUDA tests are back in CPU CI

`NativeCudaSelectionTests` substitutes a fake extension module in every one of
its nine tests, so it needs no device, driver or built extension. It lived in
`tests/test_native_cuda_backend.py`, which only the `cuda` profile owns, and
`cuda` runs only on the fixed local GPU host — so after P3.2 those tests ran
nowhere in CI and nothing failed to say so.

The class moved verbatim to a new `tests/test_native_cuda_selection.py`
assigned to `core`, rather than adding the whole GPU module to `core` as a
shared module. Cohesion is the reason: the new module is "selection, fallback
and native-call routing, verified with fakes", while what stays behind is
device integration. It also keeps `core` free of the ~26 device-gated tests
that would otherwise have joined it as skips. Seven imports the move orphaned
were removed from the original module.

Enforcement, so this cannot recur silently:

- `PORTABLE_NATIVE_MODULES` in `scripts/run_test_profile.py` names the modules
  that carry native coverage but need no device
  (`test_native_cuda_selection`, `test_native_cuda_contract`,
  `test_native_golden`). `validate_profiles` fails if one of them is owned by
  any profile other than `core`, with a message naming the current owner and
  what CPU CI would lose. It is scoped to tables that actually claim the module,
  so the validator's own fixture tables do not trip on it.
- `test_moving_a_portable_native_module_out_of_core_is_reported` and
  `test_a_portable_native_module_kept_in_core_is_accepted` exercise both
  directions.
- `test_the_portable_native_modules_really_are_portable` fails if one of those
  modules gains a `skipUnless` or `skipTest`, which would make its membership
  in `core` meaningless on a runner without a GPU.
- `test_the_cuda_selection_tests_are_in_core_not_only_in_cuda` pins the class to
  its new module and asserts it is gone from the old one.

Measured, not asserted: the nine tests run — they do not skip — under
`CUDA_VISIBLE_DEVICES=""`, and `core` on the bare `.[dev]` venv that reproduces
the CPU CI baseline job went from 179 to **193 tests, 3 skips, exit 0**. Full
`discover` went from 285 to **294 tests** (nothing lost; +9 from the new
guards). `cuda` correspondingly went 44 -> 35 and is now purely device tests.

#### 2. The 11 re-exports have an automated guard

`tests/test_benchmark_performance_policy.py::ShapeSweepReExportTests` parses the
two real consumers — `scripts/benchmarks/benchmark_native_cpu_scaling.py` and
this test module — for their
`from scripts.benchmarks.benchmark_shape_sweep import (...)` statements, and
asserts every imported name is both in `__all__` and reachable as a module
attribute. Extra names are allowed, so future additions do not fail.

Deriving the list from the consumers is what keeps it current; the literal
`DOCUMENTED` set of 11 remains as an anti-vacuity check, and
`test_the_consumers_still_import_the_documented_names` fails if a consumer stops
importing one, which is the signal to revisit the list rather than let it go
stale. Two further tests assert that everything promised by `__all__` resolves
and that `main` is still exported. It lives in the `performance` profile, next
to the consumer it belongs to, and runs in CPU CI.

#### 3. Corrected statistics

Measured on the final tree through a temporary index, so the real index is never
written:

```bash
TMPIDX=$(mktemp) && GIT_INDEX_FILE=$TMPIDX git read-tree HEAD \
  && GIT_INDEX_FILE=$TMPIDX git add -A . \
  && GIT_INDEX_FILE=$TMPIDX git diff --cached --stat HEAD
```

**27 files changed, 4492 insertions, 1645 deletions.** Plain `git diff --stat`
reports only the 18 tracked paths and undercounts by about 2,700 lines, because
the nine added files are untracked; that is what both earlier entries got wrong.

Added (9): `scripts/run_test_profile.py`,
`scripts/benchmarks/shape_sweep/{shapes,environment,timing,runners,cli}.py`,
`tests/test_checkpoint_payload.py`, `tests/test_profile_runner.py`,
`tests/test_native_cuda_selection.py`.

Modified (18): `.github/workflows/ci.yml`, `AGENTS.md`, `CONTRIBUTING.md`,
`docs/agent-handoff.md`, `docs/gpu-performance.md`,
`docs/maintainability-refactor.md`,
`scripts/benchmarks/benchmark_shape_sweep.py`,
`src/renewable_huber/backends/_dlpack.py`,
`src/renewable_huber/backends/native_cuda_backend.py`,
`src/renewable_huber/estimator.py`, `src/renewable_huber/exceptions.py`,
`src/renewable_huber/serialization.py`, `tests/test_backend_capabilities.py`,
`tests/test_benchmark_performance_policy.py`, `tests/test_dlpack_adapters.py`,
`tests/test_estimator.py`, `tests/test_native_cpu_backend.py`,
`tests/test_native_cuda_backend.py`.

#### Retained benchmark artifacts

Kept in `artifacts/p3-review/` for your review. `artifacts/` is matched by
`.gitignore:21`, confirmed with `git check-ignore -v`, so nothing here is
tracked or untracked-visible; 600 KB total. Delete the directory once acceptance
is done.

Capture commands, run on the final tree:

```bash
SWEEP="--profile smoke --backend all --penalty both --dtype both \
  --lifecycle both --operation both --warmup 1 --repeats 2 \
  --minimum-sample-seconds 0"

# pre-split: the base SHA exported read-only with `git archive` (no commit/tag)
(cd "$BASELINE_TREE" && .venv/bin/python scripts/benchmarks/benchmark_shape_sweep.py \
   $SWEEP --output artifacts/p3-review/shape-sweep-pre-split.json)
.venv/bin/python scripts/benchmarks/benchmark_shape_sweep.py \
   $SWEEP --output artifacts/p3-review/shape-sweep-post-split.json
.venv/bin/python artifacts/p3-review/compare_sweeps.py \
   artifacts/p3-review/shape-sweep-{pre,post}-split.json

.venv/bin/python scripts/benchmarks/run_interleaved_benchmark.py \
  --baseline-python .venv/bin/python --baseline-repo "$BASELINE_TREE" \
  --candidate-python .venv/bin/python --candidate-repo . \
  --output-dir artifacts/p3-review/interleaved --rounds 9 --profile smoke \
  --backend cpu --penalty none --dtype both --lifecycle cold \
  --operation partial-fit --minimum-sample-seconds 0.25 \
  --max-sample-repetitions 8
```

`--minimum-sample-seconds 0` pins the sampling block to 1 so the comparison is
deterministic; `--max-sample-repetitions 8` satisfies the runner's documented
"calibration must not change between rounds" guard.

| SHA-256 | File |
|---|---|
| `51ffca3cc111558a7cc719e1a5f7c4129c2f09f5ee1311e949345931af6d6698` | `shape-sweep-pre-split.json` |
| `bc1cb3ec2daf3280af9b05f8ee2df83cb8a7acb427d43e08f40370ee47e41a29` | `shape-sweep-post-split.json` |
| `bf82015b3e20b76143c454e9a720c077fcef421b7c99021dcd4bf4d21efe52c6` | `shape-sweep-comparison.txt` |
| `c9035943115ea0defcac09cfe1a524fac53daf125d4a3654df39def94c9a139c` | `shape-sweep-pre-split.log` |
| `b2552e8b5c925699ef9ff2e34f310a9439a69bcc11e7d306a2944e7de74bc973` | `shape-sweep-post-split.log` |
| `fde4f2bd61c4afc8ad0cdd184e50a32a7ae26763edb17a46f05b738b5317e4ba` | `interleaved/baseline.json` |
| `13b75626f2ee052f93010a0164d72fbbcd9fb185964817907312b93f7fc111cf` | `interleaved/candidate.json` |
| `c7b8d1368e44d6e24157742951a1347d03e69981493650e2ddc7e154d0a5fcc4` | `interleaved/gate.json` |
| `dae03af456597a0197994af89ed88cbedde79076422cff14d99720307efb7dc0` | `interleaved.log` |
| `c84dd395f9b14397fcbefa513a04a97d0fd4c4959c52f7a38580cd3e6aea4581` | `compare_sweeps.py` |

Summaries. **Schema comparison: PASS** — 120 cases, 52 skipped entries, 4
dataset SHA-256 checksums identical on both sides, 28 result fields present per
case, and all 22 non-volatile result fields equal, including schema name, `schema_version`,
`arguments`, environment keys, lifecycle and sampling metadata, and every skip
reason. **Interleaved A/B**: paired median candidate/baseline slowdowns
**0.945 / 1.025 / 0.997 / 0.927**, all inside the 1.10x CPU threshold. The gate
still reports `passed: false` for the same two absolute reasons documented in
the previous entry — relative MAD 6.6–8.3% against a 5% ceiling on the `smoke`
shapes, and the Rust CPU engine not being faster than NumPy at 8192x32. The
retained baseline already exhibits both absolute failures while every paired
pre/post ratio stays below 1.10; the earlier self-paired control was not
retained and is therefore not used as final evidence. `compare_sweeps.py` is a
scratch comparison tool, retained with the data rather than added to the
repository.

#### Verification run

All on WSL2 Ubuntu 24.04, Python 3.12.3, NumPy 2.5.1, Rust 1.97.1, CUDA 12.9,
RTX 5070 Ti (driver 596.49).

| Gate | Result |
|---|---|
| `python -m unittest discover -s tests` | **294 tests, OK** (was 285; none lost) |
| `run_test_profile.py --check` | 6 profiles cover 23 modules |
| `core` | 193 tests, exit 0 |
| `performance` | 42 tests, exit 0 |
| `native-cpu` | 19 tests, exit 0 |
| `optional-cpu` | 19 tests, 2 skips, exit 0 |
| `cuda` (real device) | **35 tests, exit 0** |
| `all` (real device) | **294 tests, exit 0** |
| `ruff check` / `ruff format --check` | pass, 75 files |
| `generate_native_golden.py --check` | 4 cases match |
| `validate_release_artifacts.py --source-only` | consistent for 0.6.0 |
| `g++ -fsyntax-only ... abi_contract.cpp` | pass |
| `cargo fmt --all -- --check` | pass |
| `cargo clippy --locked --workspace --all-targets -- -D warnings` | 0 diagnostics |
| `cargo check --locked --workspace --all-targets` | pass |
| `cargo test --locked -p rh-core -p rh-cpu -p rh-cuda-ffi --all-targets` | 14 tests |
| `cmake --build build/static && ctest` | 1/1 passed |

Bare `.[dev]` venv reproducing the CPU CI baseline job: `core` **193 exit 0**,
`performance` 42 exit 0, `all` 294 exit 0, `native-cpu` **exit 2** (extension
not importable), `optional-cpu` **exit 2** (pandas not installed).

While adding the portable-module check I broke `validate_profiles` for the
validator's own fixture tables (`KeyError: 'core'`); the bare-venv run caught it
before it could reach you, and the check is now scoped to tables that claim the
module.

#### Known risks and unresolved questions

1. `tests/test_native_cpu_backend.py::NativeCpuSelectionTests` is portable in
   the same way but was **left where it is**: `native-cpu` is a required profile
   that CI actually runs on nine OS/Python combinations, so nothing is currently
   lost. It is not in `PORTABLE_NATIVE_MODULES` because that module also holds
   device-dependent tests and cannot live in `core`. If you want symmetry, the
   same extraction applies.
2. Torch and TensorFlow GPU paths remain out of scope, as instructed. A future
   `optional-gpu` profile would need to solve the in-process CUDA coexistence
   problem first — most likely one module per subprocess — which is why it was
   not attempted here.
3. `BackendContractError` is still not in `renewable_huber.__all__`, keeping the
   top-level surface identical to `14f9e72`. Every other exception in that
   module is re-exported.
4. `release.yml` and `test-pypi.yml` still use plain `discover`.
5. `artifacts/p3-review/` should be deleted after acceptance.

#### Requested next action / owner

Codex performs final acceptance, commit, push and PR. Claude Code performed no
Git mutation: HEAD is `14f9e72`, the index is empty, `git diff --check` is
clean, and no generated artifact is tracked.

### 2026-08-09 — Claude Code — P3 review round: four acceptance fixes

- Base SHA / branch: still `14f9e72aae0f75c4f84923d61952438861ab1777`,
  `claude/maintainability-p3`. The accepted P3 architecture is unchanged; this
  entry records only the four items raised in Codex's initial review, plus the
  corrected file inventory. **Nothing committed or pushed.**

#### 1. The serialization boundary no longer pins the module's contents

`test_the_parsed_module_is_the_one_under_test` asserted set *equality* against
exactly three names, so any future private codec helper would have failed a test
about estimator leakage. It now asserts that the three names are present and are
the right kind of node — a class and two functions — which is all the
anti-vacuity guard needs. `test_a_new_private_helper_does_not_break_the_boundary_checks`
parses the module with a `_pack_metadata` helper appended and shows the relaxed
form still holds. The leakage checks themselves are untouched.

#### 2. Native capability tests now run in the required native profiles

`NativeBackendCapabilityTests` lived in `tests/test_backend_capabilities.py`,
which is a `core` module; without an extension its two tests could only call
`skipTest`, which is the outcome a required profile exists to forbid. They moved
to the modules those profiles already own:

- `tests/test_native_cpu_backend.py::NativeCpuCapabilityTests`, gated by the
  module's existing `_native_cpu_ready()`, so the `native-cpu` profile runs it;
- `tests/test_native_cuda_backend.py::NativeCudaCapabilityTests`, gated by
  `_native_cuda_ready()`, so the `cuda` profile runs it.

Both now call `resolve_backend` directly instead of swallowing
`BackendUnavailableError` into a skip: under a required profile an unavailable
backend is a failure, not an absence. `tests/test_backend_capabilities.py` is
left purely portable (its `resolve_backend` import went with the class), and
`test_native_capability_tests_sit_in_the_required_native_profiles` asserts the
placement, including that the portable module no longer defines the class and
that `LiveAccessorTests` is still there so the check cannot pass vacuously.
Verified executing, not skipping: `native-cpu` 18 -> 19 tests, `cuda` 40 -> 44.

#### 3. `optional-cpu` is CPU-only by construction

`Profile` gained an `environment` field, and `optional-cpu` declares
`CUDA_VISIBLE_DEVICES=""`. `run_profile` applies it around **both** suite
construction and execution — a framework that reads the variable while being
imported has to see it — through `forced_environment`, which restores the
previous value (including "was unset") on the way out, because `run_profile` is
also called in-process by its own tests. `--list` prints the forced variables
and the runner logs a `note:` line, so the behaviour is never silent.

Every other profile declares an empty environment, so `cuda` still sees the
device. Five self-tests cover this: which profiles force what, restoration when
the variable was unset, when it was set, after an exception, and that the suite
is built inside the mask. Documented in `CONTRIBUTING.md` and `AGENTS.md`.

Result on this GPU host, with no external environment variable:
`optional-cpu` 19 tests, 2 documented skips, exit 0 — previously it failed with
`cusolverDnCreate` returning `CUSOLVER_STATUS_INTERNAL_ERROR`. `cuda` and `all`
are unaffected. CI runners have no GPU and see no change.

#### 4. The DLPack dtype contract error is fixed at the type level

Root cause: `_validate_features` translated *every* unrecognised `TypeError`
from `backend.asarray` into scikit-learn's coercion message, and every
`ValueError` into the shape message. The native CUDA backend's own refusals were
raised as bare `TypeError`/`ValidationError`, so they were rewritten and the real
reason survived only as `__cause__`.

Fixed by exception type, with no string inspection anywhere:

- new `renewable_huber.exceptions.BackendContractError(RenewableHuberError,
  TypeError)`. It subclasses `TypeError`, so every existing handler still
  catches it and no caller contract changes;
- `native_cuda_backend.py` (3 sites) and `_dlpack.py` (1 site) raise it instead
  of a bare `TypeError`;
- `_validate_features`, `_validate_target` and `_validate_sample_weight` each
  re-raise `BackendContractError` and `ValidationError` unchanged before the
  translating handlers. The translating handlers themselves are untouched, so
  the scikit-learn-compatible messages still appear for genuine coercion
  failures.

The `ValidationError` half was a second, previously invisible instance of the
same defect: the C-contiguity check in the same GPU test was being rewritten
into "X must be a two-dimensional numeric array". Both halves now pass.

Regression coverage that runs on CPU CI:

- `tests/test_estimator.py::BackendErrorPropagationTests` (8 tests) drives a
  stub backend through all three validators and through `fit`, asserts the type
  hierarchy, and asserts that unrecognised `TypeError`/`ValueError` still get
  the documented messages. It also pins the two real coercion paths: NumPy
  rejects a string array with `ValueError` and an object array with
  `TypeError`, so they land in different handlers — behaviour confirmed
  identical on the pristine base tree.
- `tests/test_dlpack_adapters.py::DeviceInputContractErrorTests` (3 tests)
  parses `_dlpack.py` and `native_cuda_backend.py` and fails if either raises a
  bare `TypeError`, with an anti-vacuity check on the parse. Parsing rather
  than importing is what lets this run without the extension.

Mutation-tested: reverting one `BackendContractError` to `TypeError` fails the
CPU-only AST guard *and* the GPU test; restoring it makes both pass.

`renewable_huber.exceptions.BackendContractError` is deliberately **not** added
to `renewable_huber.__all__`, keeping the top-level public surface identical to
`14f9e72`. Every other exception in that module is re-exported, so promoting it
is a public-surface decision left to you.

#### Corrected file inventory

25 files, 3690 insertions, 1260 deletions across tracked and untracked paths,
measured through a temporary index so the real one stays untouched:

```bash
TMPIDX=$(mktemp) && GIT_INDEX_FILE=$TMPIDX git read-tree HEAD \
  && GIT_INDEX_FILE=$TMPIDX git add -A . \
  && GIT_INDEX_FILE=$TMPIDX git diff --cached --stat HEAD
```

Plain `git diff --stat` reports 17 files and undercounts by roughly 2,200 lines,
because the eight added files are untracked. That is what the previous entry
got wrong.

Modified (17): `.github/workflows/ci.yml`, `AGENTS.md`, `CONTRIBUTING.md`,
`docs/agent-handoff.md`, `docs/gpu-performance.md`,
`docs/maintainability-refactor.md`,
`scripts/benchmarks/benchmark_shape_sweep.py`,
`src/renewable_huber/backends/_dlpack.py`,
`src/renewable_huber/backends/native_cuda_backend.py`,
`src/renewable_huber/estimator.py`, `src/renewable_huber/exceptions.py`,
`src/renewable_huber/serialization.py`, `tests/test_backend_capabilities.py`,
`tests/test_dlpack_adapters.py`, `tests/test_estimator.py`,
`tests/test_native_cpu_backend.py`, `tests/test_native_cuda_backend.py`.

Added (8): `scripts/run_test_profile.py`,
`scripts/benchmarks/shape_sweep/shapes.py`,
`scripts/benchmarks/shape_sweep/environment.py`,
`scripts/benchmarks/shape_sweep/timing.py`,
`scripts/benchmarks/shape_sweep/runners.py`,
`scripts/benchmarks/shape_sweep/cli.py`, `tests/test_checkpoint_payload.py`,
`tests/test_profile_runner.py`.

No generated build output, artifact or benchmark JSON is tracked or untracked
inside the repository; every capture lives in the session scratchpad.

#### Verification run

| Gate | Command | Result |
|---|---|---|
| Full discovery | `python -m unittest discover -s tests` | **285 tests, OK** |
| Profile table | `scripts/run_test_profile.py --check` | 6 profiles cover 22 modules |
| core | `scripts/run_test_profile.py core` | 179 tests, exit 0 |
| performance | `... performance` | 38 tests, exit 0 |
| native-cpu | `... native-cpu` | 19 tests, exit 0 |
| optional-cpu | `... optional-cpu` | 19 tests, 2 skips, exit 0 |
| cuda (real RTX 5070 Ti) | `... cuda` | **44 tests, exit 0** |
| all (real GPU) | `... all` | **285 tests, exit 0** |
| Lint | `ruff check src tests scripts` | pass |
| Format | `ruff format --check src tests scripts` | 74 files, pass |
| Golden corpus | `scripts/generate_native_golden.py --check` | 4 cases match |
| Release metadata | `scripts/native/validate_release_artifacts.py --source-only` | consistent for 0.6.0 |
| C++ ABI | `g++ -std=c++17 -fsyntax-only -I native/cuda/include native/cuda/src/abi_contract.cpp` | pass |
| Rust format | `cargo fmt --all -- --check` | pass |
| Rust lint | `cargo clippy --locked --workspace --all-targets -- -D warnings` | 0 diagnostics |
| Rust check | `cargo check --locked --workspace --all-targets` | pass |
| Rust tests | `cargo test --locked -p rh-core -p rh-cpu -p rh-cuda-ffi --all-targets` | 3 suites, 14 tests |
| CUDA C ABI smoke | `cmake --build build/static && ctest --test-dir build/static` | 1/1 passed |

Required-profile enforcement re-checked on the separate bare venv holding only
`.[dev]`, which reproduces the CPU CI baseline job: `core` 179 exit 0,
`performance` 38 exit 0, `all` 285 exit 0, `native-cpu` **exit 2**
(`the Rust native CPU extension is not importable`), `optional-cpu` **exit 2**
(`pandas is not installed`). That run also caught a defect in one of my own new
tests, which had depended on PyTorch being installed; it now uses a fixture
profile that borrows only the `environment` declaration.

Benchmark evidence re-captured after these changes, unchanged from the first
round:

- all 11 re-exports and `benchmark_native_cpu_scaling.py` still import;
- pre/post schema comparison with a deterministic sampling block: 120 cases,
  52 skips, 4 identical dataset checksums, 28 result fields — **PASS, every
  compared field identical**;
- 9-round interleaved A/B, pre-split baseline against post-split candidate,
  paired median slowdowns **0.985 / 1.000 / 1.000 / 1.003**. The same two
  absolute gate reasons as before (relative MAD above 5% on the `smoke` latency
  shape, and native CPU not faster than NumPy at 8192x32); both reproduce in the
  self-paired control, so neither is attributable to the split.

#### Known risks and unresolved questions

1. `BackendContractError` is not re-exported at package level (see item 4).
2. The `optional-cpu-profile` CI job installs both PyTorch and TensorFlow. It is
   the only way that required profile is enforced in CI, but the cost is real.
3. `release.yml` and `test-pypi.yml` still use plain `discover`; they smoke-test
   an installed artifact, and tightening their gate is your call.
4. `tests.test_dlpack_adapters` remains the single declared `SHARED_MODULES`
   entry. With the capability tests relocated, no other module needs one.

#### Requested next action / owner

Codex performs final acceptance, commit, push and PR. Claude Code performed no
Git mutation: `git status` shows only working-tree changes, the index is clean,
and HEAD is still `14f9e72`.

### 2026-08-09 — Claude Code — maintainability P3 implementation

- Base SHA / branch: `14f9e72aae0f75c4f84923d61952438861ab1777`, worktree
  `/home/untrollor/renewable-huber-claude-p3` on `claude/maintainability-p3`.
  The tree was clean at that SHA before editing. **Nothing is committed or
  pushed; no branch, tag, remote or release state was touched.**

#### Scope and architecture decisions

**P3.1 — `CheckpointPayload` boundary.** `serialization.py` is now a codec and
nothing else: a frozen keyword-only `CheckpointPayload`
(`format_version`, `config`, `state`, `feature_names`, `diagnostics`) plus
`write_checkpoint` / `read_checkpoint`. Building a payload from a fitted model
(`_checkpoint_payload`) and restoring one (`_from_checkpoint_payload`) moved to
the estimator layer. `state_dict()` is now derived from the payload so the two
cannot drift. `save`/`load`/`state_dict` behaviour, the v2 archive layout and
`allow_pickle=False` are unchanged.

- `save_model` and `load_model` were **removed**, not renamed. `load_model` must
  construct an estimator, which is exactly what the boundary forbids, and a
  half-kept pair would be worse than a clean codec. Neither name is exported,
  documented, or used outside `estimator.py`. This is the only internal-surface
  change in P3 and it is yours to accept or reject.
- Diagnostics are modelled but never persisted. `write_checkpoint` does not emit
  them and decoding v1 or v2 always yields `None`, so a loaded model still
  raises `NotFittedError` from `diagnostics_`. No format bump was made; that
  decision is left to you as the brief requires.
- Two behaviour changes inside `read_checkpoint`, both flagged deliberately:
  a file that is not a zip previously escaped as `zipfile.BadZipFile` and now
  becomes `ValidationError("Invalid or corrupted...")`; and the archive handle
  is opened by us so a rejected checkpoint no longer leaks a file descriptor
  (it previously emitted `ResourceWarning`). Both are pre-existing defects that
  the required "corrupted archive" test surfaced.

**P3.2 — executable unittest profiles.** `scripts/run_test_profile.py` defines
`core`, `optional-cpu`, `native-cpu`, `cuda`, `performance` and `all` with
written-out module membership. The five required profiles probe their declared
dependency or device first, using the same readiness conditions the test
modules skip on, and exit 2 when one is missing or when a run is empty or
entirely skipped. `all` is optional and keeps its documented skips.
`--check` reports duplicates, missing modules, typos, non-test modules, unknown
requirements and any module on disk assigned to no profile.
`python -m unittest discover -s tests` is untouched and is itself asserted by
`test_plain_discovery_still_imports_every_module`.

- `tests.test_dlpack_adapters` is the one module deliberately in two profiles
  (`core` and `cuda`): its adapter class drives fakes on any CPU, its three
  integration classes need a device. `SHARED_MODULES` records that exception;
  every other duplicate fails `--check`.
- `tests.test_native_golden` forces `backend="numpy"`, so it is in `core`, not
  `native-cpu`. The CI native job therefore no longer runs it — it runs on all
  nine baseline matrix entries instead, with no coverage lost.
- CI: the `baseline` job runs `--check`, `core` and `performance`; the
  `native-cpu` job runs the `native-cpu` profile; a new CPU-only
  `optional-cpu-profile` job installs the whole optional stack and runs the
  `optional-cpu` profile. **No GPU Actions job was added and
  `gpu-validation.yml` was not restored.** `release.yml` and `test-pypi.yml`
  were left on plain `discover`; they smoke-test an installed artifact and
  changing their gate is your call.

**P3.3 — split `benchmark_shape_sweep.py`.** 1,140 lines became five modules
under `scripts/benchmarks/shape_sweep/` — `shapes` (schema, shapes, dataset
generation, 80), `environment` (capture metadata, 119), `timing` (calibration
and the measurement discipline, 411), `runners` (four engines, 212), `cli`
(orchestration and JSON, 416). `benchmark_shape_sweep.py` is now a 103-line
entry point that re-exports all 11 consumer-visible names under their original
underscored spelling, listed in `__all__` so a deletion is visible.

#### Files changed

Superseded by the corrected inventory in the review-round entry above. The
statistics originally printed here counted only tracked files, so the five new
`shape_sweep/` modules, `scripts/run_test_profile.py` and the two new test
modules were missing from both the file list and the diff total.

#### Verification run

Environment: WSL2 Ubuntu 24.04, Python 3.12.3, NumPy 2.5.1, Rust 1.97.1,
CUDA 12.9, RTX 5070 Ti (driver 596.49), `cuda-full` venv built by
`scripts/setup-wsl-venv.sh` with all optional CPU frameworks installed.

| Gate | Result |
|---|---|
| `ruff check src tests scripts` | pass |
| `ruff format --check src tests scripts` | pass, 74 files |
| `python -m unittest discover -s tests` | 268 tests, **1 pre-existing failure** (below) |
| `scripts/generate_native_golden.py --check` | 4 cases match |
| `scripts/native/validate_release_artifacts.py --source-only` | consistent for 0.6.0 |
| `g++ -fsyntax-only ... abi_contract.cpp` | pass |
| `cargo fmt --all -- --check` | pass |
| `cargo clippy --locked --workspace --all-targets -- -D warnings` | 0 diagnostics |
| `cargo check --locked --workspace --all-targets` | pass |
| `cargo test --locked -p rh-core -p rh-cpu -p rh-cuda-ffi --all-targets` | 14 passed |
| CUDA static build + `ctest` C ABI smoke | 1/1 passed |
| `run_test_profile.py --check` | 6 profiles cover 22 modules |
| profile `core` | 164 tests, exit 0 |
| profile `performance` | 38 tests, exit 0 |
| profile `native-cpu` | 18 tests, exit 0 |
| profile `optional-cpu` | 19 tests, 2 skips, exit 0 (see note) |
| profile `cuda` | 40 tests, exit 1 — the pre-existing failure |
| profile `all` | 268 tests, exit 1 — the same failure |
| `tests/test_checkpoint_payload.py` | 33 tests, exit 0 |
| `tests/test_profile_runner.py` | 29 tests, exit 0 |

Required-profile enforcement was verified against a **separate bare venv holding
only `.[dev]`**, which reproduces the CPU CI baseline job: `core` 164 tests /
5 skips exit 0, `performance` 38 tests exit 0, `all` 268 tests / 55 skips
exit 0, and `native-cpu` **exit 2** with
`the Rust native CPU extension is not importable`. Under `discover` that same
environment reports success. That difference is the whole point of P3.2.

**P3.3 pre/post schema comparison.** The base SHA tree was exported read-only
with `git archive` (no commit, no tag) and the identical sweep
(`--profile smoke --backend all --penalty both --dtype both --lifecycle both
--operation both --warmup 1 --repeats 2`) was captured on both trees:

- 120 cases, 52 skipped entries, 4 dataset SHA-256 checksums and 28 result
  fields per case, identical on both sides; schema name, `schema_version`,
  `arguments`, environment keys, all comparison metadata, all lifecycle and
  sampling-policy fields and every skip reason match exactly.
- With `--minimum-sample-seconds 0.05` the two records differ in exactly one
  field, `sample_repetitions`, in 81 of 120 cases. **Running the pre-split tree
  against itself produces 57 differences in the same single field**, so this is
  the calibration block size tracking its own timing noise, not a behavioural
  change. `minimum_sample_seconds`, the policy input, is identical.
- Repeating both captures with `--minimum-sample-seconds 0` pins the block size
  and the comparison is then **byte-identical on every compared field**.

**Interleaved A/B.** `run_interleaved_benchmark.py` was exercised both
self-paired and against the pre-split checkout, 3 and 9 rounds, `smoke`,
`--backend cpu`. Paired median candidate/baseline slowdowns:

| Run | latency f32 | latency f64 | reference f32 | reference f64 |
|---|---|---|---|---|
| self-paired, identical trees | 0.937 | 0.897 | 0.934 | 0.965 |
| pre-split vs post-split | 0.922 | 0.986 | 0.985 | 0.968 |

All are below 1.0 and far inside the 1.10x CPU threshold. Both runs report
`passed: false` for the same two *absolute* reasons — relative MAD 5.6–8.9%
against a 5% ceiling, and `native/reference ratio` above 1.0 against
`numpy_cpu` on `reference-smoke`. **The self-paired control fails identically**,
which is the evidence that neither is attributable to the split: the `smoke`
shapes are too short for a 5% MAD on this host, and the Rust CPU engine is
genuinely not faster than NumPy at 8192x32. A promotion decision still needs the
`standard` profile on the fixed runner. The first attempt also tripped the
documented guard `sample repetition calibration changed between interleaved
rounds`; `--max-sample-repetitions 8` pins the contract as the policy prescribes.

#### Known risks and unresolved questions

1. **Pre-existing failure, not caused by P3.**
   `tests/test_native_cuda_backend.py::NativeCudaDlpackTests::test_dlpack_requires_exact_dtype_and_c_contiguity`
   fails on this GPU host. `native_cuda_backend.py:148` raises
   `TypeError("native CUDA DLPack dtype must exactly match float64")`, and
   `estimator.py::_validate_features` catches every `TypeError` from
   `backend.asarray(X)` and replaces it with the scikit-learn-compatible
   `"float() argument must be a string or a number"`. The specific message
   survives only as `__cause__`, which `assertRaisesRegex` does not inspect.
   **Reproduced on the pristine `14f9e72` export**, so it predates this work; it
   was invisible because it needs CuPy plus the native CUDA extension plus a
   real device, which neither the Codex sandbox nor GitHub Actions has. The
   `cuda` and `all` profiles fail solely on it. Fixing it means changing a
   public error message, so per the stop condition I left it to you.
2. **`optional-cpu` cannot pass on a GPU host as written.** Running
   `tests.test_tensorflow_backend` before `tests.test_torch_backend` in one
   process makes `torch.linalg.solve` fail with
   `cusolver error: CUSOLVER_STATUS_INTERNAL_ERROR ... cusolverDnCreate`.
   Reproduced on the pristine base tree; `TF_FORCE_GPU_ALLOW_GROWTH=1` does not
   help; each module passes alone; `CUDA_VISIBLE_DEVICES=""` makes the profile
   pass (19 tests, 2 skips). This is torch 2.13.0+cu130 and TensorFlow 2.21.0
   coexisting in one CUDA process, not a defect in this repository, and CPU CI
   runners never hit it. I did **not** mask the GPU inside the runner, because
   that would hide the interaction. Options for you: mask CUDA for that profile,
   split it per framework, or run its modules in separate processes.
3. The new `optional-cpu-profile` CI job installs both PyTorch and TensorFlow,
   the two heaviest wheels in the matrix. It is the only way the required
   profile is actually enforced in CI, but the runtime cost is real.
4. `benchmark_native_cpu_scaling.py` still keeps its own copy of the
   `sys.path`/`PROJECT_ROOT` bootstrap, and `timing.py` and `cli.py` each repeat
   it. Sharing it would need a relative import that an import sorter moves below
   the third-party block, so the existing idiom was preserved rather than
   invented around.
5. `SHARED_MODULES` currently has one entry. `tests.test_backend_capabilities`
   has the same mixed shape (two device-gated tests inside an otherwise portable
   module) and was left in `core` only; if you want its native cases guaranteed
   on the GPU host, add it to `cuda` and to `SHARED_MODULES`.

#### Requested next action / owner

Codex reviews the uncommitted diff, decides on the four open items above
(removal of `save_model`/`load_model`, the two `read_checkpoint` behaviour
changes, the `optional-cpu` GPU-host collision, and whether to fix or file the
DLPack dtype masking), then performs acceptance, commit, push and PR. Claude
Code performed no Git mutation.

### 2026-08-09 — Codex — make development GPU validation local-only

- Base SHA / branch: `6bb2a2f`, `codex/local-gpu-validation`.
- Scope and decisions: cancelled GitHub run `31277407653` and removed the
  manual GPU Actions workflow. Pull-request CUDA correctness, C ABI smoke,
  profiling, shape sweeps and interleaved performance gates now run only on
  the fixed local GPU host. The trusted-tag release workflow still builds the
  CUDA wheel artifacts required for PyPI, but performs no GPU runtime test.
- Files changed: removed `.github/workflows/gpu-validation.yml`; removed GPU
  runtime tests from `.github/workflows/release.yml`; updated contribution,
  release, support and performance documentation, changelog, maintainability
  report and this hand-off.
- Verification run: 206 Python tests passed with 43 documented optional/GPU
  skips; Ruff check/format, native source metadata, `git diff --check`, stale
  workflow-reference scan and focused workflow diff review passed.
- Known risks or unresolved questions: GitHub branch protection cannot enforce
  a local GPU result. Codex acceptance therefore requires the exact commit,
  environment fingerprint and machine-readable JSON gate report before merging
  CUDA behavior or performance changes.
- Requested next action / owner: Claude Code implements P3 without adding a GPU
  Actions job. P3 GPU profiles must be runnable locally and must fail when the
  required device/dependencies are absent; Codex reviews the local evidence.

### 2026-08-09 — Codex — stages 1–3 implementation and publication

- Base SHA / branch: reviewed working tree based on `08b75d3`,
  `codex/maintainability-p0`; Codex publishes its final tree against remote
  `main` (`27322c4`) through the GitHub Git Data API because this sandbox
  mounts local `.git` read-only.
- Scope and decisions: added Codex-only Git ownership and durable hand-offs;
  completed WSL `minimal`/`cpu-full`/`cuda-full` profiles and explicit
  passed/failed/skipped verification; added fixed-runner CPU/CUDA interleaved
  A/B capture, paired slowdown checks, sampling-contract guards and manual GPU
  workflow integration.
- Files changed: `.github/workflows/gpu-validation.yml`, `.gitignore`,
  `AGENTS.md`, `CONTRIBUTING.md`, this file,
  `docs/maintainability-refactor.md`, `docs/native-performance-policy.md`,
  `native/cuda/src/engine_state.cuh`, both WSL setup scripts, three new
  environment/performance scripts and two new test modules.
- Verification run: 205 Python tests passed with 43 documented optional/GPU
  skips; Ruff check/format, golden corpus, native metadata, C++ ABI compile,
  Rust fmt/clippy/check/tests, CUDA static build and C ABI smoke passed. A
  three-pair native CPU end-to-end self A/B reported paired slowdown `1.002x`
  and passed its smoke thresholds. Full `cpu-full`/`cuda-full` dependency
  installation and the 9-pair RTX workflow require network/GPU access outside
  this sandbox.
- Known risks or unresolved questions: `actionlint`/PyYAML was unavailable, so
  GitHub Actions must still parse the workflow. The local branch cannot record
  the connector-created commit until `.git` is writable; fetch and align it
  before the next implementation session. Direct GitHub DNS and GPU access are
  blocked in this sandbox. Claude Code performed no Git mutation.
- Requested next action / owner: superseded by the newer local-only GPU policy
  entry above. Claude Code starts P3 only from a clean, synchronized worktree.

### 2026-08-09 — Codex — Claude Code implementation brief for maintainability P3

- Base SHA / branch: use the latest Codex-merged `main` containing the
  local-only GPU policy; before editing, record the exact SHA here and confirm
  the worktree is clean.
- Ownership: Claude Code may edit a dedicated local worktree but must not
  commit, push, open a PR or change release state. Codex will review the
  uncommitted diff, run acceptance, and perform all Git mutations.
- Scope: implement all three maintainability-P3 items below. Do not change an
  algorithm, native ABI, kernel order, stream behavior, benchmark schema or
  public estimator behavior.

#### P3.1 — `CheckpointPayload` boundary

1. Introduce a typed payload that owns checkpoint format version, estimator
   configuration, `RenewableHuberState`, feature names and optional diagnostics.
2. The serialization layer may encode/decode that payload, but must not import
   an estimator class, instantiate one, or call `_restore_state`.
3. `RenewableHuberRegressor.load()` must decode the payload, apply explicit
   backend/device/dtype overrides, instantiate `cls`, and restore state from
   inside the estimator layer. Subclass loading must continue to work.
4. Preserve pickle-free `.npz` storage and v1/v2 compatibility. If diagnostics
   are persisted, use a new format version with optional diagnostics; never
   invent last-batch diagnostics for legacy checkpoints that do not contain
   them. Document the legacy behavior.
5. Preserve backend/dtype migration, feature-name validation, weighted state,
   native CPU/CUDA resume and sklearn subclass round trips.

Acceptance tests must cover v1, v2, current-format round trips, corrupted
payloads, non-finite state, configuration overrides, subclass restoration,
feature names, weighted/L1 history, and optional diagnostics. Persistence tests
must prove `serialization.py` no longer references `_restore_state` or imports
`RenewableHuberRegressor` at runtime.

#### P3.2 — executable unittest profiles

1. Keep `unittest` as the authoritative runner; do not add pytest markers that
   CI never executes.
2. Add one small profile runner that builds explicit suites for at least:
   `core`, `optional-cpu`, `native-cpu`, `cuda`, `performance`, and `all`.
3. A required profile must fail when its dependency/device is absent instead
   of reporting a successful suite made entirely of skips. The ordinary `all`
   developer profile may retain documented optional skips.
4. Update CPU CI and the documented local GPU validation command to invoke the
   appropriate required profiles, while preserving direct
   `python -m unittest discover -s tests` compatibility. Do not add a GPU
   GitHub Actions job.
5. Add self-tests for profile membership, duplicate modules and required-skip
   detection.

#### P3.3 — split `benchmark_shape_sweep.py`

Split by cohesive responsibility, not target line count:

- shape/schema/environment/dataset generation;
- lifecycle calibration and timing;
- NumPy/native CPU/CuPy/native CUDA runners;
- CLI orchestration and JSON output.

Keep `benchmark_shape_sweep.py` as the stable CLI entry point. Preserve the
schema-v2 JSON byte-level field contract and re-export the helpers currently
consumed by `benchmark_native_cpu_scaling.py` and
`tests/test_benchmark_performance_policy.py`: `PROFILES`, `_calibration_run`,
`_dataset_checksum`, `_fit_batch`, `_lifecycle_metadata`, `_measure`,
`_restore_empty_state`, `_run_operation`, `environment_metadata`, and
`make_batches`. The new interleaved runner must work against both the pre-split
baseline checkout and the post-split candidate.

- Required verification: all commands in `AGENTS.md`, the new unittest profile
  self-tests, golden generation check, source metadata check, a CPU self-paired
  interleaved smoke, CUDA C ABI smoke when available, and a comparison proving
  representative pre/post split shape-sweep records have identical keys and
  schema.
- Stop conditions: if compatibility requires a public API, checkpoint format,
  native ABI, benchmark schema or numerical change, stop and leave the decision
  to Codex rather than expanding scope.
- Requested next action / owner: Claude Code implements and appends a hand-off
  entry listing every file, command and unresolved issue; Codex reviews and is
  the only agent allowed to commit/push.

### 2026-08-09 — Codex — P0–P2 acceptance and P3 boundary

- Base SHA / branch: `08b75d3`, `codex/maintainability-p0`.
- Scope and decisions: maintainability P0–P2 passed code-level review. WSL
  reproducibility and fixed-runner interleaved performance gates are being
  completed before P3. Commit/push ownership is now Codex-only.
- Verification run: Python, Ruff, C++ ABI assertions, Rust fmt/clippy/check/tests,
  golden corpus, native metadata, CUDA static/shared builds, C ABI smoke and 17
  exported symbols passed. GPU numerical tests could not run inside the Codex
  sandbox because NVML/GPU access was blocked.
- Known risks or unresolved questions: the live GitHub refs must be fetched
  before publishing; ignored local tracking refs copied from Windows are not
  authoritative.
- Requested next action / owner: Codex completes environment/performance gates
  and publishes. Claude Code then implements maintainability P3 strictly from
  the P3 hand-off entry that Codex will add.
