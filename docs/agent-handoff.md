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
