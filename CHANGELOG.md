# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and version
numbers follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) while the public API is
stabilised.

## [Unreleased]

### Changed

- The Rust CPU engine is faster on every standard shape that engages its
  thread pool. Four changes, each measured against the 0.7.0 engine; see
  `docs/native-core-p1.md` for the measurements and the rejected alternative:
  - the row-chunked gradient (all L1 batches, and narrow unpenalized ones)
    now accumulates into worker-local buffers. In place, neighbouring
    workers wrote the same cache line on every row, and the loop did not
    speed up with threads at all;
  - Newton iterations reuse the residual the accepted line-search trial
    already computed, as L1 iterations did, instead of recomputing `X @ beta`;
  - the weighted Gram matrix is built 256 rows at a time through a
    cache-resident scratch block, reading the batch once instead of writing
    and re-reading a full `n * p` weighted copy. That workspace is gone;
  - `dot` keeps eight independent partial sums, so the residual and
    prediction row kernels vectorize.

  The first three are bitwise identical to 0.7.0. The `dot` change fixes a
  different, still deterministic summation order, so native CPU coefficients
  and information matrices differ from 0.7.0 in the last bits; the golden
  corpora and their tolerances are unchanged.
- The approved CPU schema-v2 baseline is now
  `benchmarks/baselines/p6-windows-ryzen9900x-native-cpu-v2.json`, captured on
  the fixed Ryzen 9 9900X runner with the optimized engine: NumPy/native
  1.38x-7.86x (median 1.81x) across all 32 cases, gate passed with 0.5-second
  samples. `p3-windows-ryzen9900x-native-cpu-v2.json` stays as history. The
  wide unpenalized ratios are lower than before because the NumPy reference
  became faster there, while native also got faster.
- A fixed-host interleaved A/B against `v0.7.0` (cold and steady,
  `--freeze-sample-repetitions`, records under `benchmarks/baselines/`)
  measured the optimized engine 1.21x (cold) and 1.18x (steady) faster at the
  median, with no case slower. Neither gate passed: three float32 L1 cases
  take a different number of solver iterations after the `dot` change
  (difference 2-5, limit 1), and in the steady capture one of them also
  exceeded the 5% MAD limit (5.02%).

## [0.7.0] - 2026-10-03

This minor release breaks the native CUDA interface: C ABI 2 and Python API 4
replace C ABI 1 and Python API 3, and the two generations refuse each other.
The public estimator API and checkpoint format 2 are unchanged.

### Added

- Native CUDA L1 (`penalty="l1"`): the LAMM proximal-gradient transition now
  runs whole-batch on the GPU, mirroring the NumPy reference and the Rust CPU
  engine, with transactional `previous_lambda` commits and L1 checkpoints that
  resume across all three engines. This raises the CUDA C ABI to 2 and the
  CUDA Python API to 4; older and newer components refuse each other instead
  of treating L1 as `none`.
- Native CUDA `predict` accepts CUDA DLPack tensors (CuPy, PyTorch, TensorFlow
  eager) and reads them in place on the engine stream; raw feature matrices
  are widened on device for both host and device input. Predictions are still
  returned as NumPy arrays.
- A `native_update_penalties` backend capability. The core refuses a penalty a
  native engine does not advertise with `ValidationError` before the engine is
  called.
- A second, frozen golden corpus (`tests/golden/native_core_v2.json`) of four
  converged L1 streams, replayed by the NumPy, Rust CPU and CUDA engines.
- Linux x86-64 (`manylinux_2_28`) CUDA 12 plugin wheels, built by
  `scripts/native/build_linux_cuda_wheel.sh` in both the release workflow and
  a new no-GPU pull-request CI job.
- CPython 3.13 support for the base package and both native wheels, across CI
  and the release matrix (20 CPU wheels, 8 CUDA wheels).
- An English README (`README.en.md`) and a MkDocs Material documentation site
  (`docs` extra, `mkdocs.yml`) with a generated Python API reference. A new
  Docs workflow builds it with `--strict` on every pull request and deploys it
  to GitHub Pages only when run by hand.
- Development tooling: mypy over `src/renewable_huber` and branch coverage of
  the required `core` profile (`fail_under = 72`) in the CI quality job, and
  `proptest` property tests for the `rh-core` validation contracts and the
  `rh-cpu` numeric kernels.
- First on-device verification of the native CUDA C ABI 2 / Python API 4
  engine, on the fixed Windows GPU host (RTX 5070 Ti, SM 12.0, CUDA 12.9,
  driver 616.64, Python 3.11) at `4114918`. The `core`, `native-cpu`, `cuda`
  and `performance` profiles, both golden corpora, the C ABI smoke test, the
  17-symbol export check and the clean CUDA wheel smoke all passed. The
  PyTorch and TensorFlow CUDA DLPack integration tests skipped because neither
  framework was installed.
- On-device verification of PyTorch CUDA DLPack input to native CUDA, on the
  same host at `33cb075` with `torch` 2.9.0+cu129: `PyTorchDlpackIntegrationTests`
  passed in the `cuda` profile, and the native extension still reported
  CUDA runtime 12090. TensorFlow eager CUDA DLPack remains unverified on a
  device, because TensorFlow has no GPU support on native Windows since 2.11;
  the support matrix says so.
- Fixed-host native CUDA L1 baselines
  (`benchmarks/baselines/p5-windows-rtx5070ti-native-cuda-l1-run{1,2,3}.json`):
  three standard-profile runs against CuPy under the same input transport.
  Native CUDA L1 stays explicit opt-in; `backend="auto"` never selects native
  CUDA. The results and the reasoning are in
  `docs/native-penalty-completion-plan.md`.
- An accepted interleaved `penalty="none"` A/B of native CUDA across the
  ABI 2 change: `fca7b83` (ABI 1 / API 3) against `10fc363` (ABI 2 / API 4)
  on the same host, cold and steady. Both gates passed on all 16 native cases,
  and every difference was inside ±10%, which is no measurable difference.
  The records, gate reports and sample-repetition plans are
  `benchmarks/baselines/p5-windows-rtx5070ti-native-cuda-none-ab-{cold,steady}-*.json`.
- `run_interleaved_benchmark.py --allow-native-version-change`, for an A/B
  between builds whose native extensions report a different `abi_version` or
  `python_api_version`. Only those two fields leave the hardware and runtime
  fingerprint. `check_performance_regression.py` has no such option and still
  rejects an interface change.
- `run_interleaved_benchmark.py --freeze-sample-repetitions`. It calibrates
  every case once for both variants, writes `sample-repetitions-plan.json`
  (each case gets the larger block size, still capped by
  `--max-sample-repetitions`), and uses that plan in every round. Rounds
  therefore merge without the `--max-sample-repetitions 1` workaround. In this
  mode both variants run the candidate's sweep harness against their own
  source tree through `RENEWABLE_HUBER_BENCHMARK_SOURCE_ROOT`, and records keep
  `git_revision` apart from `benchmark_harness_git_revision`. The shape sweep
  gains the matching `--sample-repetitions-plan` option.

### Changed

- The `Documentation` project URL of all three distributions points to the
  MkDocs site at <https://funtrollor.github.io/renewable-huber/> (the home page
  for the base package, the rendered P1/P2 notes for the native packages)
  instead of the GitHub README and the raw Markdown notes.
- `renewable-huber-native-cuda` now depends on NVIDIA's `nvidia-*-cu12` runtime
  wheels (cudart, cuBLAS, cuSOLVER, cuSPARSE, nvJitLink) and loads that set at
  import, falling back to a system CUDA 12 toolkit only when the set is
  incomplete. `CUDA_PATH` is no longer required.
- The C ABI 2 break renames `RhCudaUnpenalizedConfig` to `RhCudaUpdateConfig`
  and `rh_cuda_engine_predict_host`/`RhCudaHostPrediction` to
  `rh_cuda_engine_predict`/`RhCudaPrediction`; the library still exports
  exactly 17 `rh_cuda_*` symbols.
- The native CUDA shape sweep and profiler accept `penalty="l1"`.
- The interleaved gate report (`gate.json`) is at schema version 2. It always
  records `allow_native_version_change` and, for each gated native family,
  both sides' `abi_version`/`python_api_version` and whether they changed. It
  also records how sample blocks were sized, as
  `sample_repetitions: {policy, plan_sha256, harness}`, where `policy` is
  `frozen_plan` or `per_round_calibration`. Each merged record's
  `interleaved_capture` carries the same field.
- `validate_record` accepts native CUDA `penalty="l1"` benchmark records when
  the recorded `native_cuda_abi` shows `abi_version` 2 or later or lists `l1`
  in `supported_penalties`. ABI 1 records that claim L1 are still rejected.
- The GPU-host runbook (`docs/gpu-host-runbook.md`) now matches what the
  scripts require:
  - the interleaved `penalty="none"` A/B runs with `--backend gpu
    --allow-native-version-change --freeze-sample-repetitions`, gives the
    steady run a separate output directory, and is accepted only when
    `gate.json` shows `sample_repetitions.policy` `frozen_plan`;
  - both venvs use the same Python 3.10–3.12 version and pinned
    NumPy/SciPy/CuPy;
  - the native build scripts get absolute `-Python` paths, and the developer
    shell is located with `vswhere` and started with `-SkipAutomaticLocation`;
  - the two expected DLPack integration skips are documented;
  - GPU clients are recorded before and after each run instead of being
    closed.

## [0.6.1] - 2026-08-09

This is the first published native-core release. The earlier `v0.6.0` tag did
not complete its artifact workflow and was never published to PyPI; its tag is
retained as an immutable historical record rather than moved or reused.

### Added

- An opt-in `native_cpu` backend built from the `rh-core`, `rh-cpu`, and
  `rh-python-cpu` Rust crates.
- Whole-batch native CPU Newton and LAMM solvers for contiguous NumPy
  `float32`/`float64` inputs, including portable checkpoints and
  minimum-norm singular-system fallback.
- An opt-in Rust/PyO3 and CUDA C++ whole-batch engine for unpenalized
  Renewable Huber updates, with persistent device state and workspaces.
- A unified Rust workspace and separate compatible CPU/CUDA native wheel
  projects for the `renewable-huber` 0.6 API.
- Cross-platform native CPU CI, golden-corpus differential tests, clean native
  wheel installation checks, and NumPy/native shape-sweep benchmarking.
- Native CUDA golden differential tests, shape-sweep support, profiling
  support, and a reproducible Windows source-build script.
- Schema-v2 native performance records, a fixed-runner regression/competitor
  gate, and a conservative calibration-based backend advisor.
- Native CUDA DLPack device input with strict dtype/device/contiguity checks
  and same-transport CuPy performance gates.
- Public native CPU `n_jobs` control with estimator-local Rayon pools,
  checkpoint/scikit-learn parameter compatibility, effective `n_jobs_`
  reporting, and a reproducible thread-scaling benchmark.
- Framework-neutral CUDA DLPack adapters for CuPy, PyTorch CUDA, and
  TensorFlow eager GPU tensors, including explicit stream/lifetime safety and
  a no-implicit-device-to-host-copy contract.
- Opt-in `cuda_graphs` and float32 `cuda_fast_math` tuning, runtime capability
  reporting through `cuda_features_`, and reproducible benchmark/Nsight
  evidence.
- Release-gated native distributions: 15 CPU wheels across five OS/architecture
  targets and three Windows CUDA 12 plugin wheels, with exact base-version,
  clean-install, artifact-set, and OIDC publishing checks.
- Runtime CPU dispatch for `backend="auto"`: a bounded, process-local host
  calibration may now select the `native_cpu` engine for a large enough batch.
  It reads no CPU brand or model string, persists nothing, generalises to
  unseen shapes through a normalised log work/cost ratio model with a
  conservative uncertainty allowance, and falls back to NumPy on any failure.
  Every native selection clears the same entry margin independently, so no
  estimator's choice influences another's; cached measurements are keyed on an
  observable runtime signature (CPU affinity, thread environment, and optional
  effective BLAS/OpenMP pool sizes) and are discarded when it changes or after
  `fork`.
  Fitted estimators report the decision through a new `auto_dispatch_`
  attribute, and `scripts/benchmarks/benchmark_auto_dispatch.py` compares
  `auto` against both explicit CPU backends while isolating calibration from
  steady-state cost. See `docs/cpu-auto-dispatch-rfc.md`.

### Changed

- Checkpoint persistence now uses a codec-only `CheckpointPayload` boundary;
  the undocumented deep-import helpers `serialization.save_model` and
  `serialization.load_model` were removed while the public estimator
  `save()`/`load()` API and version-2 archive format remain unchanged.
- CI now runs explicit required unittest profiles that reject missing
  dependencies, missing devices and all-skipped native suites; GPU runtime
  validation remains local to the maintainer host.
- The shape-sweep benchmark was split into cohesive modules while preserving
  its CLI, schema-v2 records and all consumer-visible helper imports.
- Pull-request GPU validation now runs only on the maintainer's fixed local GPU
  host; GitHub Actions remains responsible for CPU CI and release artifact
  assembly.
- Explicit `backend="native_cuda"` requests now use the native engine and fail
  clearly when its extension or requested capability is unavailable; automatic
  CUDA backend selection remains unchanged.
- `backends.resolve_backend("auto")` still resolves to NumPy on CPU. The
  workload-aware choice happens one level up, in the estimator, because the
  batch shape only exists after validation.
- Native CPU residual, gradient, and weighted-Gram hot paths now use a
  size-gated Rayon/SIMD implementation with bounded scratch memory, a
  row-major small-batch gradient path, transactional zero-clone resident
  state, a float64 Cholesky fast path, and accepted-residual reuse.
- Native CUDA now batches device scalar reductions, uses a guarded Cholesky
  fast path before the existing LU/SVD fallbacks, fuses update/state export,
  and uses pinned scalar results plus a library-owned stream-ordered memory
  pool.
- Native CUDA host input now transfers raw feature matrices and appends the
  intercept on-device, avoiding a full CPU-side design-matrix copy.
- Native CUDA lazily creates SVD fallback resources and consumes device X/y
  without redundant D2D staging while keeping DLPack capsules alive through
  stream completion.
- Cold shape-sweep timings now exclude fitted-model destruction and record
  that lifecycle boundary explicitly for fair native/CuPy comparisons. Short
  operations use fixed block samples with recorded GC/timer policy and no
  outlier filtering.
- Repeated CuPy cold estimators reuse one process-level Windows CUDA DLL
  directory registration instead of leaking a loader handle per estimator.
- Resident native engines use an exact process-local state token instead of
  `batch_count` alone when deciding whether a checkpoint mirror must be
  restored.
- Native performance gates now reject non-finite or unconverged measurements,
  fingerprint native providers and CUDA drivers, and refuse to mix different
  dataset or solver contracts in one dispatch decision.
- The Rust workspace now requires Rust 1.83 and uses PyO3 0.29 plus rust-numpy
  0.29, closing the three PyO3 advisories reported against the previous lockfile.
- Release and TestPyPI source validation use required unittest profiles rather
  than allowing missing dependencies to appear as successful all-skip suites.

### Fixed

- Corrected the manylinux release matrix so x86-64 and aarch64 CPU wheels pass
  the intended Rust target to Maturin.
- Synchronized the public documentation with CPU auto-dispatch, exact native
  package dependencies, measured schema-v2 performance ranges, WSL2-first
  development and local-only GPU validation.
- Moved CUDA wheel compilation to a pinned CUDA 12.9 toolkit on GitHub-hosted
  Windows 2022/Visual Studio 2022 runners; GPU runtime correctness and
  performance remain local-only. The hosted build now installs and preflights
  the complete cuBLAS/cuSOLVER/cuSPARSE/nvJitLink runtime and development
  closure, verifies the SASS/PTX payload, and clean-imports the wheel without a
  GPU. Published wheels do not bundle NVIDIA DLLs; users provide the compatible
  CUDA 12 runtime through their driver/toolkit installation.
- Compiled the CUDA engine as whole-program device code so the release wheel
  keeps its SM 120 PTX. Separable compilation links every architecture through
  nvlink, which emits SASS only, and the resulting wheel would have run on
  nothing newer than Blackwell.
- Replaced the retired macOS 13 x86-64 release runner with macOS 15 Intel and
  moved the Apple Silicon release wheel to macOS 15.
- Restricted `Requires-Python` to the tested CPython 3.10–3.12 range across
  the base, native CPU and native CUDA distributions, with fail-closed source
  and artifact metadata validation.

### Security

- Upgraded PyO3 from 0.23 to the patched 0.29 series, addressing
  GHSA-36hh-v3qg-5jq4, GHSA-chgr-c6px-7xpp and GHSA-pph8-gcv7-4qj5.

## [0.5.1] - 2026-07-28

### Fixed

- Check out the tagged repository before verifying and creating a GitHub Release.
- Make GitHub Release artifact uploads safe to rerun after a partial release failure.

## [0.5.0] - 2026-07-28

### Added

- NumPy, CuPy/CUDA, PyTorch, and TensorFlow computation backends.
- Renewable Huber estimation and L1-penalised renewable variable selection.
- pandas feature-name validation, frequency-style sample weights, and a scikit-learn adapter.
- Versioned checkpoints with explicit backend, device, and dtype migration.
- CPU and GPU benchmark tools, cross-platform CI, and manual self-hosted GPU validation.
- Community health files, dependency automation, citation metadata, and release automation.
- Apache-2.0 licensing and an independent-implementation attribution notice.

### Changed

- The package version is now read from a single source file.

### Security

- Documented private vulnerability reporting and supported-version policy.

[Unreleased]: https://github.com/Funtrollor/renewable-huber/compare/v0.7.0...HEAD
[0.7.0]: https://github.com/Funtrollor/renewable-huber/compare/v0.6.1...v0.7.0
[0.6.1]: https://github.com/Funtrollor/renewable-huber/compare/v0.5.1...v0.6.1
[0.5.1]: https://github.com/Funtrollor/renewable-huber/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/Funtrollor/renewable-huber/releases/tag/v0.5.0
