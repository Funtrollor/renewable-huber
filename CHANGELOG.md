# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and version
numbers follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) while the public API is
stabilised.

## [Unreleased]

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

[Unreleased]: https://github.com/Funtrollor/renewable-huber/compare/v0.6.1...HEAD
[0.6.1]: https://github.com/Funtrollor/renewable-huber/compare/v0.5.1...v0.6.1
[0.5.1]: https://github.com/Funtrollor/renewable-huber/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/Funtrollor/renewable-huber/releases/tag/v0.5.0
