# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and version
numbers follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) while the public API is
stabilised.

## [Unreleased]

The next minor release is `0.6.0`; the published stable package remains
`0.5.1` until this native-core work is released.

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

### Changed

- Explicit `backend="native_cuda"` requests now use the native engine and fail
  clearly when its extension or requested capability is unavailable; automatic
  backend selection remains unchanged.
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

[Unreleased]: https://github.com/Funtrollor/renewable-huber/compare/v0.5.1...HEAD
[0.5.1]: https://github.com/Funtrollor/renewable-huber/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/Funtrollor/renewable-huber/releases/tag/v0.5.0
