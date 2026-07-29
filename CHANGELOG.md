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

### Changed

- Explicit `backend="native_cuda"` requests now use the native engine and fail
  clearly when its extension or requested capability is unavailable; automatic
  backend selection remains unchanged.

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
