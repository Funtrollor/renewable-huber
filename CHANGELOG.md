# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and version
numbers follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) while the public API is
stabilised.

## [Unreleased]

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
