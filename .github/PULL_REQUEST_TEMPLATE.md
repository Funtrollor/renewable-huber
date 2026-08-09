## Summary

Describe the problem, the chosen solution, and the user-visible effect.

## Validation

List the exact commands, environments, and benchmark inputs used.

## Checklist

- [ ] The change is focused and based on the latest `main`.
- [ ] Tests cover new behavior and important failure cases.
- [ ] NumPy parity is demonstrated for backend-specific numerical changes.
- [ ] Public API, support-matrix, or architecture documentation is updated when relevant.
- [ ] `CHANGELOG.md` is updated for user-visible behavior.
- [ ] No datasets, model artifacts, research PDFs, credentials, or generated build output are added.
- [ ] `ruff check`, `ruff format --check`, relevant tests, and package build pass locally.
- [ ] Required test profiles were used; missing native dependencies/devices did not become silent skips.
- [ ] Golden corpus and native ABI/contract gates pass when algorithm or native boundaries are touched.
- [ ] GPU changes include local-only evidence tied to the exact commit, environment fingerprint, artifact hash, and machine-readable gate result.
- [ ] Distribution versions, exact native dependencies, citation, security policy, changelog, and release metadata stay synchronized when preparing a release.
- [ ] Performance claims include reproducible before/after measurements.

## Compatibility

Note any effect on checkpoints, dtype, device, backend, operating system, or Python version.
