# Contributing to renewable-huber

Thanks for helping improve robust streaming regression. Bug reports, reproducible performance
results, documentation fixes, and focused code contributions are welcome.

## Before opening an issue

- Search existing issues and pull requests.
- Use the bug or feature template and include the smallest reproducible example.
- Do not publish security vulnerabilities in an issue; follow `SECURITY.md`.
- For performance reports, include OS, Python version, backend, dtype, array shape, hardware,
  dependency versions, warm-up policy, and benchmark command.

## Development setup

Create an isolated environment with Python 3.10–3.12:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,sklearn,pandas]"
```

Install only the optional compute framework you need. CUDA tests require compatible NVIDIA
drivers and the CuPy CUDA 12 extra:

```powershell
python -m pip install -e ".[dev,gpu-cupy]"
```

## Validation

Run the checks relevant to the change before opening a pull request:

```powershell
python -m unittest discover -s tests -v
ruff check src tests scripts
ruff format --check src tests scripts
python -m build
```

Backend-specific changes must include parity tests against NumPy. Performance changes must include
correctness tests and reproducible before/after benchmark output; a faster result is not accepted
if it changes the documented numerical contract.

## Pull requests

- Branch from the latest `main` and keep the change focused.
- Add or update tests and public documentation with code changes.
- Update `CHANGELOG.md` under `Unreleased` for user-visible behavior.
- Do not commit datasets, model files, generated build artifacts, research PDFs, credentials, or
  local benchmark output.
- Let CI pass on all required platforms. GPU validation is manual because untrusted pull-request
  code must never run automatically on the self-hosted GPU runner.
- Use Conventional Commit-style imperative subjects when practical, for example
  `perf: fuse CUDA renewal kernels`.

The `main` branch requires a pull request, an up-to-date branch, all cross-platform and optional
CPU integration checks, package smoke tests, and resolved review conversations. Force pushes and
branch deletion are disabled. The applied settings are recorded in
`.github/branch-protection.json`.

By submitting a contribution, you agree that it may be distributed under the repository's chosen
license once that license is added.
