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

WSL2/Linux is the primary native-core development environment. Bootstrap a
new WSL checkout with one of the declared profiles:

```bash
bash scripts/setup-wsl-toolchain.sh             # add --cuda for CUDA 12
bash scripts/setup-wsl-venv.sh --profile minimal
# or: cpu-full / cuda-full
```

`minimal` covers the base package and native CPU extension. `cpu-full` also
installs pandas, SciPy, scikit-learn, PyTorch and TensorFlow. `cuda-full` adds
CuPy, builds the native CUDA extension and fails verification when WSL GPU
passthrough is unavailable. `scripts/verify_wsl_environment.py` reports every
required component explicitly; an unavailable integration must never be
mistaken for a fully passing suite.

The PowerShell setup below remains supported for Windows-only work.

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

`discover` remains supported and is the quickest local pass. It is tolerant by
design: a suite whose dependency or device is missing reports success as a set
of skips. For anything that has to *prove* it ran, use a named profile:

```bash
python scripts/run_test_profile.py --list     # membership and requirements
python scripts/run_test_profile.py --check    # membership consistency only
python scripts/run_test_profile.py core       # portable NumPy behaviour
python scripts/run_test_profile.py all        # everything, optional skips kept
```

`core`, `optional-cpu`, `native-cpu`, `cuda` and `performance` are *required*
profiles: each probes its declared dependency or device first and exits with
status 2 when one is missing, so a green result cannot be an empty one. `all`
is the developer default and keeps its documented optional skips. GPU work uses
the `cuda` profile on the fixed local host; it must not run in GitHub Actions.
Adding a test module without assigning it to a profile fails `--check`.

`optional-cpu` is CPU-only by definition and enforces it: the runner sets
`CUDA_VISIBLE_DEVICES=""` for that profile and restores the previous value
afterwards. PyTorch and TensorFlow each initialise their own CUDA runtime in
one process, and on a machine with a GPU the second one to do so can fail
(`cusolverDnCreate` returning `CUSOLVER_STATUS_INTERNAL_ERROR`), which would
make the profile's result depend on import order. No other profile forces any
environment, so `cuda` still sees the device. `run_test_profile.py --list`
prints the forced variables.

Backend-specific changes must include parity tests against NumPy. Performance changes must include
correctness tests and reproducible before/after benchmark output; a faster result is not accepted
if it changes the documented numerical contract.

## Pull requests

- Branch from the latest `main` and keep the change focused.
- Add or update tests and public documentation with code changes.
- Update `CHANGELOG.md` under `Unreleased` for user-visible behavior.
- Do not commit datasets, model files, generated build artifacts, research PDFs, credentials, or
  local benchmark output.
- Let CI pass on all required CPU platforms. Run GPU correctness, CUDA smoke,
  profiling, and performance gates locally on the fixed GPU host; GPU
  validation must not run in GitHub Actions for pull requests.
- Use Conventional Commit-style imperative subjects when practical, for example
  `perf: fuse CUDA renewal kernels`.

For assisted development, Claude Code implements an accepted engineering plan
and records its hand-off in `docs/agent-handoff.md`. Codex owns review,
acceptance, commits, pushes and pull requests. The two agents must use separate
worktrees when active concurrently.

The `main` branch requires a pull request, an up-to-date branch, all cross-platform and optional
CPU integration checks, package smoke tests, and resolved review conversations. Force pushes and
branch deletion are disabled. The applied settings are recorded in
`.github/branch-protection.json`.

Unless explicitly marked otherwise, contributions intentionally submitted for inclusion are
distributed under the Apache License, Version 2.0, as described in section 5 of that license.
