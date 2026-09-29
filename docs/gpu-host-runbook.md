# GPU host runbook (Windows)

This is the checklist for the work that only the fixed GPU host can do:
running the native CUDA engine for the first time on a device, the N5
measurements from the
[native penalty completion plan](native-penalty-completion-plan.md#n5--benchmark-documentation-and-release-readiness),
and preparing the 0.7.0 release. It assumes a fresh clone on a Windows
machine with an NVIDIA GPU (the committed baselines were taken on an
RTX 5070 Ti, SM 12.0). Read [`AGENTS.md`](../AGENTS.md) first.

The native CUDA engine is at C ABI 2 / Python API 4. As of `main` at
`6a0b1b2`, the code has only been compiled, locally for SM 120 and in CI for
SM 75 and 120. Nothing has run on a GPU yet. That makes stage 1 below a real
first run, not a formality.

## 0. Prerequisites

| Component | Requirement | Check |
|---|---|---|
| NVIDIA driver | supports CUDA 12.9 | `nvidia-smi` shows "CUDA Version" ≥ 12.9 |
| CUDA Toolkit | 12.9 (SM 120 needs ≥ 12.8) | `nvcc --version` |
| Visual Studio 2022 Build Tools | "Desktop development with C++" (MSVC x64) | `vswhere -latest -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64` |
| Rust | stable, `x86_64-pc-windows-msvc` | `rustc --version`, `cargo --version` |
| Python | 3.12 x64 (any of 3.10–3.13 works) | `py -3.12 --version` |
| CMake, Ninja | on `PATH` | `cmake --version`, `ninja --version` |
| Git, GitHub CLI | `gh auth login` done | `gh auth status` |

`.gitattributes` pins every text file to LF, so the golden corpus and the
shell scripts are byte-identical to Linux regardless of `core.autocrlf`.

## 1. Environment

In PowerShell, from the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev,sklearn,pandas,gpu-cupy]" "maturin>=1.8,<2"

# Both scripts import the VS x64 toolchain into their own process, build a
# wheel and install it into the interpreter given by -Python.
.\scripts\native\build_native_cpu.ps1  -Python .\.venv\Scripts\python.exe
.\scripts\native\build_native_cuda.ps1 -Python .\.venv\Scripts\python.exe
```

`build_native_cuda.ps1` builds for `RH_CUDA_ARCHITECTURES`, which defaults to
`native` (the local GPU). The CMake and `ctest` commands below need the
compiler on `PATH`. Run them in an x64 developer shell:

```powershell
& "${env:ProgramFiles}\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64 -HostArch amd64
```

Use the `Community` or `Professional` directory instead of `BuildTools` if
that is what is installed.

## 2. Stage 1: correctness on the device (release blocker)

Every command must pass. A required profile exits 2 when its dependency or
device is missing. Treat that as a setup failure, never as a pass.

```powershell
$py = ".\.venv\Scripts\python.exe"
& $py scripts/run_test_profile.py --check
& $py scripts/run_test_profile.py core --verbose
& $py scripts/run_test_profile.py native-cpu --verbose
& $py scripts/run_test_profile.py cuda --verbose
& $py scripts/run_test_profile.py performance --verbose
& $py scripts/generate_native_golden.py --check
& $py scripts/generate_native_golden.py --corpus v2 --check

# C ABI smoke test (includes the five L1 / device-prediction cases).
cmake -S native/cuda -B build/static -G Ninja -DCMAKE_BUILD_TYPE=Release `
      -DRH_CUDA_BUILD_SHARED=OFF -DRH_CUDA_BUILD_TESTS=ON -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build/static
ctest --test-dir build/static --output-on-failure

# Export surface: exactly 17 rh_cuda_* symbols.
cmake -S native/cuda -B build/shared -G Ninja -DCMAKE_BUILD_TYPE=Release `
      -DRH_CUDA_BUILD_SHARED=ON -DRH_CUDA_BUILD_TESTS=OFF -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build/shared
(dumpbin /exports (Get-ChildItem build/shared -Recurse -Filter renewable_huber_cuda.dll | Select-Object -First 1).FullName |
    Select-String '\brh_cuda_\w+').Count   # must print 17

# Installed-wheel smoke on the device.
& $py -m build --wheel --outdir build/base-wheel
& $py scripts/native/smoke_test_cuda_wheels.py --base-dir build/base-wheel --native-dir build/native-cuda-wheel
```

If anything fails, find the root cause in the CUDA/FFI code and fix it with a
regression test. Never skip a test, loosen a tolerance to hide a numeric
difference, or mark a profile optional. The four cases in
`tests/golden/native_core_v2.json` are the L1 oracle. The five smoke cases in
`native/cuda/tests/rh_cuda_smoke.cpp` carry NumPy reference values.

Record the environment with the evidence:

- the commit SHA;
- `nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv`;
- `nvcc --version`;
- the Python and package versions (`pip freeze`).

## 3. Stage 2: N5 measurements

Measurements need the GPU to themselves. Do not run tests, builds, or a second
benchmark at the same time. Drift of about 2–19% between runs of the same
binary is normal on this class of host. Never claim a difference within about
±10%.

### 3a. `penalty="none"` must not have regressed

This is an interleaved A/B of the pre-L1 engine against the current one:

- **Baseline:** `fca7b83`, the `main` commit before the ABI 2 merge.
- **Candidate:** current `main`.

Each side needs its own clone and its own venv, with the native CUDA
extension built from that clone's own sources. ABI 1 Python code cannot load
an ABI 2 extension, or the reverse.

```powershell
git worktree add ..\rh-baseline fca7b83
Push-Location ..\rh-baseline
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,gpu-cupy]" "maturin>=1.8,<2"
.\scripts\native\build_native_cuda.ps1 -Python .\.venv\Scripts\python.exe
Pop-Location

.\.venv\Scripts\python.exe scripts/benchmarks/run_interleaved_benchmark.py `
  --baseline-python ..\rh-baseline\.venv\Scripts\python.exe --baseline-repo ..\rh-baseline `
  --candidate-python .\.venv\Scripts\python.exe --candidate-repo . `
  --output-dir artifacts/n5-none-ab --profile standard --backend native_cuda `
  --penalty none --dtype both --lifecycle cold --operation partial-fit --rounds 9
```

The plan's acceptance criteria:

- at least nine aligned pairs;
- every case converges;
- the median iteration difference is at most 1;
- relative MAD is at most 10%;
- the candidate slowdown is at most 1.15x.

These are the script's GPU defaults. Also repeat the run once with
`--lifecycle steady`.

### 3b. L1: native CUDA against CuPy under the same transport

There is no earlier native CUDA L1 baseline, so this establishes one:

```powershell
.\.venv\Scripts\python.exe scripts/benchmarks/benchmark_shape_sweep.py `
  --profile standard --backend gpu --penalty l1 --dtype both `
  --lifecycle both --operation both --warmup 3 --repeats 9 `
  --output artifacts/n5-l1-run1.json
```

Run it three times, as `run1` through `run3`, to show the result is
repeatable. Compare native CUDA against CuPy per case, with transport, dtype,
lifecycle, operation, and shape all held equal.

L1 on native CUDA stays explicit opt-in. `backend="auto"` never picks it.
That remains true unless the same-transport native result beats CuPy
repeatably, and the decision goes in the plan either way.

### 3c. Record the evidence

`artifacts/` is ignored, so copy the accepted records into
`benchmarks/baselines/`. Follow the existing naming, for example
`p5-windows-rtx5070ti-native-cuda-l1.json` and
`p5-windows-rtx5070ti-native-cuda-none-ab.json`. Then update these together:

- `docs/native-penalty-completion-plan.md`: the status line and the "Still
  open" list;
- `docs/gpu-performance.md`;
- `docs/native-performance-policy.md`, if a threshold or baseline changes;
- `docs/support-matrix.md`;
- `CHANGELOG.md` under `Unreleased`.

## 4. Stage 3: prepare 0.7.0

Start this only after stages 1 and 2 have merged. The CUDA C ABI changed, so
this is a minor release. Follow the "Release gate" in
[`release-process.md`](release-process.md) and
[`release-checklist.md`](release-checklist.md):

- Bump `src/renewable_huber/_version.py` and both native `pyproject.toml`
  files, including their exact `renewable-huber==` pins.
- Move `Unreleased` in `CHANGELOG.md` to `0.7.0`.
- Update the version and date in the README, `CITATION.cff`, the API, the
  architecture doc, the support matrix, and `SECURITY.md`.
- Replace the checklist's hard-coded `0.6.1` item.

Run `scripts/native/validate_release_artifacts.py --source-only` before
opening the pull request.

An agent stops at the pull request. The maintainer runs the build-only
`release.yml` rehearsal, creates the `v0.7.0` tag, and approves the three PyPI
publish jobs.

## Pull requests

Branch from the latest `main` for each stage, keep each pull request focused,
and put the stage's evidence summary in its description: commit, environment,
and the gate output. The maintainer merges.
