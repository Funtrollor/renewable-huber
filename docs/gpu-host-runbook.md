# GPU host runbook (Windows)

This is the checklist for the work that only the fixed GPU host can do:
running the native CUDA engine for the first time on a device, the N5
measurements from the
[native penalty completion plan](native-penalty-completion-plan.md#n5--benchmark-documentation-and-release-readiness),
and preparing the 0.7.0 release. It assumes a fresh clone on a Windows
machine with an NVIDIA GPU (the committed baselines were taken on an
RTX 5070 Ti, SM 12.0). Read [`AGENTS.md`](https://github.com/Funtrollor/renewable-huber/blob/main/AGENTS.md) first.

The native CUDA engine is at C ABI 2 / Python API 4. Stage 1 below first ran
on a GPU on 2026-09-29 at `4114918`, and every command passed. Stage 2 ran
the same day; its §3a A/B was accepted on a frozen-plan recapture at
`10fc363` on 2026-09-30. The results, including what is still open, are in
[N5 fixed-host results](native-penalty-completion-plan.md#n5-fixed-host-results).
Wherever that run showed this page disagreeing with what the scripts
actually require, the page now follows the scripts and says why.

## 0. Prerequisites

| Component | Requirement | Check |
|---|---|---|
| NVIDIA driver | supports CUDA 12.9 | `nvidia-smi` shows "CUDA Version" ≥ 12.9 |
| CUDA Toolkit | 12.9 (SM 120 needs ≥ 12.8) | `nvcc --version` |
| Visual Studio 2022 Build Tools | "Desktop development with C++" (MSVC x64) | `vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64` (`vswhere.exe` is in `${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer`, not on `PATH`; without `-products *` it does not list Build Tools) |
| Rust | stable, `x86_64-pc-windows-msvc` | `rustc --version`, `cargo --version` |
| Python | 3.10–3.12 x64; the N5 run used 3.11.0 (see below) | `py -3.11 --version` |
| CMake, Ninja | on `PATH` | `cmake --version`, `ninja --version` |
| Git, GitHub CLI | `gh auth login` done | `gh auth status` |

`.gitattributes` pins every text file to LF, so the golden corpus and the
shell scripts are byte-identical to Linux regardless of `core.autocrlf`.

The candidate supports CPython 3.10–3.13, but the §3a A/B needs the **same**
interpreter version in both venvs (the gate compares the Python version as
part of the fingerprint), and its baseline `fca7b83` declares
`requires-python = ">=3.10,<3.13"`. Use one of 3.10–3.12 for everything on
this host. The commands below use 3.11, which the N5 run used; substitute the
version you have.

## 1. Environment

In PowerShell, from the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev,sklearn,pandas,gpu-cupy]" "maturin>=1.8,<2"

# Both scripts import the VS x64 toolchain into their own process, build a
# wheel and install it into the interpreter given by -Python. They
# Push-Location into native\python-* before running that interpreter, so the
# path must be absolute; a relative .\.venv\... no longer resolves there.
$venvPy = (Resolve-Path .\.venv\Scripts\python.exe).Path
.\scripts\native\build_native_cpu.ps1  -Python $venvPy
.\scripts\native\build_native_cuda.ps1 -Python $venvPy
```

`build_native_cuda.ps1` builds for `RH_CUDA_ARCHITECTURES`, which defaults to
`native` (the local GPU). The CMake and `ctest` commands below need the
compiler on `PATH`. Run them in an x64 developer shell. VS 2022 Build Tools
installs under `${env:ProgramFiles(x86)}` by default, so locate it with
`vswhere` the way the build scripts do, and pass `-SkipAutomaticLocation`:
without it, `Launch-VsDevShell.ps1` changes the current directory and the
relative `native/cuda` and `build/...` paths below stop resolving.

```powershell
$vs = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" `
    -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath
& "$vs\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
```

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
# --verbose shows "rh_cuda_smoke passed", which the binary prints only after
# every named case has passed: positive evidence, not just an exit code.
ctest --test-dir build/static --output-on-failure --verbose

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

Two skips are expected in both `core` and `cuda`, and only these two:
`PyTorchDlpackIntegrationTests` and `TensorFlowDlpackIntegrationTests` in
`tests/test_dlpack_adapters.py`. They need CUDA builds of PyTorch and
TensorFlow, which the extras above do not install. A skip means those two
DLPack paths were not exercised on the device, so say so with the evidence.

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

Measurements need the GPU to themselves as far as work goes: do not run tests,
builds, or a second benchmark at the same time. The maintainer uses this
computer during measurements, so desktop programs that render on the GPU may
keep running; closing them is not required. Instead, record which GPU clients
`nvidia-smi --query-compute-apps=pid,process_name --format=csv` lists before
and after each run, and keep that list with the evidence. Drift of about
2–19% between runs of the same binary is normal on this class of host. Never
claim a difference within about ±10%.

### 3a. `penalty="none"` must not have regressed

This is an interleaved A/B of the pre-L1 engine against the current one:

- **Baseline:** `fca7b83`, the `main` commit before the ABI 2 merge.
- **Candidate:** current `main`.

Each side needs its own clone and its own venv, with the native CUDA
extension built from that clone's own sources. ABI 1 Python code cannot load
an ABI 2 extension, or the reverse. Create the baseline venv with the same
Python version as the candidate's (see §0), and pin NumPy, SciPy and CuPy to
the candidate's versions: the gate fingerprints the Python, NumPy and CuPy
versions, and an unpinned install picks whatever is newest that day.

```powershell
$pins = & .\.venv\Scripts\python.exe -c "import importlib.metadata as m; print(' '.join(f'{p}=={m.version(p)}' for p in ('numpy', 'scipy', 'cupy-cuda12x')))"
git worktree add ..\rh-baseline fca7b83
Push-Location ..\rh-baseline
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,gpu-cupy]" "maturin>=1.8,<2" $pins.Split(' ')
.\scripts\native\build_native_cuda.ps1 -Python (Resolve-Path .\.venv\Scripts\python.exe).Path
Pop-Location

.\.venv\Scripts\python.exe scripts/benchmarks/run_interleaved_benchmark.py `
  --baseline-python ..\rh-baseline\.venv\Scripts\python.exe --baseline-repo ..\rh-baseline `
  --candidate-python .\.venv\Scripts\python.exe --candidate-repo . `
  --output-dir artifacts/n5-none-ab --profile standard --backend gpu `
  --penalty none --dtype both --lifecycle cold --operation partial-fit --rounds 9 `
  --allow-native-version-change --freeze-sample-repetitions
```

Why these arguments:

- **`--backend gpu`, not `native_cuda`.** The interleaved gate hard-codes
  `require_competitor_parity=True`, so every native case needs its matched
  CuPy case under the same transport. With `--backend native_cuda` there is
  none, and every check fails with "candidate lacks matched ... competitor
  case". A consequence is that the gate also enforces native/CuPy ≤ 1.0
  (`--max-competitor-slowdown`).
- **`--allow-native-version-change`.** This option is required for an A/B
  across a native ABI change such as this one (ABI 1 / API 3 against ABI 2 /
  API 4). The gate's hardware and runtime fingerprint includes the native
  `abi_version` and `python_api_version`, so without the option every native
  case fails with `hardware or runtime fingerprint differs` whatever its
  timings. The option drops exactly those two fields; driver, runtime, GPU,
  CuPy, Python and every other fingerprint field must still match. It exists
  only in the interleaved runner, and `gate.json` (schema version 2) records it
  as `allow_native_version_change` together with both sides' versions under
  `native_versions`. `check_performance_regression.py` has no such option and
  still rejects an ABI change against a stored baseline.
- **`--freeze-sample-repetitions`.** Before the first timed round, the runner
  runs one calibration sweep for each variant. It writes a plan that gives
  every case the larger of the two calibrated sample block sizes, which the
  sweep's `--max-sample-repetitions` (default 64) still caps. Every round of
  both variants then uses that plan, so a sample averages about
  `--minimum-sample-seconds` of work and the block size cannot drift between
  rounds. An older baseline checkout cannot read a plan, so in this mode both
  variants run the **candidate's** sweep harness against their own source tree
  (through `RENEWABLE_HUBER_BENCHMARK_SOURCE_ROOT`). The baseline records then
  carry `git_revision` of the baseline tree and
  `benchmark_harness_git_revision` of the candidate. The runner writes
  `calibration/`, `sample-repetitions-plan.json`, `rounds/`, `baseline.json`,
  `candidate.json` and `gate.json` into the output directory.

History: without a frozen plan, each round sized its own sample blocks from
its own warmup. On this host the sizes differed between rounds, and
`merge_round_records` refused to merge them ("sample repetition calibration
changed between interleaved rounds"), so no gate report was written. The
first workaround, `--max-sample-repetitions 1`, merged, but it made every
sample a single short GPU call. On a desktop in use that pushed relative MAD
over 10% in three of the 32 N5 cases. Both attempts are recorded in the plan
as superseded.

The plan's acceptance criteria:

- at least nine aligned pairs;
- every case converges;
- the median iteration difference is at most 1;
- relative MAD is at most 10%;
- the candidate slowdown is at most 1.15x.

These are the script's GPU defaults; do not relax them. If a case misses the
MAD limit, recapture, and raise `--rounds` or `--minimum-sample-seconds` if
needed. Also repeat the run once with `--lifecycle steady`, giving it its own
output directory, for example `--output-dir artifacts/n5-none-ab-steady`, so
it does not overwrite the cold run's records.

Before accepting a run, check its `gate.json`:

- `passed` is `true` and no check lists a reason;
- `allow_native_version_change` is `true`, and `native_versions` shows the
  expected change (here `native_cuda_abi` from ABI 1 / API 3 to ABI 2 /
  API 4);
- `sample_repetitions.policy` reads `frozen_plan`, with `harness` set to
  `candidate`. A different policy means the plan was not used;
- `sample_repetitions.plan_sha256` equals the SHA-256 of
  `sample-repetitions-plan.json` in the same directory.

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
The N5 run met the "repeatable same-transport advantage" condition and the
plan records that, but the decision stayed opt-in: automatic CUDA selection is
outside the plan's scope and needs its own RFC. See
[N5 fixed-host results](native-penalty-completion-plan.md#n5-fixed-host-results).

### 3c. Record the evidence

`artifacts/` is ignored, so copy the accepted records into
`benchmarks/baselines/`, following the existing naming:

- L1 baseline: the three runs, `p5-windows-rtx5070ti-native-cuda-l1-run1.json`
  through `-run3.json`.
- Accepted `none` A/B: for each lifecycle (`cold`, `steady`), the output
  directory's `baseline.json`, `candidate.json`, `gate.json` and
  `sample-repetitions-plan.json`. Name them
  `p5-windows-rtx5070ti-native-cuda-none-ab-<lifecycle>-baseline.json`,
  `-candidate.json`, `-gate.json` and `-sample-repetitions-plan.json`. Keep
  `calibration/` and `rounds/` on the host.

Record the SHA-256 of every record in the plan, including records that stay on
the host because they were not accepted. The shape sweep and the interleaved
runner write their JSON with CRLF on Windows, and `.gitattributes` stores LF,
so a committed record's hash differs from the host copy's; record both. The
plan files are the exception: they are written as bytes with LF, so the
committed file keeps its hash, and that hash must still equal
`sample_repetitions.plan_sha256` in `gate.json`.

`validate_record` in `scripts/benchmarks/performance_policy.py` accepts native
CUDA L1 records whose recorded `native_cuda_abi` shows ABI 2 or later or lists
`l1` in `supported_penalties` (it rejects ABI 1 records that claim L1). Run it
on every committed baseline and candidate record before opening the pull
request.

Then update these together:

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
