# renewable-huber

English | [繁體中文](README.md)

<!--
Each region between a pair of start/end snippet markers below is
included verbatim in the MkDocs site (docs/index.md and docs/guide/*.md), so
editing one updates both. Keep relative links out of those regions: they
resolve against the including page there, and `mkdocs build --strict` fails.
-->

[![CI](https://github.com/Funtrollor/renewable-huber/actions/workflows/ci.yml/badge.svg)](https://github.com/Funtrollor/renewable-huber/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/renewable-huber.svg)](https://pypi.org/project/renewable-huber/)
[![Python versions](https://img.shields.io/pypi/pyversions/renewable-huber.svg)](https://pypi.org/project/renewable-huber/)
[![GitHub Release](https://img.shields.io/github/v/release/Funtrollor/renewable-huber)](https://github.com/Funtrollor/renewable-huber/releases/latest)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-D22128.svg)](LICENSE)

<!-- --8<-- [start:intro] -->
`renewable-huber` is a Renewable Huber Regression package for streaming data. It implements robust linear regression based on the Huber loss, and while processing batches it retains only the coefficients and an accumulated information matrix, never every historical observation.

The latest version is **0.7.1**, published on [PyPI](https://pypi.org/project/renewable-huber/), but the package is still in **pre-alpha** development. It provides RHE and L1-penalised RPSHE updates on NumPy/CPU, Rust/Rayon native CPU, CuPy/CUDA, Rust/CUDA native, PyTorch and TensorFlow (CPU/CUDA), resumable `.npz` checkpoints, and integration with pandas and scikit-learn Pipeline/model-selection tools. Run `renewable-huber --version` to see the installed version.
<!-- --8<-- [end:intro] -->

<!-- --8<-- [start:auto-dispatch] -->
`backend="auto"` follows predictable device rules: it chooses CuPy only when `device="cuda"` is set explicitly and otherwise stays on the CPU. On the CPU, `auto` still defaults to NumPy; it switches to `native_cpu` only when the batch is large enough and a runtime measurement on this machine shows that the Rust native CPU engine is clearly faster. The decision rests entirely on live measurements of **the current host and the current execution environment**: it never reads a CPU model string and never writes a cache file. Measurements live only in memory and are valid only for the execution environment in which they were taken: they are discarded as soon as the CPU affinity mask (not just the core count), the `*_NUM_THREADS` settings, or the effective BLAS/OpenMP thread-pool sizes observable through optional `threadpoolctl` change, and a child process after `fork` also clears them and measures again. Within one execution environment the same shape always gets the same answer, regardless of what other estimators asked before. Any failure along the way (a missing extension, an engine that cannot be constructed, insufficient measurements, or any ordinary exception raised by a native engine that `auto` selected) quietly falls back to NumPy.
<!-- --8<-- [end:auto-dispatch] -->

For details and cost bounds, see the [CPU auto-dispatch RFC](docs/cpu-auto-dispatch-rfc.md).

<!-- --8<-- [start:auto-frameworks] -->
`auto` does not guess a backend from PyTorch or TensorFlow tensors passed in; set `backend="torch"` or `backend="tensorflow"` explicitly when you need those frameworks. Explicitly setting `backend="numpy"` or `backend="native_cpu"` never triggers the measurement above.
<!-- --8<-- [end:auto-frameworks] -->

For the full support scope, see the [support matrix](docs/support-matrix.md).

## Installation

Requires Python 3.10–3.13. The base installation depends only on NumPy:

```powershell
python -m pip install renewable-huber
renewable-huber --version
```

### Optional Rust CPU core

<!-- --8<-- [start:native-cpu-install] -->
The native CPU core is distributed separately from the pure-Python base package. Once
`renewable-huber-native-cpu` is installed or built locally, you can select it
explicitly, or let CPU `auto` select it when its conservative measurement passes.

Install the matching native wheel directly; it depends on exactly the same version of the base package:

```powershell
python -m pip install renewable-huber-native-cpu==0.7.1
```

The 0.7.1 release wheels cover CPython 3.10–3.13, Windows x86-64, Linux
x86-64/aarch64 and macOS x86-64/Apple Silicon. Regular users do not need to install Rust or
compile the extension locally.
<!-- --8<-- [end:native-cpu-install] -->

<!-- --8<-- [start:native-cpu-usage] -->
```python
from renewable_huber import RenewableHuberRegressor

native_model = RenewableHuberRegressor(
    backend="native_cpu",
    device="cpu",
    dtype="float64",
    n_jobs=-1,  # or a positive integer, e.g. n_jobs=8
)
native_model.fit(X_train, y_train)
```

`n_jobs=None` uses the native extension's default Rayon pool, `n_jobs=-1` uses every
logical CPU, and a positive integer builds a fixed-size, dedicated thread pool for this
estimator. After fitting, `native_model.n_jobs_` reports the actual number of workers.
If an outer layer already parallelises several models with
joblib/`GridSearchCV(n_jobs=...)`, set the inner estimator to
`n_jobs=1` so that nested thread pools do not compete for the CPU.
<!-- --8<-- [end:native-cpu-usage] -->

It supports `penalty="none"` and `penalty="l1"`, with C-contiguous NumPy
`float32`/`float64` input. `backend="auto"` on the CPU may choose this engine, but only
when the batch is large enough and the local measurement supports it; to always use it,
still set `backend="native_cpu"` explicitly. For build, correctness and benchmark details see
[Native-core P1](docs/native-core-p1.md); for the dispatch rules see the
[CPU auto-dispatch RFC](docs/cpu-auto-dispatch-rfc.md).

<!-- --8<-- [start:extras] -->
Install the extra that matches your use case:

| Use case | Install command |
| --- | --- |
| pandas input | `python -m pip install "renewable-huber[pandas]"` |
| scikit-learn adapter | `python -m pip install "renewable-huber[sklearn]"` |
| CuPy / CUDA 12 | `python -m pip install "renewable-huber[gpu-cupy]"` |
| PyTorch | `python -m pip install "renewable-huber[gpu-torch]"` |
| TensorFlow | `python -m pip install "renewable-huber[gpu-tensorflow]"` |
<!-- --8<-- [end:extras] -->

The GPU extras install only the corresponding framework; they do not install the NVIDIA driver or
the CUDA runtime for you. First check compatibility between the framework, operating system,
Python and GPU driver; see the [support matrix](docs/support-matrix.md) for detailed limits.

For development from source:

```bash
python -m pip install -e ".[dev]"
```

## Quick start

<!-- --8<-- [start:quickstart] -->
```python
import numpy as np
from renewable_huber import RenewableHuberRegressor

model = RenewableHuberRegressor(penalty="l1", lambda_scale=0.5)

for X_batch, y_batch in data_stream:
    model.partial_fit(X_batch, y_batch)

print(model.coef_, model.intercept_)
prediction = model.predict(X_test)
model.save("checkpoints/model.npz")

restored = RenewableHuberRegressor.load("checkpoints/model.npz")
assert np.allclose(prediction, restored.predict(X_test))
```
<!-- --8<-- [end:quickstart] -->

<!-- --8<-- [start:cupy] -->
To run on a GPU, install the CUDA 12 CuPy extra and keep batches and state on CUDA:

```python
import cupy as cp

gpu_model = RenewableHuberRegressor(backend="cupy", device="cuda", dtype="float32")
gpu_model.partial_fit(cp.asarray(X_batch), cp.asarray(y_batch))
gpu_prediction = gpu_model.predict(cp.asarray(X_test))  # cupy.ndarray, not copied back to the CPU
```
<!-- --8<-- [end:cupy] -->

<!-- --8<-- [start:native-cuda-install] -->
The CUDA 12 plugin wheel can be installed directly. 0.7.1 provides Windows x86-64 and Linux
x86-64 (`manylinux_2_28`) wheels for CPython 3.10–3.13:

```powershell
python -m pip install renewable-huber-native-cuda==0.7.1
```

The wheel already contains the native extension compiled for the supported GPU architectures, so
Rust, CMake, Visual Studio and a local `nvcc` are not needed. The wheel itself does not bundle the
NVIDIA libraries: the CUDA 12 runtime closure (`cudart`, cuBLAS/cuBLASLt, cuSOLVER, cuSPARSE and
nvJitLink) is installed automatically as dependencies from NVIDIA's official `nvidia-*-cu12`
wheels, and that set is loaded first at import time. Only when the set is incomplete does it fall
back to the system toolkit (`CUDA_PATH` on Windows, the default loader path on Linux), so
`CUDA_PATH` does not need to be set. A compatible NVIDIA driver is still required.
<!-- --8<-- [end:native-cuda-install] -->

<!-- --8<-- [start:native-cuda-usage] -->
With the separate native CUDA extension installed, you can explicitly select the Rust/CUDA
whole-batch engine, which consumes CuPy, PyTorch CUDA or TensorFlow eager GPU batches directly
through DLPack with no host staging at any point:

```python
native_gpu = RenewableHuberRegressor(
    backend="native_cuda",
    device="cuda",
    dtype="float32",
    penalty="l1",  # "none" or "l1"; both are solved whole-batch on the GPU
    cuda_graphs=True,  # opt-in; falls back safely when capture is unavailable
    cuda_fast_math=False,  # opt-in TF32; strict float32 by default
)
native_gpu.partial_fit(cp.asarray(X_batch), cp.asarray(y_batch))
gpu_prediction = native_gpu.predict(cp.asarray(X_test))  # reads device X in place, returns NumPy
```

The 0.7.1 native CUDA engine (C ABI 2/Python API 4) supports `penalty="none"` and
`penalty="l1"`; L1 uses the same LAMM proximal-gradient update as NumPy and Rust CPU, and
checkpoints can be resumed across all three engines.

The native CUDA Python API also offers opt-in `cuda_graphs=True`, and a `float32`-only
`cuda_fast_math=True` (TF32) high-speed mode. Both are off by default; when a CUDA Graph
cannot be captured safely, execution falls back to the regular path. After fitting, inspect
`native_gpu.cuda_features_` for the actual enabled state and the capture, replay and fallback counts.

All device batch inputs must reside on the same GPU, have exactly the same dtype and be
C-contiguous; anything else raises an error immediately instead of being copied implicitly.
CuPy/PyTorch use DLPack consumer-stream negotiation; the TensorFlow legacy DLPack adapter first
performs the necessary producer synchronization, but still does not copy storage. From API 4,
`predict` also accepts CUDA DLPack tensors that meet the same conditions: the engine reads X in
place on its own stream (appending the intercept column on the device when needed), only the
`n_rows` predictions come back to the host, and the return type remains `numpy.ndarray`.

<!-- --8<-- [end:native-cuda-usage] -->

<!-- --8<-- [start:torch-tensorflow] -->
PyTorch can use native `torch.Tensor` objects on the CPU or on an explicitly specified CUDA device:

```python
import torch

torch_model = RenewableHuberRegressor(backend="torch", device="cuda", dtype="float32")
torch_model.partial_fit(
    torch.as_tensor(X_batch, device="cuda"), torch.as_tensor(y_batch, device="cuda")
)
torch_prediction = torch_model.predict(
    torch.as_tensor(X_test, device="cuda")
)  # detached torch.Tensor
```

The TensorFlow backend uses eager execution and likewise supports native `tf.Tensor` objects:

```python
import tensorflow as tf

tensorflow_model = RenewableHuberRegressor(backend="tensorflow", device="cuda", dtype="float32")
tensorflow_model.partial_fit(
    tf.convert_to_tensor(X_batch),
    tf.convert_to_tensor(y_batch),
)
tensorflow_prediction = tensorflow_model.predict(tf.convert_to_tensor(X_test))  # tf.Tensor
```
<!-- --8<-- [end:torch-tensorflow] -->

<!-- --8<-- [start:inputs] -->
The scikit-learn adapter works directly with Pipeline, clone and GridSearchCV:

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from renewable_huber.integrations.sklearn import SklearnRenewableHuberRegressor

pipeline = make_pipeline(StandardScaler(), SklearnRenewableHuberRegressor())
pipeline.fit(X_train, y_train)
prediction = pipeline.predict(X_test)
```

`numpy.ndarray` and tabular objects with `.to_numpy()` (such as `pandas.DataFrame` / `Series`) can be used directly as input. When a DataFrame has string column names, later DataFrame batches and predictions must keep the same names in the same order. `fit`, `partial_fit` and `score` support a non-negative `sample_weight`:

```python
model.partial_fit(X_batch, y_batch, sample_weight=batch_weights)
weighted_r2 = model.score(X_test, y_test, sample_weight=test_weights)
```

Weights have frequency-weight semantics: an integer weight is equivalent to repeating the observation, and an all-zero batch is rejected. A SciPy sparse matrix is never densified behind your back; assess the memory cost and call `X.toarray()` explicitly.
<!-- --8<-- [end:inputs] -->

<!-- --8<-- [start:checkpoint-migration] -->
A checkpoint restores its original backend by default, and can also be migrated explicitly to the CPU or to another dtype:

```python
cpu_model = RenewableHuberRegressor.load(
    "checkpoints/gpu-model.npz",
    backend="numpy",
    device="cpu",
    dtype="float64",
)
```
<!-- --8<-- [end:checkpoint-migration] -->

<!-- --8<-- [start:streaming-notes] -->
`fit(X, y)` resets the model and then processes a single batch; a genuine streaming workflow should call `partial_fit(X_batch, y_batch)` repeatedly.

PyTorch input is `detach`ed first, and this package is not an autograd layer; the TensorFlow backend supports eager execution only and cannot be placed directly inside a `tf.function`. Streaming updates use the previous batch's coefficients and information matrix, so the way batches are split and the order of the data are part of the computation; results are not guaranteed to be bit-identical to a whole-data `fit` or to a different permutation.
<!-- --8<-- [end:streaming-notes] -->

## Project layout

```text
src/renewable_huber/     # publishable package source
native/                  # Rust workspace, PyO3 bindings and CUDA C ABI/C++ engine
tests/                   # unit tests that need no external data
docs/                    # API contract, architecture and release checklists
scripts/renewable_huber/ # reproducible dataset experiment scripts
scripts/benchmarks/      # shape sweep, interleaved A/B and dispatch benchmarks
benchmarks/baselines/    # reviewed, schema-versioned small performance baselines
legacy/                  # pre-refactor prototypes, kept only for result comparison, never published
data/                    # local research data, never packaged or uploaded to PyPI
```

## Native performance baselines

In the schema-v2 cold baseline on a fixed Ryzen 9 9900X with a 24-thread Rayon pool, Rust CPU
is **1.38×–7.86×** (median 1.81×) relative to NumPy across 32 shape/dtype/penalty/operation
pairs. In the matched cold baseline on a fixed RTX 5070 Ti, native CUDA relative to CuPy on the
same transport is **1.04×–1.96×** (median 1.35×) for host input and **1.06×–2.04×**
(median 1.53×) for DLPack device input.

These are approved results for specific hardware, BLAS, driver, shape, lifecycle and transport,
not a guarantee for every computer. The GPU can drift by about 2%–19% for the same binary on this
host, so a difference below about 10% should not be read as a definite speed-up; formal
comparisons must use the paired interleaved runner. The raw JSON and the measurement contract are
in the [native performance policy](docs/native-performance-policy.md).

## Documentation and research sources

- [Public API and state contract](docs/api.md)
- [Support matrix and limitations](docs/support-matrix.md)
- [Package architecture and compute paths](docs/architecture.md)
- [CUDA performance paths](docs/gpu-performance.md)
- [Rust/CUDA native-core RFC](docs/native-core-rfc.md)
- [Native-core P0 performance baseline](docs/native-core-p0-baseline.md)
- [Native-core P2 CUDA engine](docs/native-core-p2.md)
- [Native-core P4 CUDA tuning](docs/native-core-p4.md)
- [Pre-release checklist](docs/release-checklist.md)
- [Versioning and GitHub Release process](docs/release-process.md)
- [Contributing guide](CONTRIBUTING.md)
- [Security policy](SECURITY.md)
- [Changelog](CHANGELOG.md)
- The technical report `docs/reports/Technical_Report.pdf` is local project material, deliberately excluded from the Git repository and the published packages; please ask the project maintainers for it.
- [Original Renewable Huber paper (Electronic Journal of Statistics, DOI)](https://doi.org/10.1214/24-EJS2223)

This project is an independent software implementation written from the method in the paper above; it is not an official package created, sponsored, approved or endorsed by the paper's authors or their institutions. The research article is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); this project's source code is licensed under the [Apache License 2.0](LICENSE), and detailed attribution notices are in [NOTICE](NOTICE).

## Development and verification

```bash
python scripts/run_test_profile.py --check
python scripts/run_test_profile.py core --verbose
python scripts/run_test_profile.py native-cpu --verbose
python -m ruff check src tests scripts
python -m ruff format --check src tests scripts
python -m build
```

GPU correctness and performance run only on a fixed local GPU host, not in regular GitHub
Actions; without a real device the required `cuda` profile exits with status 2 rather than
pretending to succeed with every test skipped. The full WSL2/Linux setup and acceptance commands
are in [CONTRIBUTING.md](CONTRIBUTING.md).

The GitHub repository is set up as `Funtrollor/renewable-huber`. Please use the repository's templates for bug reports, feature proposals and pull requests; report security vulnerabilities privately as described in [SECURITY.md](SECURITY.md). Versions are driven by Git tags into GitHub Releases and published through PyPI Trusted Publishing (OIDC); no long-lived PyPI token is stored in the repository.
