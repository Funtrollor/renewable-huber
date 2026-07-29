# Native core P2: CUDA whole-batch engine

P2 moves the complete unpenalized Renewable Huber Newton update behind one
PyO3 call. The existing Python estimator still owns input validation, pandas
feature names, scikit-learn compatibility, public attributes, and portable
checkpoints.

## Implemented boundary

```text
RenewableHuberRegressor
        |
        | one call per partial_fit
        v
Rust/PyO3 NativeCudaEngine
        |
        | narrow C ABI, opaque handle
        v
CUDA C++ engine
  - persistent coefficient and information state
  - persistent cuBLAS/cuSOLVER handles and workspaces
  - fused residual/Huber/curvature and row-weight kernels
  - GEMV/GEMM through cuBLAS
  - pivoted LU solve and singular-system SVD fallback
```

The initial transport is contiguous host NumPy data. A batch is copied to the
selected GPU once, while coefficients, the information matrix, and reusable
workspaces stay allocated on that device across `partial_fit` calls. DLPack
device input is P3 and is deliberately not simulated through an implicit
CuPy-to-host copy.

The solver loop is native, but P2 still copies convergence and line-search
scalars to the C++ host between iterations. Eliminating those synchronization
points with device-side decisions is a measured follow-up optimization, not a
claim of this baseline.

The separately built extension reports both C ABI version 1 and Python payload
API version 1. The base package checks both before creating an engine, so an
older or unrelated native module fails explicitly instead of reaching a
method or result-dictionary mismatch later.

## Numerical contract

The native engine implements the existing strict `float32` and `float64`
unpenalized solver:

- frequency semantics for `sample_weight`;
- the paper bandwidth formula and Huber score/curvature;
- historical information and renewable batch-order semantics;
- monotone backtracking line search;
- relative coefficient convergence;
- finite least-squares-style handling of a rank-deficient Hessian;
- portable NumPy state returned at the estimator/checkpoint boundary.

The P2 engine does not yet implement the L1/LAMM loop. An explicit
`backend="native_cuda", penalty="l1"` request raises a `ValidationError`; it
never silently changes engines. Use `backend="cupy"` for L1 until that loop is
moved in a later phase.

## Build on Windows

Requirements:

- Python 3.10-3.12 and Maturin;
- stable Rust with the MSVC target;
- Visual Studio 2022 C++ Build Tools;
- CMake and Ninja;
- CUDA Toolkit 12.x with cuBLAS and cuSOLVER;
- a compatible NVIDIA driver.

From a PowerShell prompt:

```powershell
python -m pip install "maturin>=1.8,<2"
.\scripts\native\build_native.ps1 -Python python
```

The build script enters the Visual Studio x64 environment, asks Maturin to
compile the `cuda` feature, and installs `renewable_huber._native_cuda` into
the selected Python environment. This is a developer/source build; the base
PyPI wheel continues to require neither Rust nor CUDA.

## Use

```python
from renewable_huber import RenewableHuberRegressor

model = RenewableHuberRegressor(
    backend="native_cuda",
    device="cuda",
    dtype="float32",
    penalty="none",
)
model.partial_fit(X_batch, y_batch, sample_weight=batch_weights)
prediction = model.predict(X_test)  # NumPy array in the host-input P2 adapter
```

`backend="auto"` retains its published behavior. It does not opt users into
an experimental native extension.

One engine supports sequential calls from different Python threads; concurrent
mutation of one estimator is not supported. While `partial_fit` or `predict`
is running, callers must not mutate the same writable NumPy buffers from
another thread because the native call releases the GIL during the host-to-GPU
transfer and solve.

## Verification

Run the CUDA differential tests after building:

```powershell
python -m unittest tests.test_native_cuda_backend -v
python -m unittest discover -s tests -v
```

The native test replays every unpenalized case in
`tests/golden/native_core_v1.json`, including weighted streaming, `float32`
outliers, and the rank-deficient `ridge=0` case. It compares state,
diagnostics, and probe predictions against the frozen NumPy oracle.
The corpus remains unchanged; the CUDA differential test applies
`rtol=1e-8, atol=2e-9` for `float64`, because vendor LU can cross the relative
convergence boundary one or two iterations later than NumPy/LAPACK while
reaching the same objective and information matrix. Iteration counts are not
part of the cross-library equality contract.

Run a comparable smoke benchmark:

```powershell
python scripts/benchmarks/benchmark_shape_sweep.py `
  --profile smoke --backend all --penalty none --dtype both `
  --warmup 1 --repeats 3 --output artifacts/native-p2-shape-sweep.json
```

Capture native whole-update Nsight Systems and Nsight Compute reports:

```powershell
.\scripts\profiling\run_nsight_systems.ps1 `
  -Python python -Engine native_cuda -Penalty none
.\scripts\profiling\run_nsight_compute.ps1 `
  -Python python -Engine native_cuda -Penalty none
```

## P2 measured baseline

The committed [shape sweep](../benchmarks/baselines/p2-windows-rtx5070ti-shape-sweep.json)
and [Nsight Systems summary](../benchmarks/baselines/p2-windows-rtx5070ti-nsys-summary.json)
were captured on an RTX 5070 Ti with CUDA Runtime 12.9. For three steady-state
`float32` repeats, native host input measured:

| Shape | NumPy CPU | CuPy host input | Native host input |
| --- | ---: | ---: | ---: |
| 100,000 x 90 | 77.5 ms | 30.4 ms | 36.0 ms |
| 16,384 x 256 | 2,288.6 ms | 81.4 ms | 47.0 ms |
| 1,000,000 x 32 | 208.2 ms | 71.9 ms | 82.4 ms |

This is a correctness-first native baseline, not a universal speedup claim.
It wins on the wide `float32` case and the reference `float64` case, but trails
CuPy for the `float32` reference and long streaming cases. Small GPU workloads
remain slower than NumPy because launch and transfer overhead dominate.

The native figures are steady-state measurements: one engine is primed once
and restored outside each timed repeat. The NumPy and CuPy entries construct a
new estimator in every timed repeat. These policies are recorded per result in
the JSON and the table must not be read as an initialization-cost comparison.

The profiled three-repeat reference window was 120.8 ms. It contained 396
device-to-host copies and 401 stream-wait synchronizations; those counts make
device-side convergence and reduction results the next optimization target.
CUDA API time, kernel time, and memcpy time overlap and must not be summed as
wall-clock time.

The acceptance gate is correctness first. Performance results must record the
GPU, driver, toolkit, shape, dtype, solver iterations, transfer policy, and
git revision; they are not portable headline numbers across machines.

## Rollback and lifetime rules

- Destruction of `NativeCudaEngine` releases device allocations and CUDA
  library handles exactly once.
- Python exceptions are raised from stable status codes and engine-owned error
  text; C++ exceptions and Rust panics do not cross the ABI.
- A failed first update does not mark the Python estimator as fitted; a native
  initialization failure is reported as `BackendUnavailableError`.
- Loading a checkpoint creates a fresh engine and restores the portable state
  before prediction or the next update.
- Failure to import or initialize an explicitly requested native backend is a
  `BackendUnavailableError`, not an implicit CuPy/NumPy fallback.
