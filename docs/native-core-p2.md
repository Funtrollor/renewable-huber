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

The compatible transport remains contiguous host NumPy data. Raw features are
copied to the selected GPU once and the intercept column is appended in
reusable device workspace, avoiding a second full host-side design-matrix
copy. A separate DLPack hot path accepts C-contiguous CUDA `float32`/`float64`
tensors already resident on the engine device. PyO3 consumes each capsule with
the engine stream, validates dtype/device/shape/strides, and keeps the producer
allocation alive until the CUDA ABI completes its device-to-device copy. There
is no implicit CUDA-to-host fallback. Coefficients, the information matrix,
and reusable workspaces stay allocated on that device across `partial_fit`
calls.

The solver loop is native. Objective/convergence scalars still cross to the
C++ host between iterations, but they use one pinned four-scalar buffer and
paired synchronization. Update and portable state export share the final
stream completion. A library-owned CUDA memory pool amortizes workspace
allocation without changing the process-wide default allocator.

Cold engines initialize only the regular SPD/LU resources. `gesvdjInfo`,
singular values, U/V matrices, and the SVD work buffer are allocated lazily
after LU actually reports a singular system. This preserves the exact
minimum-norm fallback while removing unused SVD setup from ordinary
Cholesky-only fits. The rank-deficient golden case and a CUDA-DLPack
rank-deficient test both exercise this lazy path.

For DLPack input with an intercept, the append kernel reads the producer's X
allocation directly instead of first staging an identical device copy. The
target vector is safely aliased for the duration of the native call; capsule
ownership and the final stream completion guarantee its lifetime. Host input
continues to use owned engine workspace.

The separately built extension reports C ABI version 1 and Python payload API
version 2. The base package checks both before creating an engine, so an
older or unrelated native module fails explicitly instead of reaching a
native method or result-dictionary mismatch later. Compatible builds
additionally advertise `device_input="dlpack"`.

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
.\scripts\native\build_native_cuda.ps1 -Python python
```

The build script enters the Visual Studio x64 environment, asks Maturin to
compile the `cuda` feature, and installs `_renewable_huber_native_cuda`; the
base package shim exposes it as `renewable_huber._native_cuda`. This is a
developer/source build; the base PyPI wheel continues to require neither Rust
nor CUDA.

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
  --lifecycle cold --operation partial-fit `
  --warmup 3 --repeats 9 --output artifacts/native-p2-cold-v2.json

# Capture reusable native/CuPy/NumPy streaming throughput separately.
python scripts/benchmarks/benchmark_shape_sweep.py `
  --profile smoke --backend all --penalty none --dtype both `
  --lifecycle steady --operation partial-fit `
  --warmup 3 --repeats 9 --output artifacts/native-p2-steady-v2.json
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
It is a schema-v1 historical observation, not an eligible dispatch or
regression-gate baseline. Schema v2 now applies the same cold or steady
reset/initialization contract to every engine; see the
[native performance policy](native-performance-policy.md).

The profiled three-repeat reference window was 120.8 ms. It contained 396
device-to-host copies and 401 stream-wait synchronizations; those counts make
device-side convergence and reduction results the next optimization target.
CUDA API time, kernel time, and memcpy time overlap and must not be summed as
wall-clock time.

The optimized engine now uses a device-pointer cuBLAS reduction handle so the
two objective scalars and the two convergence norms are each transferred and
synchronized as one pair. Exactly symmetric, positive-ridge information uses
cuSOLVER Cholesky; non-symmetric state and failed factorization retain the
full-matrix LU and minimum-norm SVD paths. Both solve paths validate cuSOLVER
device status. SVD metadata and workspaces are initialized only if LU confirms
a singular system, while a dedicated rank-deficient DLPack test preserves the
minimum-norm fallback. The PyO3 adapter also marks freshly allocated state
snapshots, allowing the Python backend to avoid a second `p + p²` host copy.

A GPU-loaded diagnostic run reduced the reference float32 native steady time
from the historical 36.0 ms to 19.9 ms, but that run is intentionally not a
publishable comparison: another graphics workload occupied the GPU.

The approved 2026-08-01 schema-v2 cold run used three warmups, nine measured
samples, and 0.5 seconds of fixed block work per sample for all standard
shapes, both dtypes, and both public operations. Native CUDA passed all 16
matched CuPy host-input comparisons and all 16 matched CuPy device-input
comparisons. Host speedup ranged from 1.04x to 1.96x (median 1.35x); DLPack
device speedup ranged from 1.06x to 2.04x (median 1.53x). Maximum Native
relative MAD was 3.62% for host input and 3.28% for device input, below the
unchanged 10% ceiling. The fixed-runner record is
[`p3-windows-rtx5070ti-native-cuda-v2.json`](../benchmarks/baselines/p3-windows-rtx5070ti-native-cuda-v2.json).

The original single-call capture rejected two cold latency pairs because
Windows/WDDM relative MAD exceeded 10%. The runner now uses fixed block
sampling and isolates cyclic GC without trimming or retrying observations.
Sampling policy is part of the comparison key, so that historical capture is
not compared with the approved stabilized baseline.

The acceptance gate is correctness first. Performance results must record the
GPU, driver, toolkit, shape, dtype, solver iterations, transfer policy, thread
configuration, and BLAS provider; they are not portable headline numbers
across machines. Native CUDA must pass the matched CuPy competitor parity gate
under the same host/device transport before a calibration can recommend it.

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
