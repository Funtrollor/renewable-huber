# Native core P2: CUDA whole-batch engine

> This document preserves the P2 delivery history. In 0.6.1 the Python payload
> API was version 3, after P4 added capability metadata, and the C ABI was
> version 1. 0.7.0 raised them to C ABI 2 and payload API 4.

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

The original P2 extension reported C ABI version 1 and Python payload API
version 2; 0.6.1 reported C ABI 1 and payload API 3. The current extension
reports C ABI version 2 and payload API 4.
The base package checks both before creating an engine, so an
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

The P2 engine did not implement the L1/LAMM loop, and an explicit
`backend="native_cuda", penalty="l1"` request raised a `ValidationError`. C ABI
2 / Python API 4 adds that loop; see `docs/native-penalty-completion-plan.md`.
An extension that does not advertise `l1` in `supported_penalties` is still
refused before any native call, never silently changed to another engine.

## Build on Windows

Requirements:

- Python 3.10-3.13 and Maturin;
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

`backend="auto"` never opts users into native CUDA. CPU auto-dispatch may
independently choose native CPU when its host-local performance gate passes.

One engine supports sequential calls from different Python threads; concurrent
mutation of one estimator is not supported. While `partial_fit` or `predict`
is running, callers must not mutate the same writable NumPy buffers from
another thread because the native call releases the GIL during the host-to-GPU
transfer and solve.

## Verification

Run the CUDA differential tests after building:

```powershell
python scripts/run_test_profile.py cuda --verbose
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

The committed [shape sweep](https://github.com/Funtrollor/renewable-huber/blob/main/benchmarks/baselines/p2-windows-rtx5070ti-shape-sweep.json)
and [Nsight Systems summary](https://github.com/Funtrollor/renewable-huber/blob/main/benchmarks/baselines/p2-windows-rtx5070ti-nsys-summary.json)
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
[`p3-windows-rtx5070ti-native-cuda-v2.json`](https://github.com/Funtrollor/renewable-huber/blob/main/benchmarks/baselines/p3-windows-rtx5070ti-native-cuda-v2.json).

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

## P7 fixed-host breakdown (C ABI 2 / API 4)

On 2026-10-09 the current engine (`f2dc7cb`, unchanged at the profiled
`4373a82`) was profiled on the fixed RTX 5070 Ti host: driver 616.64, CUDA
12.9, CPython 3.11.0, NumPy 2.4.6, CuPy 14.2.0, Nsight Systems 2025.3.2, PCIe
5.0 x16. The desktop was in use; each capture's `nvidia-smi` compute clients
and top CPU processes were recorded before and after it. Each configuration
ran `scripts/profiling/run_nsight_systems.ps1 -Engine native_cuda` with the
shape sweep's standard shapes, two warmups and three resident-engine repeats
of the whole stream. `--phase-ranges` marks the estimator's prepare, update and
commit phases, and `summarize_nsys_sqlite.py` (schema 2) assigns every instant
of every `partial_fit` to exactly one class, so the classes add up to its wall
time. The summaries are committed as
`benchmarks/baselines/p7-windows-rtx5070ti-nsys-<configuration>.json`.

| Configuration | ms per stream | Iterations per batch | GPU compute | H2D | Launch and sync¹ | Python and binding (input validation) | Launches per iteration | Syncs per iteration | Line-search candidates per iteration | H2D API / DMA per batch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| latency f32 none host | 2.83 | 5.00 | 16% | 2% | 79% | 4% (2%) | 56 | 2.8 | 2.20 | 0.038 / 0.007 ms |
| reference f32 none host | 11.36 | 3.75 | 24% | 14% | 44% | 18% (15%) | 52 | 2.6 | 2.00 | 0.390 / 0.337 ms |
| reference f32 none device | 9.28 | 3.75 | 30% | 0% | 56% | 14% (11%) | 52 | 2.6 | 2.00 | — |
| reference f64 none host | 37.44 | 3.75 | 70% | 8% | 14% | 8% (7%) | 50 | 2.5 | 1.87 | 0.769 / 0.703 ms |
| reference f32 l1 host | 18.98 | 8.00 | 15% | 8% | 66% | 11% (9%) | 45 | 2.3 | 2.06 | 0.380 / 0.329 ms |
| streaming f32 none host | 38.36 | 2.31 | 14% | 18% | 46% | 23% (19%) | 53 | 2.8 | 1.89 | 0.404 / 0.348 ms |
| wide f32 none host | 11.05 | 3.00 | 31% | 7% | 52% | 10% (8%) | 68 | 3.8 | 3.00 | 0.177 / 0.132 ms |
| wide f32 none host, `cuda_graphs=True` | 8.53 | 3.00 | 31% | 10% | 45% | 14% (10%) | 34 | 3.8 | 3.00 | 0.185 / 0.138 ms |

¹ Includes device-to-host and device-to-device copies and memsets, which are
0.3%–1.2% in every configuration.

Nsight inflates API time: profiled latency costs 0.565 ms per iteration, the
unprofiled p5 sweep about 0.47 ms. The shares are for prioritizing, not a
prediction of the unprofiled speedup.

What the breakdown shows:

- **Launch and sync latency dominates every `float32` configuration (44%–79%).**
  A Newton iteration issues 45–68 kernel launches at about 5 µs each under
  Nsight, 0.28–0.32 ms of API time, while the GPU sits idle between them.
- **The line search evaluates 1.9–3.0 candidates per iteration, and each costs
  about 21 API calls and a stream synchronization.** Two patterns produce
  them. The first iteration of a stream starts from zero coefficients, and its
  Newton step overshoots far: reference needs 8 step sizes, wide 11 and then
  9, latency 4 and 4. In warm `float32` batches the second iteration often
  backtracks one to five more times, because the `float32` cuBLAS `asum`
  objective cannot resolve the remaining decrease. Together the extra
  candidates are roughly 12%–28% of the wall time.
- **Pageable host-to-device copies are not the bottleneck they look like.**
  The API time per batch is only 13%–16% above the DMA time (23 GB/s against
  27 GB/s on reference), so the driver's own staging is already efficient. H2D
  is 14%–18% of host-input `float32` time.
- **`float64` is compute-bound.** Kernels are 70% of its wall time; one
  `float64` Gram GEMM takes 980 µs.
- **The fixed cost of a `partial_fit` call is small.** The Python core and the
  PyO3 entry before the first CUDA call take 0.011–0.033 ms, decoding the
  result 0.015–0.032 ms and committing it 0.009–0.018 ms. The final
  synchronization waits 2–3 µs, and the state export's device-to-host copy is
  within the under-1.2% "other copies" share. Input validation
  (`np.isfinite` over the batch) is most of the Python column at 0.21–0.65 ms
  per batch; it is outside this work and only recorded.
- `cuda_graphs=True` halves the launches on wide and cuts its stream time by
  23% under Nsight.

### P7 optimizations: what was adopted, and what was not

The breakdown named four candidates. One was adopted.

**Adopted: fused line-search rounds** (`1815ddc`, `a6f9fe6`, `238df5b`,
`67d60bb`). A round evaluates up to four Newton steps, or four LAMM
curvatures, in two kernels:

- the first forms the candidates and reduces their coefficient-space terms
  (history quadratic, step and coefficient norms, the L1 inner products);
- the second computes every candidate's residuals and Huber loss in one pass
  over X, and its last block folds all partial sums.

The host pays one parameter copy, two launches, one transfer and one
synchronization per round instead of about 21 API calls and a
synchronization per candidate, and accepts the first candidate in step order
exactly as the sequential search did. The accepted candidate's residuals feed
the next gradient directly.

Three refinements followed measurements:

- a full-width round made streaming `float64` 12.7% slower in the first cold
  A/B, because `float64` is compute-bound there and Newton almost always
  accepts the full step. A Newton search now opens with the full step alone,
  unless it starts from an empty state or the previous iteration backtracked;
  LAMM opens with two curvatures. The width never changes a value;
- Nsight then put the round's residual kernel at 290 µs per call on
  streaming `float64`, against 64 µs for the cuBLAS GEMV it replaced: with a
  warp per row, the shuffle tree spends a warp-wide `float64` add per level
  and candidate. Up to 64 coefficients a thread now owns its row; the kernel
  takes 40 µs there;
- the loss sum takes absolute values like the `asum` it replaced, which only
  matters for a negative weight passed through the raw C ABI.

`cuda_graphs=True` now captures the round, so the graph path stays
bit-identical to the stream path; Nsight needs `--cuda-graph-trace=node` to
import such a capture, and `run_nsight_systems.ps1` passes it.

**Numerics.** Every sum across rows or coefficients accumulates in double in
a fixed order over a grid that depends only on the batch shape, so repeated
runs are bit-identical and a candidate's value never depends on which round it
shared. Against `main` the results change at rounding level, because the
objective no longer comes from `float32` cuBLAS reductions. On the 16 standard
workloads `float64` iteration counts are unchanged and coefficients agree to
about 1e-7 relative or better; `float32` unpenalized counts are unchanged except wide,
one iteration fewer. `float32` L1 counts move, as they already did between
engines: `float32` at `tol=1e-6` sits at the rounding floor. Over 20 dataset
seeds per shape the mean change in total iterations is −0.3 (latency), +0.2
(reference), +1.5 (wide) and +4.9 (streaming, 69.2 → 74.0; NumPy needs 67.6).
Accumulating the narrow rows' dot products in double was tried and made
streaming worse (+10.2), so it was reverted.

**Final A/B against `main`** (`f2dc7cb` → `5e7a45d`, frozen plan,
`--rounds 9 --minimum-sample-seconds 1.0`, standard profile, both penalties,
both dtypes, host and device input):

| Lifecycle | Speedup range | Median | Cases > 10% faster | Cases slower |
|---|---:|---:|---:|---:|
| cold | 1.00x–2.34x | 1.58x | 29 of 32 | 0 |
| steady | 1.01x–4.30x | 1.69x | 30 of 32 | 0 |

Per shape, the cold medians are latency 1.61x, reference 1.25x, wide 1.90x
and streaming 1.40x; the steady medians are latency 2.23x, reference 1.26x, wide
1.87x and streaming 1.46x. Native CUDA stays faster than CuPy under
the same transport in every case, and no case is slower than `main`.

Neither strict gate (`gate.json`) passes. In cold the only failing check is
the iteration difference of the four `float32` L1 cases below, above the
one-iteration limit. Steady fails the same four, plus relative MAD above 10%
in three cases: two on the baseline side (latency f64 L1 host 19.1%,
streaming f32 unpenalized host 14.6%) and one on the candidate side (streaming
f32 L1 device 12.5%). A steady recapture with a larger repetition cap was
started and stopped on the maintainer's instruction, so those three stand as
measured.

On the maintainer's instruction the iteration limit was then relaxed for this
A/B only: `gate-max-iteration-delta-11.json` beside each strict gate recomputes
it from the same merged records with `max_iteration_delta=11`, the smallest
value covering the observed differences, and every other threshold at the
runner default. Nothing was remeasured and the runner's default stays 1. The
relaxed cold gate passes; the relaxed steady gate fails only the three MAD
cases. The time per iteration in the four cases is:

| Case | Iterations main → candidate | Cold ms per iteration | Steady ms per iteration |
|---|---:|---:|---:|
| streaming f32 L1, device input | 52 → 63 | 0.781 → 0.275 | 0.609 → 0.252 |
| streaming f32 L1, host input | 52 → 63 | 0.961 → 0.434 | 0.797 → 0.385 |
| wide f32 L1, device input | 49 → 51 | 0.549 → 0.260 | 0.432 → 0.178 |
| wide f32 L1, host input | 49 → 51 | 0.553 → 0.240 | 0.458 → 0.193 |

**After the change**, the same Nsight configurations
(`p7-windows-rtx5070ti-nsys-after-<configuration>.json`) show:

| Configuration | ms per batch before → after | Launches per iteration | Launch and sync share |
|---|---:|---:|---:|
| latency f32 none host | 2.83 → 1.22 | 56 → 21 | 78% → 63% |
| reference f32 none host | 2.84 → 2.51 | 52 → 23 | 43% → 31% |
| reference f32 none device | 2.32 → 1.97 | 52 → 23 | 55% → 43% |
| reference f64 none host | 9.36 → 9.77 | 50 → 22 | 14% → 15% |
| reference f32 l1 host | 4.75 → 2.64 | 45 → 9 | 65% → 38% |
| streaming f32 none host | 2.40 → 1.92 | 53 → 25 | 45% → 36% |
| wide f32 none host | 2.76 → 2.07 | 68 → 24 | 51% → 35% |
| wide f32 none host, graphs | 2.13 → 2.02 | 34 → 23 | 44% → 31% |

**Not adopted:**

- **Pinned staging for host input (D).** Pageable copies already run within
  13%–16% of their DMA time, so a pinned buffer could save about 0.05 ms per
  reference batch, and filling it costs a host `memcpy` of the same size.
  Not implemented.
- **Fusing the remaining small kernels (C).** After the rounds, an iteration
  issues about 21 calls, most of them inside cuSOLVER `potrf`/`potrs` and
  cuBLAS. The fusions that keep results bit-identical (Hessian assembly with
  its factor copy, the gradient-delta subtraction) remove three or four, an
  estimated 8% of a latency iteration at most, below the 10% acceptance
  threshold. Not implemented.
- **A whole-iteration CUDA Graph (A).** After the rounds, wide with
  `cuda_graphs=True` measures 2.02 ms per batch against 2.07 ms without: the
  graph's advantage was the candidate DAG the rounds now fuse. Capturing the
  rest would mean capturing cuSOLVER for an opt-in path only. Not
  implemented.

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
