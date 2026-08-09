# RFC 0001: Native Rust and CUDA core

- Status: Delivered and accepted
- Scope: P0 contracts and measurement baseline
- Delivered milestone: 0.6.1 native engine distributions
- Public API owner: `renewable_huber.estimator`

## Decision

Renewable Huber will retain its Python estimator, validation, integration, and
checkpoint layers. The numerical `renewable_update` boundary will move behind
a native dispatcher with two engines:

```text
Python estimator / validation / serialization
                  |
          PyO3 native dispatcher
             /             \
Rust CPU engine              Rust-owned CUDA FFI
BLAS/LAPACK                   |
                       CUDA C++ engine
                       cuBLAS/cuSOLVER
```

Rust owns the portable algorithm contract, CPU solver, resource lifetimes, and
Python binding. CUDA C++ owns the CUDA hot path and exposes only a narrow C ABI
with opaque handles. Matrix products and dense solves remain delegated to
vendor libraries; custom kernels are limited to operations that benefit from
fusion, such as Huber terms, row weighting, reductions, and layout conversion.

The existing Python implementation remains the reference oracle and fallback
until the native engines have passed the correctness and performance gates in
this RFC.

## Motivation

The current solver in `src/renewable_huber/core/update.py` performs the
following work in each unpenalized Newton iteration:

- `X @ beta` and `X.T @ score`, each `O(n p)`;
- `X.T @ (curvature[:, None] * X)`, `O(n p^2)`;
- a dense solve, `O(p^3)`;
- one or more objective evaluations during line search, each `O(n p)`.

The L1 path avoids the Hessian but may evaluate the objective repeatedly during
backtracking. On CUDA, scalar objective and norm conversions currently cross
the device/host boundary inside the iteration loop. Moving the whole update
loop behind one native call creates room to reuse allocations, retain state on
device, and reduce synchronization without changing the statistical method.

## Goals

1. Preserve the estimator's numerical and streaming semantics.
2. Remove Python dispatch from the inner Newton and LAMM loops.
3. Accept contiguous NumPy arrays without copying on CPU.
4. Accept CUDA tensors through a stream-correct DLPack path without copying.
5. Keep coefficients, information matrix, handles, and workspaces resident on
   the selected CUDA device across batches.
6. Support strict `float32` and `float64` execution.
7. Publish installable CPU and CUDA 12 wheels whose distributions share the
   base release version and depend on that exact version.
8. Measure speed with reproducible records rather than single headline
   timings.

## Non-goals

- Changing the public `RenewableHuberRegressor` API during the native migration.
- Changing the paper equations, batch-order semantics, or L1 penalty.
- Making the estimator a PyTorch autograd layer.
- Supporting TensorFlow graph mode in the first native release.
- Reimplementing GEMM, Cholesky, QR, or least-squares kernels.
- Enabling TF32, FP16, BF16, or Tensor Core reduced precision by default.
- Requiring CUDA or a Rust toolchain for the base Python installation.

## Compatibility contract

The following behavior is frozen before native work begins:

- `sample_weight` has frequency-weight semantics.
- Batch boundaries and arrival order are semantically significant.
- The intercept is excluded from the L1 penalty.
- The historical subgradient uses `previous_lambda`.
- `fit` resets state; `partial_fit` advances exactly one batch.
- `float32` and `float64` are the only supported numerical dtypes.
- Singular systems use a least-squares-style fallback instead of silently
  returning non-finite coefficients.
- `coef_`, `intercept_`, `state_dict()`, and the version 2 `.npz` checkpoint
  remain portable between engines and devices.
- Pandas feature names continue to be validated by the Python layer.
- PyTorch inputs are detached values, not differentiable operations.
- TensorFlow support remains eager-only.

The committed corpus in `tests/golden/native_core_v1.json` is the machine
readable version of this contract. It records inputs, configuration, state
after every batch, diagnostics, and probe predictions. The current NumPy
`float64` implementation is the primary oracle. Cross-engine comparisons use
dtype-specific tolerances rather than bitwise equality.

## Native workspace

The planned implementation layout is:

```text
native/
  Cargo.toml
  crates/
    rh-core/          algorithm types, state invariants, error model
    rh-cpu/           CPU solver, BLAS/LAPACK dispatch, fused loops
    rh-cuda-ffi/      safe Rust wrapper over the C ABI
    rh-python-cpu/    CPU PyO3 module and NumPy adapter
    rh-python-cuda/   CUDA PyO3 module and host NumPy adapter
  cuda/
    include/rh_cuda.h
    src/c_api.cu          extern "C" entry points, guards, error translation
    src/pipeline.cu       one batch transition, host prediction
    src/linear_solver.cu  Cholesky -> LU -> lazy SVD ladder
    src/objective.cu      objective, gradient/Hessian, candidate CUDA Graph
    src/batch.cu          batch validation, staging, intercept append
    src/workspace.cu      device allocation and the engine destructor
    src/engine_internal.cu  memory-pool registry, Failure key function
    src/huber_kernels.cu  elementwise and fused device kernels
    src/abi_contract.cpp  static_asserts pinning the public struct layout
  contracts/
    rh_cuda_contract.json  single source of truth for the CUDA ABI
  python-cpu/         CPU wheel metadata and legal files
  python-cuda/        CUDA wheel metadata and legal files
```

`rh-core` must not depend on Python or CUDA. The Python extension will release
the GIL while an update is executing. C++ implementation details must not cross
the CUDA C ABI.

### P1 CPU implementation amendment

P1 is delivered as an independently installable
`renewable-huber-native-cpu` distribution. Its extension module is
`_renewable_huber_native_cpu`, and users opt in with
`backend="native_cpu"`. Since PR #29, the automatic CPU selector may choose
this engine from bounded, process-local runtime evidence; it falls back to
NumPy conservatively and never persists a machine-specific map.

The P1 dense-solver boundary uses a portable partial-pivot LU implementation
with a minimum-norm SVD fallback. The provider is replaceable without changing
`rh-core`, the PyO3 ABI, or checkpoint layout. A future static BLAS/LAPACK
provider must pass the same golden corpus and avoid oversubscription before it
can become a wheel default.

## Native update boundary

The native layer receives one complete batch and returns one complete state
transition:

```text
update(
    engine,
    previous_state,
    X,
    y,
    optional_sample_weight,
    immutable_config
) -> (next_state, diagnostics)
```

The ABI must use:

- fixed-width integers;
- explicit dtype, device, shape, and stride descriptors;
- opaque engine and allocation handles;
- integer status codes plus an engine-owned error message;
- no Rust or C++ objects in public structs;
- no callback into Python while the solver loop is active.

Panics and C++ exceptions must be caught before crossing their respective FFI
boundaries.

## CPU engine rules

- Use a BLAS/LAPACK provider for GEMV, GEMM, Cholesky, QR, and least squares.
- Benchmark a portable provider against available vendor BLAS before choosing
  wheel defaults.
- Use Rust SIMD/Rayon only for fused elementwise work and reductions.
- Prevent oversubscription between Rayon and the BLAS thread pool.
- Reuse residual, score, curvature, weighted-design, Hessian, and solve
  workspaces.
- Accept C-contiguous NumPy buffers initially. Other layouts may make one
  explicit contiguous copy in the Python adapter.

## CUDA engine rules

One `CudaEngine` is bound to one device and owns:

- a CUDA stream or an explicitly borrowed caller stream;
- cuBLAS/cuBLASLt and cuSOLVER handles;
- coefficients and information matrix;
- residual, score, curvature, weighted-design, Hessian, reduction, and solver
  workspaces;
- a cache keyed by dtype, feature count, and maximum batch size.

The initial update pipeline is:

```text
X beta                         cuBLAS GEMV
residual + score + curvature   fused CUDA kernel
X.T score                      cuBLAS GEMV
row weighting                  fused CUDA kernel
X.T W X                        cuBLAS/cuBLASLt GEMM
Hessian solve                  cuSOLVER Cholesky
line search and convergence    device reductions, minimal host decisions
state update                   device-resident
```

Cholesky is the fast path for a positive-definite Hessian. A failed
factorization must enter a tested QR/least-squares fallback. A custom weighted
Gram kernel is considered only if Nsight Compute shows that row weighting plus
vendor GEMM is a material bottleneck.

### P2 implementation amendment

The P2 compatibility engine uses pivoted cuSOLVER LU for its regular dense
solve, with a minimum-norm SVD fallback for singular systems. Portable
checkpoints may contain a general, slightly asymmetric information matrix, so
mirroring one triangle through Cholesky would change valid state. The current
engine therefore takes Cholesky only after an explicit symmetry check and
retains LU plus a lazily initialized SVD fallback. P2 started with contiguous
host NumPy transport; the P3 extension now consumes same-device contiguous
DLPack tensors. Fully device-side convergence decisions remain future work.

## DLPack and stream ownership

The native tensor adapter accepts single-device, C-contiguous `float32`/`float64`
CuPy, PyTorch CUDA, and TensorFlow eager GPU tensors. CuPy and PyTorch receive
the engine's consumer stream according to the Python DLPack protocol. Because
TensorFlow's legacy exporter has no stream parameter, its adapter explicitly
waits for pending eager work before exporting the zero-copy capsule, and rejects
versions where it cannot establish that synchronization boundary. The consumer
retains each managed tensor until native work using its memory is complete,
claims the capsule once, and invokes its deleter exactly once.

Non-contiguous tensors are rejected with a stable public error; this path never
silently copies, changes dtype, changes device, or stages through host memory.
Cross-device state migration is an explicit operation; it is never an implicit
side effect of `partial_fit`. Device-resident prediction remains a separate
future ABI and is rejected rather than implicitly copied to host.

Integration order is NumPy, CuPy, PyTorch, then TensorFlow eager. Framework
custom operators are deferred until profiling demonstrates that the generic
DLPack adapter is a significant cost.

## Precision modes

The initial engines implement only `precision="strict"` semantics, matching the
existing dtype contract and disabling silent reduced-precision matrix math.

An opt-in `precision="fast"` may be proposed later. It requires:

- an explicit public configuration change;
- separate golden tolerances and benchmark records;
- documented hardware-dependent behavior;
- no change to the strict default.

## Golden corpus policy

The version 1 corpus covers:

- unpenalized, weighted, two-batch streaming with an intercept;
- L1, two-batch streaming and historical subgradient state;
- `float32` without an intercept and with response outliers;
- a rank-deficient design with `ridge=0` to exercise solver fallback.

`scripts/generate_native_golden.py --check` regenerates the corpus in memory
and fails if the committed JSON differs. Updating a golden value requires an
RFC amendment or a documented numerical bug fix.

Native differential tests must compare:

- coefficients and information matrix after every batch;
- sample count, effective weight, batch count, and previous lambda;
- convergence flag, objective, lambda, and bandwidth;
- predictions on fixed probe rows.

Iteration counts are recorded for diagnosis but are not a cross-library
equality requirement when the final state meets the numerical contract.

## Benchmark protocol

`scripts/benchmarks/benchmark_shape_sweep.py` emits schema-versioned JSON.
Schema v2 records cold and steady-state contracts separately: data generation,
dtype conversion, CUDA context creation, and warm-up are excluded from both;
steady-state also excludes the one-time model/engine prime and empty-state
restore. Host-fed and device-resident CUDA paths are reported separately.

Every record includes:

- sample count, feature count, batch size, batch count;
- penalty, dtype, seed, maximum iterations, and tolerance;
- individual timings, median, minimum, throughput, and solver iterations;
- data seed and SHA-256 input fingerprint, lifecycle, operation, and reset
  timing policy;
- Python, OS, CPU, NumPy, CuPy, CUDA runtime, and GPU metadata;
- thread environment plus the NumPy BLAS/LAPACK provider;
- whether input transfer was inside the measured interval.

The required shape classes are:

| Class | Samples | Features | Batch size | Purpose |
|---|---:|---:|---:|---|
| latency | 4,096 | 16 | 4,096 | launch and dispatch overhead |
| reference | 100,000 | 90 | 32,768 | current project workload |
| wide | 16,384 | 256 | 4,096 | weighted Gram and solve pressure |
| streaming | 1,000,000 | 32 | 65,536 | sustained device throughput |

Both penalties and both dtypes are measured where memory and runtime permit.
The `smoke` profile is suitable for local validation; the `standard` profile
is the publishable baseline.

Results from different hardware are not directly merged. The fixed-runner
regression/competitor gate and conservative calibration advisor are specified
in [native performance policy](native-performance-policy.md). Native is not
promoted for a shape unless it is at least as fast as its paired NumPy/CuPy
reference under an identical contract; the advisor uses a stricter 10% margin.

## Nsight baseline

`scripts/profiling/profile_cuda_update.py` places NVTX ranges around warm-up,
complete fits, and individual batches. Wrapper scripts invoke:

- Nsight Systems for launch, synchronization, transfer, and library-call
  timelines;
- Nsight Compute for kernel occupancy, memory traffic, divergence, and selected
  roofline metrics.

The baseline capture must record tool version, driver, CUDA runtime, GPU,
command line, workload shape, and git revision next to the report. Binary
`.nsys-rep` and `.ncu-rep` files are local artifacts and are not committed.
Only the metadata and summarized findings belong in version control.

The first trace must answer:

1. How much time is spent in weighted Gram and dense solve?
2. How many synchronizations occur per solver iteration?
3. What fraction of wall time is host-to-device transfer?
4. Are branch-heavy Huber kernels memory-bound or divergence-bound?
5. Which allocations repeat after warm-up?

## Delivery and rollback

The migration is additive:

1. build the Rust CPU engine behind an internal selector;
2. run it in differential/shadow tests;
3. opt in explicitly for benchmarks;
4. promote it only after correctness and performance gates pass;
5. repeat the process for CUDA;
6. retain the Python reference for at least one release after promotion.

If native loading, capability checks, or initialization fail, `backend="auto"`
may use the documented Python fallback. An explicitly requested native or CUDA
engine must raise a clear error instead of silently changing execution mode.
