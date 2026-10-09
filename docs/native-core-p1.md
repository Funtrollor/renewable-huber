# Native-core P1: Rust CPU engine

P1 moves one complete renewable Huber batch transition behind a Rust/PyO3
boundary while keeping the public estimator, validation, feature-name handling,
and checkpoint format in Python.

## Delivered architecture

```text
RenewableHuberRegressor
        |
        +-- NumPy validation and portable state
        |
        +-- NativeCpuBackend
                |
                +-- rh-python-cpu (PyO3, exact dtype/contiguity checks)
                        |
                        +-- rh-core (state/config/error contract)
                        |
                        +-- rh-cpu (Newton, LAMM, workspace, dense solver)
```

The native distribution is optional. The base `renewable-huber` wheel remains
pure Python. When this wheel is installed, CPU `backend="auto"` may select it
from bounded host-local runtime evidence; without decisive evidence it remains
on NumPy. An explicit
`backend="native_cpu"` request raises `BackendUnavailableError` if the native
wheel is absent or its ABI/API versions do not match.

The first published compatible base release was `renewable-huber` 0.6.1.
Every native wheel declares an exact dependency on the base release of the
same version (`renewable-huber==X.Y.Z`, `==0.7.0` for 0.7.0), so plugin and
public API/checkpoint contracts cannot drift independently.

## P1 numerical scope

- C-contiguous NumPy `float32` and `float64` input;
- unpenalized renewable Huber Newton updates;
- L1 renewable penalized Huber LAMM updates;
- frequency-style sample weights;
- streaming state and historical `previous_lambda`;
- general row-major information matrices, including asymmetric checkpoints;
- partial-pivot LU with minimum-norm SVD fallback for singular systems;
- engine-owned reusable batch workspaces;
- GIL release around update and prediction.

The Python adapter may make one explicit contiguous conversion. The direct
PyO3 binding rejects non-contiguous or wrong-dtype arrays. The caller must not
mutate a writable NumPy buffer from another thread while a native call is
using it.

## Build locally

Create or activate a Python 3.10-3.13 virtual environment with Maturin and the
base project installed, then run:

```powershell
python -m pip install -e ".[dev]" maturin
.\scripts\native\build_native_cpu.ps1 -Python python
```

The equivalent portable build command is:

```powershell
Push-Location native/python-cpu
python -m maturin build --release
Pop-Location
```

## Correctness gates

```powershell
cargo fmt --manifest-path native/Cargo.toml --all -- --check
cargo check --manifest-path native/Cargo.toml --workspace --all-targets
cargo test --manifest-path native/Cargo.toml --workspace --all-targets
cargo clippy --manifest-path native/Cargo.toml --workspace --all-targets -- -D warnings
python -m unittest tests.test_native_cpu_backend tests.test_native_golden -v
python scripts/generate_native_golden.py --check
```

The native differential test replays every state transition in
`tests/golden/native_core_v1.json`: weighted streaming, L1 historical
subgradients, float32 outliers without an intercept, and rank-deficient
minimum-norm fallback.

## Benchmark

Run a local smoke comparison:

```powershell
python scripts/benchmarks/benchmark_shape_sweep.py `
  --profile smoke --backend cpu --penalty both --dtype both `
  --lifecycle cold --operation partial-fit `
  --output artifacts/p1-cpu-smoke.json
```

Benchmark schema v2 makes lifecycle explicit. The cold command above includes
engine initialization for both NumPy and native CPU; use a separate
`--lifecycle steady --operation partial-fit` run for reusable-engine
throughput. See the [native performance policy](native-performance-policy.md)
for fair comparison and promotion rules.

The checked-in Windows/CPython 3.11 baseline at commit `4ea5ff7` is
[`benchmarks/baselines/p1-windows-ryzen9900x-shape-sweep.json`](https://github.com/Funtrollor/renewable-huber/blob/main/benchmarks/baselines/p1-windows-ryzen9900x-shape-sweep.json).
The table reports `NumPy median / native median`; values above 1 mean the
native engine is faster. It is a schema-v1 historical record, not a hard
schema-v2 dispatch or regression baseline.

| Standard shape | penalty | float32 | float64 |
| --- | --- | ---: | ---: |
| latency (4,096 × 16) | none | 1.69× | 1.56× |
| reference (100,000 × 90) | none | 0.84× | 0.60× |
| wide (16,384 × 256) | none | 25.68× | 19.65× |
| streaming (1,000,000 × 32) | none | 0.97× | 0.79× |

The optimized weighted Gram path makes the wide unpenalized case substantially
faster, while reference/streaming shapes and most L1 cases remain slower than
this host's NumPy build. Those results define optimization targets rather than
being hidden: reducing dense-solver allocation and LAMM overhead is the next
CPU-provider work.

The portable dense solver retains LU/SVD compatibility for asymmetric or
singular checkpoints and uses a scale-checked Cholesky fast path for symmetric
float64 Hessians. The binding moves resident state into each transition
transactionally instead of cloning the `p^2` information matrix, and the L1
loop reuses the accepted candidate residual.

## Optimized schema-v2 baseline

The current fixed-runner record is
[`p6-windows-ryzen9900x-native-cpu-v2.json`](https://github.com/Funtrollor/renewable-huber/blob/main/benchmarks/baselines/p6-windows-ryzen9900x-native-cpu-v2.json),
captured at `be8984a` after the post-0.7.0 engine optimizations below, with
its gate report
[`p6-windows-ryzen9900x-native-cpu-v2-gate.json`](https://github.com/Funtrollor/renewable-huber/blob/main/benchmarks/baselines/p6-windows-ryzen9900x-native-cpu-v2-gate.json).
It uses identical cold lifecycles for both engines, three warmups, nine
measured samples, 0.5 seconds of fixed block work per sample, CPython 3.11.0,
NumPy 2.4.6 (scipy-openblas 0.3.31), and a 24-thread Rayon pool. It covers all
four standard shapes, both dtypes, both penalties, and both public operations
(`fit` and `partial_fit`).

The strict competitor and 5% relative-MAD gate passed all 32 Native/NumPy
pairs with `--max-competitor-slowdown 1.0`. NumPy/native median speedup ranged
from 1.38x to 7.86x, with a 1.81x median; the largest native relative MAD was
3.8%. A first capture with 0.25-second samples had three pairs at 6.6%-9.7%
relative MAD, so the whole sweep was recaptured with 0.5-second samples rather
than more repeats. The host was a desktop in light use during the capture
(browser, chat client, game launcher); the processes using the most CPU were
recorded before and after each run, and together they averaged well under one
of the 24 hardware threads.

The wide unpenalized ratios are lower than in the previous record
(3.06x-7.86x against 6.19x-15.65x) because the NumPy reference got faster, not
because native got slower. With the same NumPy and OpenBLAS, the NumPy wide
`none` medians fell to 0.31x-0.47x of the earlier record's, after the backend
and solver changes since that capture, while native fell to 0.71x-0.79x.
Every other pair kept or widened its native advantage (median 1.15x the
earlier ratio).

The previous record,
[`p3-windows-ryzen9900x-native-cpu-v2.json`](https://github.com/Funtrollor/renewable-huber/blob/main/benchmarks/baselines/p3-windows-ryzen9900x-native-cpu-v2.json)
(`8c631ab`, 0.25-second samples, 1.17x-15.65x, median 1.68x), is kept as a
historical record.

The implementation uses size-gated Rayon
row partitions for residuals and gradients, and reduces multiple
matrixmultiply SIMD Gram blocks without nested thread pools. Partial Gram
scratch is capped at 64 MiB; smaller workloads stay on a row-major serial
gradient fast path. Large L1 gradients use contiguous row chunks and private
thread accumulators. The extension also marks its returned result arrays as
detached so the Python adapter does not copy the information matrix twice.

## Post-0.7.0 engine optimizations

An Amdahl decomposition of the 0.7.0 engine, timed per phase at one, two and
four threads, showed that the code written as serial (dense solve, per-row
Huber loops, `p^2` terms, input validation) was only 1%-11% of an update.
The poor scaling came from elsewhere, and four changes address it. Python-side
batch validation was deliberately left alone.

- **False sharing in the row-chunked gradient.** Each worker accumulated its
  p-wide partial gradient in place inside one shared `Vec`. Because
  `p * size_of::<T>()` is rarely a multiple of 64 bytes, neighbouring workers
  wrote the cache line they shared on every row, and this phase ran at
  0.7x-1.2x on four threads. A worker-local accumulator, copied out once,
  brings it to 2.2x-3.0x. The per-element summation order is unchanged.
- **Redundant Newton residual.** `gradient_and_hessian` recomputed
  `y - X @ beta`, although the preceding objective evaluation (the initial
  one, or the accepted line-search trial) had just left exactly that
  residual in the workspace. That was about one of every 3.5 full passes over
  `X` per iteration. L1 already reused it.
- **Gram memory traffic.** The weighted Gram matrix was two full passes:
  scale every row into an `n * p` workspace, then read `X` and that
  workspace back for GEMM. It is now built `GRAM_ROW_CHUNK = 256` rows at a
  time: weight a block into cache-resident scratch, then accumulate its
  product straight into the output. The weighting plus GEMM measured
  1.2x-2.0x faster on one thread, and the `n * p` workspace is gone.
  matrixmultiply already splits the row dimension into `KC = 256` blocks and
  accumulates them in order, so a chunk of exactly that size reproduces a
  whole-range GEMM bit for bit.
  `tests.rs::chunked_gram_is_bitwise_identical_to_one_gemm` enforces it.
- **Vectorized `dot`.** A single running sum is a chain of dependent
  floating-point adds that the compiler may not reorder, so it could not
  vectorize the residual and prediction row kernels. Eight independent
  lanes, combined pairwise, make those kernels 1.3x-3.3x (f64) and 2x-4.9x
  (f32) faster on a single thread. Sixteen lanes and an AVX build measured
  no better.

The first three are bitwise identical to 0.7.0, which was checked by hashing
the coefficients and information matrices of three batches of every standard
shape, dtype and penalty at one, two and four threads. The `dot` change fixes
a different summation order. It is still deterministic and independent of
the host and of the thread count, but native CPU results now differ from 0.7.0
in the last bits. Some float32 L1 streams take one or two more or fewer
iterations near `tol`. The golden corpora and their tolerances are unchanged.

**Rejected: a triangular Gram product.** The Gram matrix is symmetric, so
multiplying each 16-128-row block only against the columns from its diagonal
onwards halves the arithmetic. With matrixmultiply this measured 0.43x-1.24x
of a single full GEMM, usually slower. Each call repacks a long panel of the
weighted design, and at these shapes the product is limited by reading the
two `n * p` operands, not by multiply-adds. The full GEMM was kept.

An interleaved A/B on a 4-vCPU x86-64 cloud VM compared the 0.7.0 engine with
the optimized one. It called `NativeCpuEngine.update` directly over the
standard-profile stream for each shape, from an empty state, taking the
median of seven alternating rounds. This is not the fixed Ryzen runner, and
the VM has the usual cloud noise of roughly ±10%. Every case was faster:

| Shape | 1 thread | 4 threads |
| --- | --- | --- |
| latency (4,096 × 16) | 1.08x-1.18x | 1.23x-2.06x |
| reference (100,000 × 90) | 1.30x-2.01x | 1.40x-2.11x |
| wide (16,384 × 256) | 1.20x-2.43x | 1.09x-1.85x |
| streaming (1,000,000 × 32) | 1.32x-1.69x | 1.51x-1.58x |

Each range covers both dtypes and both penalties. The median was 1.41x on one
thread and 1.54x on four.

The fixed Ryzen 9 9900X runner then repeated the comparison through the public
estimator with `run_interleaved_benchmark.py --freeze-sample-repetitions`:
`v0.7.0` (`53b5f20`) against `be8984a`, standard profile, `--backend cpu`,
both penalties and dtypes, `partial_fit`, nine aligned rounds, three warmups,
the default 24-thread pool, CPython 3.11.0 and NumPy 2.4.6 on both sides. Both
records report native ABI 1 / API 2, so no native-version option was used.
The records are
[`p6-windows-ryzen9900x-native-cpu-ab-cold-gate.json`](https://github.com/Funtrollor/renewable-huber/blob/main/benchmarks/baselines/p6-windows-ryzen9900x-native-cpu-ab-cold-gate.json)
and
[`p6-windows-ryzen9900x-native-cpu-ab-steady-gate.json`](https://github.com/Funtrollor/renewable-huber/blob/main/benchmarks/baselines/p6-windows-ryzen9900x-native-cpu-ab-steady-gate.json),
each with its merged baseline and candidate records and frozen sample plan.
Speedup below is `baseline / candidate`, the inverse of the gate's paired
median:

| Shape | cold | steady | 4-vCPU VM, engine only, 4 threads |
| --- | --- | --- | --- |
| latency (4,096 × 16) | 1.25x-1.30x | 1.18x-1.28x | 1.23x-2.06x |
| reference (100,000 × 90) | 1.04x-1.27x | 1.03x-1.25x | 1.40x-2.11x |
| wide (16,384 × 256) | 1.14x-1.64x | 1.20x-1.66x | 1.09x-1.85x |
| streaming (1,000,000 × 32) | 0.99x-1.09x | 1.02x-1.11x | 1.51x-1.58x |

The medians were 1.21x (cold) and 1.18x (steady). Within the ±10% noise band
of this host, ten cold and twelve steady cases are faster; the rest show no
measurable difference, and no case is slower. The cases without a measurable
difference are reference float64 (both penalties cold, unpenalized steady) and
every streaming case except steady float32 unpenalized.

The VM column measures something narrower and is not directly comparable: it
called `NativeCpuEngine.update` on prebuilt design matrices, while the fixed
runner timed the public estimator. The estimator also runs the Python-side
batch preparation (`np.isfinite` over the batch, and the `column_stack` copy
that appends the intercept), which is single-threaded and which these
optimizations deliberately left alone. On the VM that preparation was already
about a fifth of a streaming `partial_fit` at four threads. With 24 threads
the engine's share of an update shrinks further, so the unchanged
preparation takes a larger share, and the end-to-end gain on the long narrow
reference and streaming batches is diluted the most. Native stayed 1.4x-8.2x faster than NumPy in every case of both
captures.

Neither gate passed. In both lifecycles, the three float32 L1 cases with
several batches failed `median solver iterations differ`: reference 26 -> 28,
streaming 56 -> 51, wide 48 -> 45 (summed over the stream's batches). That is
the expected effect of the new `dot` summation order near `tol`, not a
slowdown, and the `--max-iteration-delta 1` limit was not changed. Per
iteration, the candidate took 0.73x-0.75x (reference), 0.995x-1.015x (streaming) and
0.62x-0.66x (wide) of the baseline's time. In the steady capture, wide float32
L1 also exceeded the 5% MAD limit on the candidate side (5.02%, one 26 ms
sample among 19-22 ms ones). Every other case passed every check, including
competitor parity against NumPy.
