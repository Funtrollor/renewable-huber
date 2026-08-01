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
pure Python and `backend="auto"` continues to select NumPy on CPU. An explicit
`backend="native_cpu"` request raises `BackendUnavailableError` if the native
wheel is absent or its ABI/API versions do not match.

The first compatible base release is `renewable-huber` 0.6.0. The native wheel
declares `renewable-huber>=0.6.0,<0.7` so it cannot be paired silently with the
published 0.5.1 API, which predates the `native_cpu` selector.

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

Create or activate a Python 3.10-3.12 virtual environment with Maturin and the
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
[`benchmarks/baselines/p1-windows-ryzen9900x-shape-sweep.json`](../benchmarks/baselines/p1-windows-ryzen9900x-shape-sweep.json).
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
[`p3-windows-ryzen9900x-native-cpu-v2.json`](../benchmarks/baselines/p3-windows-ryzen9900x-native-cpu-v2.json).
It uses identical cold lifecycles for both engines, three warmups, nine
measured samples, 0.25 seconds of fixed block work per sample, NumPy 2.4.6,
and a 24-thread Rayon pool. It covers all four standard shapes, both dtypes,
both penalties, and both public operations (`fit` and `partial_fit`).

The strict competitor and 5% relative-MAD gate passed all 32 Native/NumPy
pairs with `--max-competitor-slowdown 1.0`. NumPy/native median speedup ranged
from 1.17x to 15.65x, with a 1.68x median. The implementation uses size-gated Rayon
row partitions for residuals and gradients, and reduces multiple
matrixmultiply SIMD Gram blocks without nested thread pools. Partial Gram
scratch is capped at 64 MiB; smaller workloads stay on a row-major serial
gradient fast path. Large L1 gradients use contiguous row chunks and private
thread accumulators. The extension also marks its returned result arrays as
detached so the Python adapter does not copy the information matrix twice.
