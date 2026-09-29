# Backends

`RenewableHuberRegressor(backend=..., device=..., dtype=...)` selects where the
update runs. Every backend shares one numerical contract; the
[support matrix](../support-matrix.md) lists exactly what each one supports.

| `backend` | Runs on | Needs |
| --- | --- | --- |
| `"auto"` (default) | NumPy on the CPU, `native_cpu` when measured faster, CuPy for `device="cuda"` | nothing extra |
| `"numpy"` | CPU | nothing extra |
| `"native_cpu"` | CPU, Rust/Rayon | `renewable-huber-native-cpu` |
| `"cupy"` | CUDA | the `gpu-cupy` extra |
| `"native_cuda"` | CUDA, Rust/CUDA whole-batch engine | `renewable-huber-native-cuda` |
| `"torch"` | CPU or CUDA | the `gpu-torch` extra |
| `"tensorflow"` | CPU or CUDA, eager execution only | the `gpu-tensorflow` extra |

The examples on this page assume
`from renewable_huber import RenewableHuberRegressor` and the placeholder
batches `X_batch`, `y_batch`, `X_train`, `y_train` and `X_test` from the
[Quick start](quickstart.md).

## Automatic selection

--8<-- "README.en.md:auto-dispatch"

--8<-- "README.en.md:auto-frameworks"

For details and cost bounds, see the
[CPU auto-dispatch RFC](../cpu-auto-dispatch-rfc.md).

## Rust CPU engine

Install `renewable-huber-native-cpu` as described in
[Installation](installation.md#optional-rust-cpu-core), then:

--8<-- "README.en.md:native-cpu-usage"

It supports `penalty="none"` and `penalty="l1"`, with C-contiguous NumPy
`float32`/`float64` input. `backend="auto"` on the CPU may choose this engine,
but only when the batch is large enough and the local measurement supports it;
to always use it, set `backend="native_cpu"` explicitly. For build,
correctness and benchmark details see
[Native-core P1](../native-core-p1.md); for the dispatch rules see the
[CPU auto-dispatch RFC](../cpu-auto-dispatch-rfc.md).

## CuPy

--8<-- "README.en.md:cupy"

## Native CUDA engine

Install `renewable-huber-native-cuda` as described in
[Installation](installation.md#optional-native-cuda-engine).

--8<-- "README.en.md:native-cuda-usage"

The CUDA performance paths, and when native CUDA is faster than CuPy, are
described in [CUDA performance paths](../gpu-performance.md).

## PyTorch and TensorFlow

--8<-- "README.en.md:torch-tensorflow"
