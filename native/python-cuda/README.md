# renewable-huber-native-cuda

Optional Rust/CUDA 12 engine for
[`renewable-huber`](https://github.com/Funtrollor/renewable-huber). Version
0.7.0 requires exactly `renewable-huber==0.7.0`.

```powershell
python -m pip install renewable-huber-native-cuda==0.7.0
```

Release wheels cover CPython 3.10–3.13 on Windows x86-64 and Linux x86-64
(`manylinux_2_28`). The wheel does not bundle NVIDIA libraries. Instead it depends on
NVIDIA's own `nvidia-cuda-runtime-cu12`, `nvidia-cublas-cu12`,
`nvidia-cusolver-cu12`, `nvidia-cusparse-cu12` and `nvidia-nvjitlink-cu12`
wheels, and loads that complete set at import. When the set is incomplete it
falls back to a system CUDA 12 toolkit (`CUDA_PATH` on Windows, the default
loader path on Linux). A compatible NVIDIA driver is always required. Rust,
CMake, Visual Studio and `nvcc` are not required to install the wheel.

```python
from renewable_huber import RenewableHuberRegressor

model = RenewableHuberRegressor(
    backend="native_cuda",
    device="cuda",
    dtype="float32",
    penalty="l1",  # or "none"
)
model.fit(X, y)
```

The whole-batch engine accepts contiguous host NumPy arrays. Its update path
also accepts C-contiguous CuPy, PyTorch-CUDA and TensorFlow-eager GPU tensors
through DLPack in exact float32 or float64. All inputs must already be on the
selected engine device; dtype casts, contiguous copies, cross-device copies
and host staging are never implicit. CuPy/PyTorch negotiate the consumer stream
directly; TensorFlow uses an explicit producer synchronization boundary before
zero-copy export.

`predict` accepts host arrays and, since Python API 4, the same CUDA DLPack
tensors, which it reads in place on the engine stream; the result is always a
NumPy array. The engine implements `penalty="none"` and `penalty="l1"` (C ABI
2) and is always selected explicitly; `backend="auto"` never chooses native
CUDA. See the project
[support matrix](https://github.com/Funtrollor/renewable-huber/blob/main/docs/support-matrix.md)
for CUDA Graph and fast-math
limits.

Source builds require CUDA Toolkit 12.x (12.8+ for the release architecture
set), Visual Studio 2022 C++ Build Tools on Windows, CMake, Ninja, Rust and
Maturin. Linux release wheels are built inside `manylinux_2_28` by
`scripts/native/build_linux_cuda_wheel.sh`. Developer builds target the active GPU by default and are not portable
release wheels.
