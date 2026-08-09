# renewable-huber-native-cuda

Optional Rust/CUDA 12 engine for
[`renewable-huber`](https://github.com/Funtrollor/renewable-huber). Version
0.6.1 requires exactly `renewable-huber==0.6.1`.

```powershell
python -m pip install renewable-huber-native-cuda==0.6.1
```

Published wheels currently support CPython 3.10–3.12 on Windows x86-64. The
wheel does not bundle NVIDIA libraries. It requires a compatible NVIDIA driver
and a CUDA 12 runtime discoverable through `CUDA_PATH`, including cudart,
cuBLAS/cuBLASLt, cuSOLVER, cuSPARSE and nvJitLink DLLs. Rust, CMake, Visual
Studio and `nvcc` are not required to install the wheel.

```python
from renewable_huber import RenewableHuberRegressor

model = RenewableHuberRegressor(
    backend="native_cuda",
    device="cuda",
    dtype="float32",
    penalty="none",
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

Device-resident prediction is not part of the current ABI: `predict` accepts
host input and returns a NumPy array. The engine currently supports
`penalty="none"` and is always selected explicitly; `backend="auto"` never
chooses native CUDA. See the project
[support matrix](https://github.com/Funtrollor/renewable-huber/blob/main/docs/support-matrix.md)
for CUDA Graph and fast-math
limits.

Source builds require CUDA Toolkit 12.x (12.8+ for the release architecture
set), Visual Studio 2022 C++ Build Tools on Windows, CMake, Ninja, Rust and
Maturin. Developer builds target the active GPU by default and are not portable
release wheels.
