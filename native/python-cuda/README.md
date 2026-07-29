# renewable-huber-native-cuda

Optional Rust/CUDA engine for `renewable-huber` 0.6.x.

The P2 engine accepts contiguous host NumPy `float32` and `float64` arrays and
supports `penalty="none"`. It is selected explicitly with
`backend="native_cuda"` and is never selected by `backend="auto"`.

Building from source requires a compatible NVIDIA driver, CUDA Toolkit 12.x,
Visual Studio 2022 C++ Build Tools on Windows, CMake, Ninja, Rust, and Maturin.
Developer builds target the active GPU by default and are not portable release
wheels. Published wheels require an explicit CUDA runtime and SM architecture
policy.
