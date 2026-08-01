# renewable-huber-native-cuda

Optional Rust/CUDA engine for `renewable-huber` 0.6.x.

The engine accepts contiguous host NumPy arrays plus C-contiguous CuPy,
PyTorch-CUDA, and TensorFlow-eager GPU tensors through DLPack in strict
`float32` or `float64`. DLPack inputs must already be on the selected engine
device; dtype casts, contiguous copies, cross-device copies, and host staging
are never implicit. CuPy/PyTorch negotiate the consumer stream directly;
TensorFlow's legacy exporter uses an explicit producer synchronization boundary
before zero-copy export. Device-resident prediction is not part of the current
ABI and is rejected instead of copied to host. The engine supports
`penalty="none"`, is selected explicitly with `backend="native_cuda"`, and is
never selected by `backend="auto"`.

Building from source requires a compatible NVIDIA driver, CUDA Toolkit 12.x,
Visual Studio 2022 C++ Build Tools on Windows, CMake, Ninja, Rust, and Maturin.
Developer builds target the active GPU by default and are not portable release
wheels. Published wheels require an explicit CUDA runtime and SM architecture
policy.
