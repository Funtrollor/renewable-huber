"""Import bridge for the separately built native CUDA extension."""

from ._cuda_runtime import prepare as _prepare_cuda_runtime

# Registered DLL directories and preloaded shared objects must stay alive for
# as long as the extension is loaded; see _cuda_runtime for the search policy.
_CUDA_RUNTIME_HANDLES = _prepare_cuda_runtime()

from _renewable_huber_native_cuda import (  # noqa: E402
    NativeCudaEngine,
    device_count,
    is_available,
    version,
)

__all__ = ["NativeCudaEngine", "device_count", "is_available", "version"]
