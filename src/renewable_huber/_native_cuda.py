"""Import bridge for the separately built native CUDA extension."""

import os
import shutil
from pathlib import Path
from typing import Any


def _configure_windows_cuda_dll_path() -> Any | None:
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return None
    candidates = [os.environ.get("CUDA_PATH")]
    candidates.extend(
        value for name, value in os.environ.items() if name.startswith("CUDA_PATH_V") and value
    )
    if nvcc_path := shutil.which("nvcc"):
        candidates.append(str(Path(nvcc_path).resolve().parent.parent))
    for candidate in dict.fromkeys(path for path in candidates if path):
        bin_path = Path(candidate) / "bin"
        if bin_path.is_dir():
            return os.add_dll_directory(str(bin_path))
    return None


_CUDA_DLL_DIRECTORY = _configure_windows_cuda_dll_path()

from _renewable_huber_native_cuda import (  # type: ignore[import-not-found]  # noqa: E402
    NativeCudaEngine,
    device_count,
    is_available,
    version,
)

__all__ = ["NativeCudaEngine", "device_count", "is_available", "version"]
