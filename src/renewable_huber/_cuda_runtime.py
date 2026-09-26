"""Locate the CUDA 12 runtime libraries the native CUDA extension links against.

The extension links the CUDA runtime, cuBLAS and cuSOLVER dynamically, and
cuSOLVER in turn loads cuBLASLt, cuSPARSE and nvJitLink. The
``renewable-huber-native-cuda`` wheel depends on NVIDIA's ``nvidia-*-cu12``
packages, which install exactly those libraries under
``site-packages/nvidia/<component>/{lib,bin}``. Neither the Windows DLL search
nor the Linux dynamic loader looks there, so :func:`prepare` registers them
before the extension is imported.

The pip-installed set is used only when it is complete. Mixing a partial pip
install with a system toolkit would load two CUDA versions into one process,
so an incomplete set falls back entirely to the system: ``CUDA_PATH`` (or the
toolkit that owns ``nvcc``) on Windows, the default loader path on Linux.

This module deliberately imports nothing from the extension so it can be
tested on hosts without CUDA.
"""

from __future__ import annotations

import ctypes
import os
import shutil
from collections.abc import Iterable, Sequence
from importlib.util import find_spec
from pathlib import Path
from typing import Any

#: ``nvidia`` sub-packages that make up the runtime closure, in load order:
#: each library's dependencies precede it.
COMPONENTS = ("nvjitlink", "cuda_runtime", "cublas", "cusparse", "cusolver")

#: Linux shared objects preloaded from the pip packages, in dependency order.
LINUX_LIBRARIES: tuple[tuple[str, str], ...] = (
    ("nvjitlink", "libnvJitLink.so.12"),
    ("cuda_runtime", "libcudart.so.12"),
    ("cublas", "libcublasLt.so.12"),
    ("cublas", "libcublas.so.12"),
    ("cusparse", "libcusparse.so.12"),
    ("cusolver", "libcusolver.so.11"),
)

#: Windows DLLs the pip packages must provide for the set to count as complete.
WINDOWS_LIBRARIES: tuple[tuple[str, str], ...] = (
    ("nvjitlink", "nvJitLink_120_0.dll"),
    ("cuda_runtime", "cudart64_12.dll"),
    ("cublas", "cublasLt64_12.dll"),
    ("cublas", "cublas64_12.dll"),
    ("cusparse", "cusparse64_12.dll"),
    ("cusolver", "cusolver64_11.dll"),
)


def nvidia_roots() -> list[Path]:
    """Return every directory the ``nvidia`` namespace package spans."""

    try:
        spec = find_spec("nvidia")
    except (ImportError, ValueError):
        return []
    if spec is None or spec.submodule_search_locations is None:
        return []
    return [Path(location) for location in spec.submodule_search_locations]


def _find(roots: Iterable[Path], component: str, subdirectory: str, name: str) -> Path | None:
    for root in roots:
        candidate = root / component / subdirectory / name
        if candidate.is_file():
            return candidate
    return None


def pip_libraries(
    roots: Sequence[Path], libraries: Sequence[tuple[str, str]], subdirectory: str
) -> list[Path] | None:
    """Return the complete pip-installed library set in load order, or ``None``."""

    found = [_find(roots, component, subdirectory, name) for component, name in libraries]
    if any(path is None for path in found):
        return None
    return [path for path in found if path is not None]


def _windows_system_bin() -> Path | None:
    candidates = [os.environ.get("CUDA_PATH")]
    candidates.extend(
        value for name, value in os.environ.items() if name.startswith("CUDA_PATH_V") and value
    )
    if nvcc_path := shutil.which("nvcc"):
        candidates.append(str(Path(nvcc_path).resolve().parent.parent))
    for candidate in dict.fromkeys(path for path in candidates if path):
        bin_path = Path(candidate) / "bin"
        if bin_path.is_dir():
            return bin_path
    return None


def prepare(
    *,
    platform: str = os.name,
    roots: Sequence[Path] | None = None,
    add_dll_directory: Any = None,
    load_library: Any = None,
) -> list[Any]:
    """Make the CUDA runtime loadable and return the handles to keep alive.

    Windows receives ``os.add_dll_directory`` registrations; Linux receives
    ``RTLD_GLOBAL`` preloads, so the extension's dependencies resolve by soname
    to the pip-installed copies. The keyword arguments exist for tests.
    """

    roots = nvidia_roots() if roots is None else list(roots)
    handles: list[Any] = []
    if platform == "nt":
        add_dll_directory = add_dll_directory or getattr(os, "add_dll_directory", None)
        if add_dll_directory is None:
            return handles
        libraries = pip_libraries(roots, WINDOWS_LIBRARIES, "bin")
        if libraries is not None:
            for directory in dict.fromkeys(path.parent for path in libraries):
                handles.append(add_dll_directory(str(directory)))
            return handles
        system_bin = _windows_system_bin()
        if system_bin is not None:
            handles.append(add_dll_directory(str(system_bin)))
        return handles

    libraries = pip_libraries(roots, LINUX_LIBRARIES, "lib")
    if libraries is None:
        # A system toolkit resolves through LD_LIBRARY_PATH or ldconfig.
        return handles
    load_library = load_library or (
        lambda path: ctypes.CDLL(str(path), mode=getattr(ctypes, "RTLD_GLOBAL", 0))
    )
    for path in libraries:
        handles.append(load_library(path))
    return handles
