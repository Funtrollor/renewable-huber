"""The CUDA runtime search that runs before the native CUDA extension loads.

Every test drives :mod:`renewable_huber._cuda_runtime` against a fake
``site-packages/nvidia`` tree and recording callbacks, so none needs CUDA, a
GPU or the extension. That is why the module lives in ``core``: the loader
decides which CUDA libraries a user's process gets, and a regression here is
invisible until someone installs the wheel on a machine with a GPU.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from renewable_huber import _cuda_runtime as runtime


def _fake_nvidia_tree(root: Path, libraries: tuple[tuple[str, str], ...], subdir: str) -> Path:
    nvidia = root / "nvidia"
    for component, name in libraries:
        directory = nvidia / component / subdir
        directory.mkdir(parents=True, exist_ok=True)
        (directory / name).write_bytes(b"")
    return nvidia


class LinuxPreloadTests(unittest.TestCase):
    def test_complete_pip_set_is_preloaded_in_dependency_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            nvidia = _fake_nvidia_tree(Path(directory), runtime.LINUX_LIBRARIES, "lib")
            loaded: list[Path] = []
            handles = runtime.prepare(
                platform="posix", roots=[nvidia], load_library=lambda path: loaded.append(path)
            )
        self.assertEqual([path.name for path in loaded], [n for _, n in runtime.LINUX_LIBRARIES])
        self.assertEqual(len(handles), len(runtime.LINUX_LIBRARIES))
        # cuSOLVER needs cuBLAS, cuSPARSE and nvJitLink; each must already be
        # resident when it loads, because its own RUNPATH cannot reach them.
        names = [path.name for path in loaded]
        self.assertLess(names.index("libnvJitLink.so.12"), names.index("libcusparse.so.12"))
        self.assertLess(names.index("libcusparse.so.12"), names.index("libcusolver.so.11"))
        self.assertLess(names.index("libcublasLt.so.12"), names.index("libcublas.so.12"))
        self.assertLess(names.index("libcublas.so.12"), names.index("libcusolver.so.11"))

    def test_incomplete_pip_set_defers_entirely_to_the_system_loader(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            partial = runtime.LINUX_LIBRARIES[:-1]
            nvidia = _fake_nvidia_tree(Path(directory), partial, "lib")
            loaded: list[Path] = []
            handles = runtime.prepare(
                platform="posix", roots=[nvidia], load_library=lambda path: loaded.append(path)
            )
        # Loading five pip libraries next to a system cuSOLVER would put two
        # CUDA builds in one process.
        self.assertEqual(loaded, [])
        self.assertEqual(handles, [])

    def test_namespace_package_may_span_several_site_directories(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            half = len(runtime.LINUX_LIBRARIES) // 2
            left = _fake_nvidia_tree(Path(first), runtime.LINUX_LIBRARIES[:half], "lib")
            right = _fake_nvidia_tree(Path(second), runtime.LINUX_LIBRARIES[half:], "lib")
            loaded: list[Path] = []
            runtime.prepare(
                platform="posix", roots=[left, right], load_library=lambda p: loaded.append(p)
            )
        self.assertEqual(len(loaded), len(runtime.LINUX_LIBRARIES))

    def test_missing_nvidia_package_is_not_an_error(self) -> None:
        with mock.patch.object(runtime, "find_spec", return_value=None):
            self.assertEqual(runtime.nvidia_roots(), [])
            self.assertEqual(runtime.prepare(platform="posix", load_library=self.fail), [])


class WindowsDllDirectoryTests(unittest.TestCase):
    def test_complete_pip_set_registers_each_bin_directory_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            nvidia = _fake_nvidia_tree(Path(directory), runtime.WINDOWS_LIBRARIES, "bin")
            registered: list[str] = []
            with mock.patch.dict("os.environ", {"CUDA_PATH": directory}, clear=True):
                handles = runtime.prepare(
                    platform="nt", roots=[nvidia], add_dll_directory=registered.append
                )
        # cuBLAS and cuBLASLt share a directory; it is registered once, and the
        # system toolkit is not added next to a complete pip set.
        self.assertEqual(len(registered), len(runtime.COMPONENTS))
        self.assertEqual(len(set(registered)), len(registered))
        self.assertTrue(all("nvidia" in path for path in registered))
        self.assertEqual(len(handles), len(registered))

    def test_incomplete_pip_set_falls_back_to_cuda_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            nvidia = _fake_nvidia_tree(Path(directory), runtime.WINDOWS_LIBRARIES[:2], "bin")
            toolkit = Path(directory) / "toolkit"
            (toolkit / "bin").mkdir(parents=True)
            registered: list[str] = []
            with (
                mock.patch.dict("os.environ", {"CUDA_PATH": str(toolkit)}, clear=True),
                mock.patch.object(runtime.shutil, "which", return_value=None),
            ):
                runtime.prepare(platform="nt", roots=[nvidia], add_dll_directory=registered.append)
        self.assertEqual(registered, [str(toolkit / "bin")])

    def test_no_runtime_anywhere_registers_nothing(self) -> None:
        registered: list[str] = []
        with (
            mock.patch.dict("os.environ", {}, clear=True),
            mock.patch.object(runtime.shutil, "which", return_value=None),
        ):
            runtime.prepare(platform="nt", roots=[], add_dll_directory=registered.append)
        self.assertEqual(registered, [])


class DeclaredRuntimeMatchesPackagingTests(unittest.TestCase):
    """The loader's component list and the wheel's dependencies must agree."""

    def test_every_component_is_a_declared_wheel_dependency(self) -> None:
        from scripts.native.validate_release_artifacts import NATIVE_RUNTIME_DEPENDENCIES

        declared = {
            requirement.split(">=", 1)[0] for requirement in NATIVE_RUNTIME_DEPENDENCIES["cuda"]
        }
        expected = {
            f"nvidia-{component.replace('_', '-')}-cu12" for component in runtime.COMPONENTS
        }
        self.assertEqual(declared, expected)
        self.assertEqual(
            {component for component, _ in runtime.LINUX_LIBRARIES}, set(runtime.COMPONENTS)
        )
        self.assertEqual(
            {component for component, _ in runtime.WINDOWS_LIBRARIES}, set(runtime.COMPONENTS)
        )


if __name__ == "__main__":
    unittest.main()
