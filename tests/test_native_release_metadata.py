from __future__ import annotations

import re
import tarfile
import tempfile
import unittest
import zipfile
from io import BytesIO
from pathlib import Path
from unittest import mock

from scripts.native.smoke_test_cuda_wheels import _smoke_program
from scripts.native.validate_release_artifacts import (
    PROJECT_ROOT,
    SUPPORTED_PYTHON,
    _check_native_wheel,
    base_version,
    check_source_metadata,
    check_wheel_set,
    native_workspace_version,
    read_wheel_metadata,
)


class NativeReleaseMetadataTests(unittest.TestCase):
    def test_workflow_actions_are_immutable_pins(self) -> None:
        workflows = Path(__file__).parents[1] / ".github" / "workflows"
        found = 0
        for path in sorted(workflows.glob("*.yml")):
            workflow = path.read_text(encoding="utf-8")
            uses = re.findall(
                r"uses:\s+([^@\s]+)@([^\s#]+)(?:\s+#\s+([^\n]+))?",
                workflow,
            )
            raw_uses = re.findall(r"^\s*-?\s*uses:\s+(\S+)", workflow, flags=re.MULTILINE)
            self.assertEqual(
                len(uses),
                len(raw_uses),
                f"every action use in {path.name} must match the immutable-pin contract",
            )
            found += len(uses)
            for action, revision, declared_ref in uses:
                with self.subTest(workflow=path.name, action=action):
                    self.assertRegex(revision, r"\A[0-9a-f]{40}\Z")
                    self.assertTrue(declared_ref.strip(), "the human-readable ref is required")
                    if action in {"actions/checkout", "actions/setup-python"}:
                        self.assertRegex(declared_ref, r"\Av\d+\Z")
                        self.assertGreaterEqual(
                            int(declared_ref[1:]),
                            6 if action == "actions/setup-python" else 5,
                            "the action must use its Node.js 24 generation",
                        )
                    artifact_minimums = {
                        "actions/upload-artifact": 7,
                        "actions/download-artifact": 8,
                    }
                    if action in artifact_minimums:
                        self.assertRegex(declared_ref, r"\Av\d+(?:\.\d+){0,2}\Z")
                        self.assertGreaterEqual(
                            int(declared_ref.split(".", 1)[0][1:]),
                            artifact_minimums[action],
                            "the artifact action must use its Node.js 24 generation",
                        )
        self.assertGreater(found, 0, "no workflow actions were inspected")

    def test_release_workflow_has_single_manylinux_policy_source(self) -> None:
        workflow = (Path(__file__).parents[1] / ".github" / "workflows" / "release.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn('manylinux: "2014"', workflow)
        self.assertNotIn("--manylinux 2014", workflow)
        self.assertIn(
            "cargo test --locked -p rh-core -p rh-cpu -p rh-cuda-ffi --all-targets",
            workflow,
        )
        self.assertNotIn("cargo test --locked --workspace --all-targets", workflow)
        self.assertIn("runs-on: windows-2022", workflow)
        self.assertNotIn("runs-on: windows-latest\n    timeout-minutes: 60", workflow)
        self.assertIn("macos-15-intel", workflow)
        self.assertNotIn("macos-13", workflow)
        cuda_toolkit_pins = re.findall(
            r"uses:\s+Jimver/cuda-toolkit@([0-9a-f]{40})\s+#\s+v(\d+\.\d+\.\d+)",
            workflow,
        )
        self.assertEqual(
            len(cuda_toolkit_pins),
            1,
            "the CUDA toolkit action must appear once, pinned to a commit with a version comment",
        )
        self.assertIn(
            "sub-packages: "
            '\'["nvcc", "cuobjdump", "cudart", "cublas", "cublas_dev", '
            '"cusolver", "cusolver_dev", "cusparse", "cusparse_dev", "nvjitlink", '
            '"visual_studio_integration"]\'',
            workflow,
        )
        self.assertIn(
            'RH_CUDA_ARCHITECTURES: "75-real;80-real;86-real;89-real;90-real;120"', workflow
        )
        self.assertIn("cuobjdump --list-elf", workflow)
        self.assertIn("cuobjdump --list-ptx", workflow)
        self.assertIn("--native-dir dist-native-cuda --import-only", workflow)
        for required_file in (
            r"include\cuda_runtime.h",
            r"include\cublas_v2.h",
            r"include\cublasLt.h",
            r"include\cusolverDn.h",
            r"include\cusparse.h",
            r"lib\x64\cudart.lib",
            r"lib\x64\cublas.lib",
            r"lib\x64\cublaslt.lib",
            r"lib\x64\cusolver.lib",
            r"lib\x64\cusparse.lib",
            r"bin\cudart64_12.dll",
            r"bin\cublas64_12.dll",
            r"bin\cublasLt64_12.dll",
            r"bin\cusolver64_11.dll",
            r"bin\cusparse64_12.dll",
            r"bin\nvJitLink_120_0.dll",
            r"bin\cuobjdump.exe",
        ):
            with self.subTest(required_file=required_file):
                self.assertIn(required_file, workflow)
        self.assertNotIn("runs-on: [self-hosted, windows, x64, gpu, cuda12]", workflow)

    def test_cuda_import_only_smoke_preserves_the_full_device_gate(self) -> None:
        import_only = _smoke_program("0.6.1", import_only=True)
        full = _smoke_program("0.6.1", import_only=False)
        self.assertIn("_native_cuda.version()", import_only)
        self.assertIn("isinstance(_native_cuda.is_available(), bool)", import_only)
        self.assertNotIn("backend='native_cuda'", import_only)
        self.assertIn("assert _native_cuda.is_available()", full)
        self.assertIn("backend='native_cuda'", full)

    def test_cuda_device_code_stays_whole_program_so_wheels_keep_ptx(self) -> None:
        # The release workflow proves the shipped wheel carries SM 120 PTX, but
        # only on a Windows runner with a CUDA toolkit. This reads the one
        # setting that decides it, so CPU CI rejects the regression in advance.
        cmake = (Path(__file__).parents[1] / "native" / "cuda" / "CMakeLists.txt").read_text(
            encoding="utf-8"
        )
        self.assertIn("CUDA_SEPARABLE_COMPILATION OFF", cmake)
        self.assertNotIn("CUDA_SEPARABLE_COMPILATION ON", cmake)
        # This property does nothing without separable compilation, so its
        # return would mean the nvlink device link is back and the PTX is gone.
        self.assertNotIn("CUDA_RESOLVE_DEVICE_SYMBOLS", cmake)

    def test_source_projects_match_base_release_exactly(self) -> None:
        self.assertEqual(check_source_metadata(), base_version())
        self.assertEqual(native_workspace_version(), base_version())

    def test_source_projects_reject_python_range_drift(self) -> None:
        version = base_version()
        expected_dependency = f"renewable-huber=={version}"
        projects = {
            PROJECT_ROOT / "pyproject.toml": {"requires-python": SUPPORTED_PYTHON},
            PROJECT_ROOT / "native" / "python-cpu" / "pyproject.toml": {
                "name": "renewable-huber-native-cpu",
                "version": version,
                "dependencies": [expected_dependency],
                "requires-python": SUPPORTED_PYTHON,
            },
            PROJECT_ROOT / "native" / "python-cuda" / "pyproject.toml": {
                "name": "renewable-huber-native-cuda",
                "version": version,
                "dependencies": [expected_dependency],
                "requires-python": SUPPORTED_PYTHON,
            },
        }
        for changed_path in projects:
            with self.subTest(path=changed_path):
                drifted = {path: dict(project) for path, project in projects.items()}
                drifted[changed_path]["requires-python"] = ">=3.10"
                with mock.patch(
                    "scripts.native.validate_release_artifacts._project_metadata",
                    side_effect=lambda path: drifted[path],
                ):
                    with self.assertRaisesRegex(RuntimeError, "requires-python must be"):
                        check_source_metadata()

    def test_release_workflow_supports_non_publishing_full_build(self) -> None:
        workflow = (Path(__file__).parents[1] / ".github" / "workflows" / "release.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("workflow_dispatch:", workflow)
        self.assertIn("github.event_name == 'push' && github.ref_type == 'tag'", workflow)
        self.assertIn('test "$GITHUB_SHA" = "$(git rev-parse origin/main)"', workflow)

    def test_valid_native_wheel_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory) / "renewable_huber_native_cpu-0.6.0-cp312-cp312-win_amd64.whl"
            metadata = (
                "Metadata-Version: 2.4\n"
                "Name: renewable-huber-native-cpu\n"
                "Version: 0.6.0\n"
                "Requires-Python: >=3.10, <3.13\n"
                "Requires-Dist: renewable-huber==0.6.0\n\n"
            )
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("renewable_huber_native_cpu-0.6.0.dist-info/METADATA", metadata)
                archive.writestr("_renewable_huber_native_cpu.pyd", b"extension")
                archive.writestr(
                    "renewable_huber_native_cpu-0.6.0.dist-info/licenses/LICENSE", "license"
                )
                archive.writestr(
                    "renewable_huber_native_cpu-0.6.0.dist-info/licenses/NOTICE", "notice"
                )
            errors = _check_native_wheel(
                read_wheel_metadata(wheel),
                kind="cpu",
                expected_version="0.6.0",
            )
            self.assertEqual(errors, [])

    def test_mismatched_native_version_and_dependency_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory) / "renewable_huber_native_cuda-0.5.0-cp312-win_amd64.whl"
            metadata = (
                "Metadata-Version: 2.4\n"
                "Name: renewable-huber-native-cuda\n"
                "Version: 0.5.0\n"
                "Requires-Python: >=3.9\n"
                "Requires-Dist: renewable-huber>=0.5\n\n"
            )
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("renewable_huber_native_cuda-0.5.0.dist-info/METADATA", metadata)
            errors = _check_native_wheel(
                read_wheel_metadata(wheel),
                kind="cuda",
                expected_version="0.6.0",
                expected_requires_python=SUPPORTED_PYTHON,
            )
            self.assertTrue(any("Version" in error for error in errors))
            self.assertTrue(any("Requires-Python" in error for error in errors))
            self.assertTrue(any("exact base dependency" in error for error in errors))
            self.assertTrue(any("compiled extension" in error for error in errors))

    def test_complete_set_requires_base_sdist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            version = base_version()
            base_dir = root / "base"
            cpu_dir = root / "cpu"
            cuda_dir = root / "cuda"
            base_dir.mkdir()
            cpu_dir.mkdir()
            cuda_dir.mkdir()
            wheel = base_dir / f"renewable_huber-{version}-py3-none-any.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr(
                    f"renewable_huber-{version}.dist-info/METADATA",
                    f"Metadata-Version: 2.4\nName: renewable-huber\nVersion: {version}\n"
                    f"Requires-Python: {SUPPORTED_PYTHON}\n\n",
                )

            with self.assertRaisesRegex(RuntimeError, "expected one base sdist"):
                check_wheel_set(
                    base_dir=base_dir,
                    cpu_dir=cpu_dir,
                    cuda_dir=cuda_dir,
                    expected_cpu=0,
                    expected_cuda=0,
                )

            sdist = base_dir / f"renewable_huber-{version}.tar.gz"
            with tarfile.open(sdist, "w:gz") as archive:
                files = {
                    f"renewable_huber-{version}/PKG-INFO": (
                        f"Metadata-Version: 2.4\nName: renewable-huber\nVersion: {version}\n"
                        f"Requires-Python: {SUPPORTED_PYTHON}\n\n"
                    ).encode(),
                    f"renewable_huber-{version}/LICENSE": b"license",
                    f"renewable_huber-{version}/NOTICE": b"notice",
                }
                for name, contents in files.items():
                    info = tarfile.TarInfo(name)
                    info.size = len(contents)
                    archive.addfile(info, BytesIO(contents))

            result = check_wheel_set(
                base_dir=base_dir,
                cpu_dir=cpu_dir,
                cuda_dir=cuda_dir,
                expected_cpu=0,
                expected_cuda=0,
            )
            self.assertEqual(result["base_sdists"], 1)

    def test_base_artifacts_reject_python_range_drift(self) -> None:
        version = check_source_metadata()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_dir = root / "base"
            cpu_dir = root / "cpu"
            cuda_dir = root / "cuda"
            base_dir.mkdir()
            cpu_dir.mkdir()
            cuda_dir.mkdir()

            wheel = base_dir / f"renewable_huber-{version}-py3-none-any.whl"
            metadata = (
                f"Metadata-Version: 2.4\n"
                f"Name: renewable-huber\n"
                f"Version: {version}\n"
                "Requires-Python: >=3.10\n\n"
            )
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr(f"renewable_huber-{version}.dist-info/METADATA", metadata)

            sdist = base_dir / f"renewable_huber-{version}.tar.gz"
            with tarfile.open(sdist, "w:gz") as archive:
                files = {
                    f"renewable_huber-{version}/PKG-INFO": (
                        f"Metadata-Version: 2.4\n"
                        f"Name: renewable-huber\n"
                        f"Version: {version}\n"
                        "Requires-Python: >=3.10\n\n"
                    ).encode(),
                    f"renewable_huber-{version}/LICENSE": b"license",
                    f"renewable_huber-{version}/NOTICE": b"notice",
                }
                for name, contents in files.items():
                    info = tarfile.TarInfo(name)
                    info.size = len(contents)
                    archive.addfile(info, BytesIO(contents))

            with self.assertRaisesRegex(RuntimeError, "Requires-Python"):
                check_wheel_set(
                    base_dir=base_dir,
                    cpu_dir=cpu_dir,
                    cuda_dir=cuda_dir,
                    expected_cpu=0,
                    expected_cuda=0,
                )

    def test_native_wheels_requires_python_must_match_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory) / "renewable_huber_native_cpu-0.6.0-cp312-cp312-win_amd64.whl"
            metadata = (
                "Metadata-Version: 2.4\n"
                "Name: renewable-huber-native-cpu\n"
                "Version: 0.6.0\n"
                "Requires-Python: >=3.9\n"
                "Requires-Dist: renewable-huber==0.6.0\n\n"
            )
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("renewable_huber_native_cpu-0.6.0.dist-info/METADATA", metadata)
                archive.writestr("_renewable_huber_native_cpu.pyd", b"extension")
                archive.writestr(
                    "renewable_huber_native_cpu-0.6.0.dist-info/licenses/LICENSE", "license"
                )
                archive.writestr(
                    "renewable_huber_native_cpu-0.6.0.dist-info/licenses/NOTICE", "notice"
                )
            errors = _check_native_wheel(
                read_wheel_metadata(wheel),
                kind="cpu",
                expected_version="0.6.0",
                expected_requires_python=SUPPORTED_PYTHON,
            )
            self.assertTrue(any("Requires-Python" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
