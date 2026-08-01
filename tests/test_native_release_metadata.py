from __future__ import annotations

import tarfile
import tempfile
import unittest
import zipfile
from io import BytesIO
from pathlib import Path

from scripts.native.validate_release_artifacts import (
    _check_native_wheel,
    base_version,
    check_source_metadata,
    check_wheel_set,
    read_wheel_metadata,
)


class NativeReleaseMetadataTests(unittest.TestCase):
    def test_source_projects_match_base_release_exactly(self) -> None:
        self.assertEqual(check_source_metadata(), base_version())

    def test_valid_native_wheel_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory) / "renewable_huber_native_cpu-0.6.0-cp312-cp312-win_amd64.whl"
            metadata = (
                "Metadata-Version: 2.4\n"
                "Name: renewable-huber-native-cpu\n"
                "Version: 0.6.0\n"
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
                read_wheel_metadata(wheel), kind="cpu", expected_version="0.6.0"
            )
            self.assertEqual(errors, [])

    def test_mismatched_native_version_and_dependency_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory) / "renewable_huber_native_cuda-0.5.0-cp312-win_amd64.whl"
            metadata = (
                "Metadata-Version: 2.4\n"
                "Name: renewable-huber-native-cuda\n"
                "Version: 0.5.0\n"
                "Requires-Dist: renewable-huber>=0.5\n\n"
            )
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("renewable_huber_native_cuda-0.5.0.dist-info/METADATA", metadata)
            errors = _check_native_wheel(
                read_wheel_metadata(wheel), kind="cuda", expected_version="0.6.0"
            )
            self.assertTrue(any("Version" in error for error in errors))
            self.assertTrue(any("exact base dependency" in error for error in errors))
            self.assertTrue(any("compiled extension" in error for error in errors))

    def test_complete_set_requires_base_sdist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_dir = root / "base"
            cpu_dir = root / "cpu"
            cuda_dir = root / "cuda"
            base_dir.mkdir()
            cpu_dir.mkdir()
            cuda_dir.mkdir()
            wheel = base_dir / "renewable_huber-0.6.0-py3-none-any.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr(
                    "renewable_huber-0.6.0.dist-info/METADATA",
                    "Metadata-Version: 2.4\nName: renewable-huber\nVersion: 0.6.0\n\n",
                )

            with self.assertRaisesRegex(RuntimeError, "expected one base sdist"):
                check_wheel_set(
                    base_dir=base_dir,
                    cpu_dir=cpu_dir,
                    cuda_dir=cuda_dir,
                    expected_cpu=0,
                    expected_cuda=0,
                )

            sdist = base_dir / "renewable_huber-0.6.0.tar.gz"
            with tarfile.open(sdist, "w:gz") as archive:
                files = {
                    "renewable_huber-0.6.0/PKG-INFO": (
                        b"Metadata-Version: 2.4\nName: renewable-huber\nVersion: 0.6.0\n\n"
                    ),
                    "renewable_huber-0.6.0/LICENSE": b"license",
                    "renewable_huber-0.6.0/NOTICE": b"notice",
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


if __name__ == "__main__":
    unittest.main()
