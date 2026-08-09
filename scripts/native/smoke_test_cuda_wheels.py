"""Install matching base/native CUDA wheels into a clean environment and smoke-test them."""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
import venv
import zipfile
from pathlib import Path

if __package__:
    from .validate_release_artifacts import _check_native_wheel, read_wheel_metadata
else:
    from validate_release_artifacts import _check_native_wheel, read_wheel_metadata


def _one_wheel(directory: Path, pattern: str) -> Path:
    wheels = sorted(directory.glob(pattern))
    if len(wheels) != 1:
        raise RuntimeError(f"expected one {pattern} in {directory}, found {wheels}")
    return wheels[0].resolve()


def _assert_native_legal_files(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
    for suffix in ("/licenses/LICENSE", "/licenses/NOTICE"):
        if not any(name.endswith(suffix) for name in names):
            raise RuntimeError(f"{wheel.name} is missing {suffix}")


def _smoke_program(version: str, *, import_only: bool) -> str:
    common = (
        "import numpy as np; "
        "from renewable_huber import RenewableHuberRegressor, __version__; "
        "from renewable_huber import _native_cuda; "
        f"assert __version__=={version!r}; "
        "assert _native_cuda.version()['abi_version']==1; "
        "assert _native_cuda.version()['python_api_version']==3; "
        "assert _native_cuda.version()['supports_cuda_graphs']; "
        "assert _native_cuda.version()['supports_fast_math']; "
    )
    if import_only:
        return common + "assert isinstance(_native_cuda.is_available(), bool)"
    return (
        common + "assert _native_cuda.is_available(); "
        "X=np.arange(24,dtype=np.float32).reshape(8,3); "
        "y=np.arange(8,dtype=np.float32); "
        "m=RenewableHuberRegressor(backend='native_cuda',device='cuda',"
        "dtype='float32',fit_intercept=False,max_iter=20).fit(X,y); "
        "assert m.predict(X).shape==(8,)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--find-links", type=Path)
    parser.add_argument(
        "--import-only",
        action="store_true",
        help="validate wheel loading and the ABI without requiring a CUDA device",
    )
    args = parser.parse_args()

    base_wheel = _one_wheel(args.base_dir, "renewable_huber-*.whl")
    native_wheel = _one_wheel(args.native_dir, "renewable_huber_native_cuda-*.whl")
    _assert_native_legal_files(native_wheel)
    base_metadata = read_wheel_metadata(base_wheel)
    native_errors = _check_native_wheel(
        read_wheel_metadata(native_wheel),
        kind="cuda",
        expected_version=base_metadata.version,
    )
    if native_errors:
        raise RuntimeError("incompatible CUDA wheel metadata:\n- " + "\n- ".join(native_errors))

    with tempfile.TemporaryDirectory(prefix="renewable-huber-native-cuda-") as directory:
        environment = Path(directory)
        venv.EnvBuilder(with_pip=True, clear=True).create(environment)
        python = (
            environment / "Scripts" / "python.exe"
            if os.name == "nt"
            else environment / "bin" / "python"
        )
        install = [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
        ]
        if args.find_links is not None:
            install.extend(["--no-index", "--find-links", str(args.find_links.resolve())])
        install.extend([str(base_wheel), str(native_wheel)])
        subprocess.run(install, check=True, cwd=environment)
        subprocess.run([str(python), "-m", "pip", "check"], check=True, cwd=environment)
        subprocess.run(
            [
                str(python),
                "-c",
                _smoke_program(base_metadata.version, import_only=args.import_only),
            ],
            check=True,
            cwd=environment,
        )
    print(f"Clean wheel smoke passed: {base_wheel.name} + {native_wheel.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
