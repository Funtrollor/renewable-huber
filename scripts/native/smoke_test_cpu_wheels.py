"""Install matching base/native CPU wheels into a clean environment and smoke-test them."""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
import venv
import zipfile
from pathlib import Path


def _one_wheel(directory: Path, pattern: str) -> Path:
    wheels = sorted(directory.glob(pattern))
    if len(wheels) != 1:
        raise RuntimeError(f"expected one {pattern} in {directory}, found {wheels}")
    return wheels[0].resolve()


def _assert_native_legal_files(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
    required_suffixes = ("/licenses/LICENSE", "/licenses/NOTICE")
    for suffix in required_suffixes:
        if not any(name.endswith(suffix) for name in names):
            raise RuntimeError(f"{wheel.name} is missing {suffix}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--find-links", type=Path)
    args = parser.parse_args()

    base_wheel = _one_wheel(args.base_dir, "renewable_huber-*.whl")
    native_wheel = _one_wheel(args.native_dir, "renewable_huber_native_cpu-*.whl")
    _assert_native_legal_files(native_wheel)

    with tempfile.TemporaryDirectory(prefix="renewable-huber-native-cpu-") as directory:
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
        subprocess.run(
            [str(python), "-m", "pip", "check"],
            check=True,
            cwd=environment,
        )
        subprocess.run(
            [
                str(python),
                "-c",
                (
                    "import numpy as np; "
                    "from renewable_huber import RenewableHuberRegressor, __version__; "
                    "from renewable_huber import _native_cpu; "
                    "X=np.arange(24,dtype=np.float64).reshape(8,3); "
                    "y=np.arange(8,dtype=np.float64); "
                    "m=RenewableHuberRegressor(backend='native_cpu',"
                    "fit_intercept=False,max_iter=20).fit(X,y); "
                    "assert __version__.startswith('0.6.'); "
                    "assert _native_cpu.version()['python_api_version']==1; "
                    "assert m.predict(X).shape==(8,)"
                ),
            ],
            check=True,
            cwd=environment,
        )
    print(f"Clean wheel smoke passed: {base_wheel.name} + {native_wheel.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
