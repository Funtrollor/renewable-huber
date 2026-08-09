"""Validate base/native source metadata and a complete release wheel set."""

from __future__ import annotations

import argparse
import email
import runpy
import sys
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised by the Python 3.10 CI matrix
    from pip._vendor import tomli as tomllib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_NAME = "renewable-huber"
NATIVE_PROJECTS = {
    "cpu": ("renewable-huber-native-cpu", "_renewable_huber_native_cpu"),
    "cuda": ("renewable-huber-native-cuda", "_renewable_huber_native_cuda"),
}


@dataclass(frozen=True)
class WheelMetadata:
    path: Path
    name: str
    version: str
    requires_dist: tuple[str, ...]
    members: tuple[str, ...]


def base_version(root: Path = PROJECT_ROOT) -> str:
    namespace = runpy.run_path(str(root / "src" / "renewable_huber" / "_version.py"))
    return str(namespace["__version__"])


def _project_metadata(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        return tomllib.load(handle)["project"]


def native_workspace_version(root: Path = PROJECT_ROOT) -> str:
    with (root / "native" / "Cargo.toml").open("rb") as handle:
        manifest = tomllib.load(handle)
    return str(manifest["workspace"]["package"]["version"])


def check_source_metadata(root: Path = PROJECT_ROOT) -> str:
    """Require all separately published projects to use one exact release version."""

    version = base_version(root)
    expected_dependency = f"{BASE_NAME}=={version}"
    errors: list[str] = []
    engine_version = native_workspace_version(root)
    if engine_version != version:
        errors.append(
            f"native/Cargo.toml: workspace engine version {engine_version!r} "
            f"must equal base version {version!r}"
        )
    for kind, (expected_name, _) in NATIVE_PROJECTS.items():
        path = root / "native" / f"python-{kind}" / "pyproject.toml"
        project = _project_metadata(path)
        if project.get("name") != expected_name:
            errors.append(f"{path}: expected project name {expected_name!r}")
        if project.get("version") != version:
            errors.append(f"{path}: version must equal base version {version}")
        dependencies = tuple(str(item) for item in project.get("dependencies", []))
        if dependencies != (expected_dependency,):
            errors.append(f"{path}: dependencies must be [{expected_dependency!r}]")
    if errors:
        raise RuntimeError(
            "Native release source metadata is inconsistent:\n- " + "\n- ".join(errors)
        )
    return version


def read_wheel_metadata(path: Path) -> WheelMetadata:
    with zipfile.ZipFile(path) as archive:
        members = tuple(archive.namelist())
        metadata_members = [name for name in members if name.endswith(".dist-info/METADATA")]
        if len(metadata_members) != 1:
            raise RuntimeError(
                f"{path}: expected one .dist-info/METADATA, found {metadata_members}"
            )
        message = email.message_from_bytes(archive.read(metadata_members[0]))
    return WheelMetadata(
        path=path,
        name=str(message["Name"]),
        version=str(message["Version"]),
        requires_dist=tuple(message.get_all("Requires-Dist", [])),
        members=members,
    )


def read_sdist_metadata(path: Path) -> tuple[str, str, tuple[str, ...]]:
    """Return the core metadata and members from one gzipped source distribution."""

    with tarfile.open(path, "r:gz") as archive:
        members = tuple(member.name for member in archive.getmembers())
        metadata_members = [name for name in members if name.endswith("/PKG-INFO")]
        if len(metadata_members) != 1:
            raise RuntimeError(f"{path}: expected one top-level PKG-INFO, found {metadata_members}")
        extracted = archive.extractfile(metadata_members[0])
        if extracted is None:
            raise RuntimeError(f"{path}: unable to read {metadata_members[0]}")
        message = email.message_from_bytes(extracted.read())
    return str(message["Name"]), str(message["Version"]), members


def _normalized_requirement(requirement: str) -> str:
    return "".join(requirement.lower().split())


def _check_native_wheel(
    wheel: WheelMetadata,
    *,
    kind: str,
    expected_version: str,
) -> list[str]:
    expected_name, module_name = NATIVE_PROJECTS[kind]
    errors: list[str] = []
    if wheel.name != expected_name:
        errors.append(f"{wheel.path.name}: Name is {wheel.name!r}, expected {expected_name!r}")
    if wheel.version != expected_version:
        errors.append(
            f"{wheel.path.name}: Version is {wheel.version!r}, expected {expected_version!r}"
        )
    exact_requirement = _normalized_requirement(f"{BASE_NAME}=={expected_version}")
    requirements = {_normalized_requirement(value) for value in wheel.requires_dist}
    if exact_requirement not in requirements:
        errors.append(
            f"{wheel.path.name}: missing exact base dependency {BASE_NAME}=={expected_version}"
        )
    if not any(
        Path(member).name.startswith(module_name) and Path(member).suffix in {".so", ".pyd"}
        for member in wheel.members
    ):
        errors.append(f"{wheel.path.name}: missing compiled extension {module_name}")
    for legal_name in ("LICENSE", "NOTICE"):
        suffix = f"/licenses/{legal_name}"
        if not any(member.endswith(suffix) for member in wheel.members):
            errors.append(f"{wheel.path.name}: missing {suffix}")
    return errors


def check_wheel_set(
    *,
    base_dir: Path,
    cpu_dir: Path,
    cuda_dir: Path,
    expected_cpu: int | None = None,
    expected_cuda: int | None = None,
    root: Path = PROJECT_ROOT,
) -> dict[str, int | str]:
    """Validate names, versions, dependencies, native modules, and artifact counts."""

    version = check_source_metadata(root)
    base_wheels = sorted(base_dir.glob("renewable_huber-*.whl"))
    base_sdists = sorted(base_dir.glob("renewable_huber-*.tar.gz"))
    cpu_wheels = sorted(cpu_dir.glob("renewable_huber_native_cpu-*.whl"))
    cuda_wheels = sorted(cuda_dir.glob("renewable_huber_native_cuda-*.whl"))
    errors: list[str] = []
    if len(base_wheels) != 1:
        errors.append(f"expected one base wheel, found {len(base_wheels)} in {base_dir}")
    if len(base_sdists) != 1:
        errors.append(f"expected one base sdist, found {len(base_sdists)} in {base_dir}")
    if expected_cpu is not None and len(cpu_wheels) != expected_cpu:
        errors.append(f"expected {expected_cpu} CPU wheels, found {len(cpu_wheels)}")
    if expected_cuda is not None and len(cuda_wheels) != expected_cuda:
        errors.append(f"expected {expected_cuda} CUDA wheels, found {len(cuda_wheels)}")

    if len(base_wheels) == 1:
        base = read_wheel_metadata(base_wheels[0])
        if base.name != BASE_NAME or base.version != version:
            errors.append(
                f"{base.path.name}: expected {BASE_NAME} {version}, got {base.name} {base.version}"
            )
    if len(base_sdists) == 1:
        sdist_name, sdist_version, sdist_members = read_sdist_metadata(base_sdists[0])
        if sdist_name != BASE_NAME or sdist_version != version:
            errors.append(
                f"{base_sdists[0].name}: expected {BASE_NAME} {version}, "
                f"got {sdist_name} {sdist_version}"
            )
        for legal_name in ("LICENSE", "NOTICE"):
            if not any(member.endswith(f"/{legal_name}") for member in sdist_members):
                errors.append(f"{base_sdists[0].name}: missing top-level {legal_name}")
    for path in cpu_wheels:
        errors.extend(
            _check_native_wheel(read_wheel_metadata(path), kind="cpu", expected_version=version)
        )
    for path in cuda_wheels:
        errors.extend(
            _check_native_wheel(read_wheel_metadata(path), kind="cuda", expected_version=version)
        )

    all_names = [path.name for path in [*base_wheels, *base_sdists, *cpu_wheels, *cuda_wheels]]
    duplicates = sorted({name for name in all_names if all_names.count(name) > 1})
    if duplicates:
        errors.append(f"duplicate artifact filenames: {duplicates}")
    if errors:
        raise RuntimeError("Native release artifacts failed validation:\n- " + "\n- ".join(errors))
    return {
        "version": version,
        "base_wheels": len(base_wheels),
        "base_sdists": len(base_sdists),
        "cpu_wheels": len(cpu_wheels),
        "cuda_wheels": len(cuda_wheels),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--base-dir", type=Path)
    parser.add_argument("--cpu-dir", type=Path)
    parser.add_argument("--cuda-dir", type=Path)
    parser.add_argument("--expected-cpu", type=int)
    parser.add_argument("--expected-cuda", type=int)
    args = parser.parse_args(argv)
    if args.source_only:
        print(f"Native source metadata is consistent for {check_source_metadata()}.")
        return 0
    missing = [name for name in ("base_dir", "cpu_dir", "cuda_dir") if getattr(args, name) is None]
    if missing:
        parser.error("wheel-set validation requires --base-dir, --cpu-dir, and --cuda-dir")
    result = check_wheel_set(
        base_dir=args.base_dir,
        cpu_dir=args.cpu_dir,
        cuda_dir=args.cuda_dir,
        expected_cpu=args.expected_cpu,
        expected_cuda=args.expected_cuda,
    )
    print(
        (
            "Validated release {version}: base wheel={base_wheels}, "
            "sdist={base_sdists}, CPU={cpu_wheels}, CUDA={cuda_wheels}"
        ).format(**result)
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(error, file=sys.stderr)
        raise SystemExit(1) from error
