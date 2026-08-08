"""Report whether a WSL development profile is actually usable."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
from typing import Any

PROFILE_MODULES = {
    "minimal": ("numpy", "renewable_huber", "renewable_huber._native_cpu"),
    "cpu-full": (
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "torch",
        "tensorflow",
        "renewable_huber",
        "renewable_huber._native_cpu",
    ),
    "cuda-full": (
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "torch",
        "tensorflow",
        "cupy",
        "renewable_huber",
        "renewable_huber._native_cpu",
        "renewable_huber._native_cuda",
    ),
}
ALL_MODULES = tuple(dict.fromkeys(name for names in PROFILE_MODULES.values() for name in names))


def _probe_module(name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(name)
    except Exception as error:  # Import-time native loader failures are relevant here.
        return {"status": "failed", "error": f"{type(error).__name__}: {error}"}
    result: dict[str, Any] = {
        "status": "passed",
        "version": str(getattr(module, "__version__", "unknown")),
    }
    version = getattr(module, "version", None)
    if name.startswith("renewable_huber._native_") and callable(version):
        result["native"] = version()
    return result


def verify(profile: str) -> dict[str, Any]:
    required = frozenset(PROFILE_MODULES[profile])
    modules = {
        name: (
            _probe_module(name)
            if name in required
            else {"status": "skipped", "reason": f"not required by {profile}"}
        )
        for name in ALL_MODULES
    }
    checks: dict[str, Any] = {
        "platform": {
            "status": "passed" if "microsoft" in platform.release().lower() else "failed",
            "release": platform.release(),
        },
        "modules": modules,
    }
    if profile == "cuda-full" and modules["cupy"]["status"] == "passed":
        try:
            cupy = importlib.import_module("cupy")
            device_count = int(cupy.cuda.runtime.getDeviceCount())
            checks["cuda_device"] = {
                "status": "passed" if device_count > 0 else "failed",
                "device_count": device_count,
            }
        except Exception as error:
            checks["cuda_device"] = {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            }
    passed = checks["platform"]["status"] == "passed" and all(
        modules[name]["status"] == "passed" for name in required
    )
    if "cuda_device" in checks:
        passed = passed and checks["cuda_device"]["status"] == "passed"
    return {"profile": profile, "passed": passed, "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILE_MODULES), default="minimal")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = verify(args.profile)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"WSL profile {args.profile}: {'PASS' if report['passed'] else 'FAIL'}")
        for name, result in report["checks"]["modules"].items():
            detail = result.get("version", result.get("error", result.get("reason", "")))
            print(f"  {name:<34} {result['status'].upper():<6} {detail}")
        if cuda := report["checks"].get("cuda_device"):
            detail = cuda.get("device_count", cuda.get("error", ""))
            print(f"  {'CUDA device':<34} {cuda['status'].upper():<6} {detail}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
