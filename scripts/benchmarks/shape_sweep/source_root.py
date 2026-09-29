"""Which ``renewable_huber`` source tree a shape sweep measures.

By default the sweep measures the checkout it lives in. An interleaved A/B
with frozen sample repetitions instead runs *one* harness, the candidate's,
against both source trees, so the measurement code is identical on both sides
and only the code under test differs. The runner selects the tree with
:data:`ENVIRONMENT_VARIABLE`; every module that imports ``renewable_huber``
calls :func:`put_source_on_path` first, so none of them can fall back to the
harness's own ``src`` and silently measure the wrong code.

This module imports nothing from ``renewable_huber`` and nothing heavy, so the
runner can use it without loading an engine.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

#: Repository root whose ``src`` the sweep imports, when set.
ENVIRONMENT_VARIABLE = "RENEWABLE_HUBER_BENCHMARK_SOURCE_ROOT"

#: The checkout this harness belongs to.
HARNESS_ROOT = Path(__file__).resolve().parents[3]


def source_root() -> Path:
    """Return the repository root whose ``src`` is being measured."""

    value = os.environ.get(ENVIRONMENT_VARIABLE)
    if not value:
        return HARNESS_ROOT
    root = Path(value).resolve()
    if not (root / "src" / "renewable_huber" / "__init__.py").is_file():
        raise RuntimeError(f"{ENVIRONMENT_VARIABLE}={value!r} does not contain src/renewable_huber")
    return root


def is_overridden() -> bool:
    """Return whether the measured source tree differs from the harness."""

    return source_root() != HARNESS_ROOT


def put_source_on_path() -> Path:
    """Make the measured tree's ``src`` win the ``renewable_huber`` import.

    Without an override this keeps the historical behaviour exactly: the
    harness's ``src`` is prepended only if it is not already importable. With
    one, the selected ``src`` is moved to the front and the harness's own
    ``src`` is removed, because leaving it anywhere on the path would let a
    later insertion shadow the tree under test.
    """

    root = source_root()
    entry = str(root / "src")
    if root == HARNESS_ROOT:
        if entry not in sys.path:
            sys.path.insert(0, entry)
        return root
    harness_entry = str(HARNESS_ROOT / "src")
    sys.path[:] = [path for path in sys.path if path not in (entry, harness_entry)]
    sys.path.insert(0, entry)
    return root


def assert_measuring(module_file: str | None) -> None:
    """Fail unless an imported ``renewable_huber`` came from the measured tree."""

    if module_file is None:
        raise RuntimeError("renewable_huber has no __file__; cannot verify its source tree")
    expected = source_root() / "src" / "renewable_huber"
    if Path(module_file).resolve().parent != expected:
        raise RuntimeError(
            f"renewable_huber was imported from {Path(module_file).resolve().parent}, "
            f"not the measured source tree {expected}"
        )
