"""Fixed per-case sample repetitions shared by every round of an interleaved A/B.

Without a plan, each sweep sizes its sample block from its own warmup timing
(``timing._sample_repetitions``). Separate processes measure slightly
different warmups, so the block size can differ between rounds, and
``interleaved_regression.merge_round_records`` then refuses to merge. The
only way around that was ``--max-sample-repetitions 1``: every statistical
sample became one short operation, and on a desktop GPU in normal use a single
scheduling hiccup is enough to push relative MAD past the gate.

A plan is decided once, before the first timed round, and applied to both
variants and every round. A sample again averages enough work to reach
``--minimum-sample-seconds``, and the block size stays constant because it no
longer depends on any timing taken during the A/B.

This module imports nothing from ``renewable_huber``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

PLAN_SCHEMA = "renewable-huber-sample-repetitions-plan"
PLAN_SCHEMA_VERSION = 1


def plan_key(
    *,
    shape: str,
    dtype: str,
    penalty: str,
    lifecycle: str,
    operation: str,
    engine: str,
) -> str:
    """Identify one measured case before it is measured."""

    return "/".join((shape, dtype, penalty, lifecycle, operation, engine))


def case_plan_key(case: Mapping[str, Any]) -> str:
    """Return the :func:`plan_key` of a measured shape-sweep case."""

    result = case["result"]
    return plan_key(
        shape=str(case["shape"]["name"]),
        dtype=str(case["dtype"]),
        penalty=str(case["penalty"]),
        lifecycle=str(result["lifecycle"]),
        operation=str(result["operation"]),
        engine=str(case["engine"]),
    )


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def plan_from_records(records: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """Choose each case's block size from calibration sweeps of every variant.

    Every record must contain exactly the same cases. A case takes the largest
    block any variant calibrated, so no side's sample falls short of the
    requested minimum duration; each block already respects the sweep's
    ``--max-sample-repetitions``.
    """

    plan: dict[str, int] | None = None
    for record in records:
        repetitions: dict[str, int] = {}
        for case in record["cases"]:
            key = case_plan_key(case)
            value = case["result"].get("sample_repetitions", 1)
            if not _positive_int(value):
                raise ValueError(f"{key}: sample_repetitions must be a positive integer")
            if key in repetitions:
                raise ValueError(f"{key}: duplicate case in a calibration record")
            repetitions[key] = value
        if plan is None:
            plan = repetitions
            continue
        if set(plan) != set(repetitions):
            raise ValueError("calibration records do not contain identical benchmark cases")
        plan = {key: max(plan[key], repetitions[key]) for key in plan}
    if not plan:
        raise ValueError("calibration records contain no measured cases")
    return dict(sorted(plan.items()))


def plan_document(plan: Mapping[str, int]) -> dict[str, Any]:
    return {
        "schema": PLAN_SCHEMA,
        "schema_version": PLAN_SCHEMA_VERSION,
        "sample_repetitions": dict(sorted(plan.items())),
    }


def write_plan(path: Path, plan: Mapping[str, int]) -> str:
    """Write a plan and return the SHA-256 of the exact bytes written."""

    payload = (
        json.dumps(plan_document(plan), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def load_plan(path: Path) -> tuple[dict[str, int], str]:
    """Read and validate a plan; return it with the SHA-256 of its bytes."""

    payload = path.read_bytes()
    document = json.loads(payload.decode("utf-8"))
    if (
        not isinstance(document, dict)
        or document.get("schema") != PLAN_SCHEMA
        or document.get("schema_version") != PLAN_SCHEMA_VERSION
    ):
        raise ValueError(f"{path} is not a {PLAN_SCHEMA} v{PLAN_SCHEMA_VERSION} document")
    entries = document.get("sample_repetitions")
    if not isinstance(entries, dict) or not entries:
        raise ValueError(f"{path} contains no sample_repetitions entries")
    plan: dict[str, int] = {}
    for key, value in entries.items():
        if not isinstance(key, str) or not _positive_int(value):
            raise ValueError(f"{path}: {key!r} must map to a positive integer")
        plan[key] = value
    return plan, hashlib.sha256(payload).hexdigest()


def planned_repetitions(plan: Mapping[str, int] | None, key: str) -> int | None:
    """Return a case's planned block size; a plan must cover every case it sees."""

    if plan is None:
        return None
    try:
        return plan[key]
    except KeyError:
        raise ValueError(
            f"sample repetitions plan has no entry for {key}; "
            "recalibrate with the same benchmark arguments"
        ) from None
