"""Summarize an Nsight Systems SQLite export inside selected NVTX ranges."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


def _milliseconds(nanoseconds: int | float | None) -> float:
    return float(nanoseconds or 0) / 1_000_000.0


def summarize(report: Path, range_prefix: str, top: int) -> dict[str, Any]:
    database = sqlite3.connect(report)
    database.text_factory = lambda raw: raw.decode(errors="replace")
    window = database.execute(
        """
        SELECT MIN(start), MAX(end), COUNT(*)
        FROM NVTX_EVENTS
        WHERE text LIKE ?
        """,
        (f"{range_prefix}%",),
    ).fetchone()
    if window is None or window[0] is None or window[1] is None:
        raise ValueError(f"no NVTX ranges start with {range_prefix!r}")
    start, end, range_count = (int(window[0]), int(window[1]), int(window[2]))
    bounds = (start, end)

    kernel_rows = database.execute(
        """
        SELECT strings.value, COUNT(*), SUM(kernels.end - kernels.start)
        FROM CUPTI_ACTIVITY_KIND_KERNEL AS kernels
        JOIN StringIds AS strings ON strings.id = kernels.demangledName
        WHERE kernels.start >= ? AND kernels.end <= ?
        GROUP BY strings.value
        ORDER BY SUM(kernels.end - kernels.start) DESC
        LIMIT ?
        """,
        (*bounds, top),
    ).fetchall()
    kernel_totals = database.execute(
        """
        SELECT COUNT(*), SUM(end - start)
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        WHERE start >= ? AND end <= ?
        """,
        bounds,
    ).fetchone()

    memcpy_rows = database.execute(
        """
        SELECT operations.label, COUNT(*), SUM(copies.bytes), SUM(copies.end - copies.start)
        FROM CUPTI_ACTIVITY_KIND_MEMCPY AS copies
        JOIN ENUM_CUDA_MEMCPY_OPER AS operations ON operations.id = copies.copyKind
        WHERE copies.start >= ? AND copies.end <= ?
        GROUP BY operations.label
        ORDER BY SUM(copies.end - copies.start) DESC
        """,
        bounds,
    ).fetchall()

    synchronization_rows = database.execute(
        """
        SELECT types.label, COUNT(*), SUM(sync.end - sync.start)
        FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION AS sync
        JOIN ENUM_CUPTI_SYNC_TYPE AS types ON types.id = sync.syncType
        WHERE sync.start >= ? AND sync.end <= ?
        GROUP BY types.label
        ORDER BY SUM(sync.end - sync.start) DESC
        """,
        bounds,
    ).fetchall()

    runtime_rows = database.execute(
        """
        SELECT strings.value, COUNT(*), SUM(runtime.end - runtime.start)
        FROM CUPTI_ACTIVITY_KIND_RUNTIME AS runtime
        JOIN StringIds AS strings ON strings.id = runtime.nameId
        WHERE runtime.start >= ? AND runtime.end <= ?
        GROUP BY strings.value
        ORDER BY SUM(runtime.end - runtime.start) DESC
        LIMIT ?
        """,
        (*bounds, top),
    ).fetchall()
    database.close()

    return {
        "schema": "renewable-huber-nsys-summary",
        "schema_version": 1,
        "source": str(report),
        "nvtx": {
            "range_prefix": range_prefix,
            "range_count": range_count,
            "window_milliseconds": _milliseconds(end - start),
        },
        "kernels": {
            "count": int(kernel_totals[0] or 0),
            "total_milliseconds": _milliseconds(kernel_totals[1]),
            "top": [
                {
                    "name": name,
                    "count": int(count),
                    "total_milliseconds": _milliseconds(duration),
                }
                for name, count, duration in kernel_rows
            ],
        },
        "memcopies": [
            {
                "kind": label,
                "count": int(count),
                "bytes": int(byte_count or 0),
                "total_milliseconds": _milliseconds(duration),
            }
            for label, count, byte_count, duration in memcpy_rows
        ],
        "synchronizations": [
            {
                "kind": label,
                "count": int(count),
                "total_milliseconds": _milliseconds(duration),
            }
            for label, count, duration in synchronization_rows
        ],
        "runtime_api": [
            {
                "name": name,
                "count": int(count),
                "total_milliseconds": _milliseconds(duration),
            }
            for name, count, duration in runtime_rows
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--range-prefix", default="profile/repeat-")
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.top < 1:
        parser.error("top must be positive")
    if not args.report.is_file():
        parser.error(f"report does not exist: {args.report}")

    try:
        summary = summarize(args.report, args.range_prefix, args.top)
    except (sqlite3.DatabaseError, ValueError) as error:
        parser.error(str(error))
    if args.metadata is not None:
        if not args.metadata.is_file():
            parser.error(f"metadata does not exist: {args.metadata}")
        summary["workload_metadata"] = json.loads(args.metadata.read_text(encoding="utf-8"))
    rendered = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"Wrote Nsight summary to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
