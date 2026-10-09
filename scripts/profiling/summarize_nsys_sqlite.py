"""Summarize an Nsight Systems SQLite export inside selected NVTX ranges.

Besides the per-name totals, schema 2 adds a wall-clock breakdown of every
``profile/batch-*`` range (one ``partial_fit`` call). Each batch is cut at the
first and last CUDA API call inside its ``phase/update`` range (or the batch
itself when the workload ran without ``--phase-ranges``). Outside that native
segment the time is Python and binding work. Inside it, every instant is
assigned to exactly one class, in this order of precedence:

1. ``gpu_compute``: a kernel is executing;
2. ``h2d``: a host-to-device copy is executing, or the host is inside the
   API call that issued one (a pageable copy blocks the caller there);
3. ``other_copies``: another copy or a memset is executing;
4. ``launch_and_sync``: none of the above, so the GPU is waiting for the host
   to launch work or to read back a result.

The classes add up to the batch wall time exactly, unlike the per-name API,
kernel and copy totals, which overlap one another.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

#: Kernel launches issued through the runtime or driver API, and graph launches.
#: cuBLAS also polls cuStreamGetCaptureInfo before most calls; that is not a
#: launch and is counted on its own.
_LAUNCH_NAMES = ("cudaLaunchKernel", "cuLaunchKernel", "cudaGraphLaunch", "cuGraphLaunch")
_GRAPH_LAUNCH_NAMES = ("cudaGraphLaunch", "cuGraphLaunch")
_BREAKDOWN_CLASSES = (
    "gpu_compute",
    "h2d",
    "other_copies",
    "launch_and_sync",
    "python_and_binding",
)


def _milliseconds(nanoseconds: int | float | None) -> float:
    return float(nanoseconds or 0) / 1_000_000.0


def _covered(intervals: list[tuple[int, int]], low: int, high: int) -> int:
    """Length of the union of ``intervals`` clipped to ``[low, high]``."""

    total = 0
    reach = low
    for start, end in sorted(intervals):
        start = max(start, reach)
        end = min(end, high)
        if end > start:
            total += end - start
            reach = end
    return total


def _within(intervals: list[tuple[int, int]], low: int, high: int) -> list[tuple[int, int]]:
    return [(start, end) for start, end in intervals if start >= low and end <= high]


def _intervals(database: sqlite3.Connection, query: str, bounds: tuple[int, int]) -> list:
    return [(int(start), int(end), *rest) for start, end, *rest in database.execute(query, bounds)]


def _breakdown(
    database: sqlite3.Connection,
    bounds: tuple[int, int],
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    """Partition each batch's wall time and count the work per batch and iteration."""

    def nvtx(pattern: str) -> list[tuple[int, int]]:
        return [
            (int(start), int(end))
            for start, end in database.execute(
                "SELECT start, end FROM NVTX_EVENTS "
                "WHERE text LIKE ? AND start >= ? AND end <= ? ORDER BY start",
                (pattern, *bounds),
            )
        ]

    batches = nvtx("profile/batch-%")
    updates = nvtx("phase/update")
    prepares = nvtx("phase/prepare")
    commits = nvtx("phase/commit")
    kernels = [
        (start, end)
        for start, end in _intervals(
            database,
            "SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start >= ? AND end <= ?",
            bounds,
        )
    ]
    copies = _intervals(
        database,
        """
        SELECT copies.start, copies.end, copies.correlationId, operations.label,
               copies.bytes, source.label
        FROM CUPTI_ACTIVITY_KIND_MEMCPY AS copies
        JOIN ENUM_CUDA_MEMCPY_OPER AS operations ON operations.id = copies.copyKind
        LEFT JOIN ENUM_CUDA_MEM_KIND AS source ON source.id = copies.srcKind
        WHERE copies.start >= ? AND copies.end <= ?
        """,
        bounds,
    )
    memsets = [
        (start, end)
        for start, end in _intervals(
            database,
            "SELECT start, end FROM CUPTI_ACTIVITY_KIND_MEMSET WHERE start >= ? AND end <= ?",
            bounds,
        )
    ]
    api = _intervals(
        database,
        """
        SELECT runtime.start, runtime.end, runtime.correlationId, strings.value
        FROM CUPTI_ACTIVITY_KIND_RUNTIME AS runtime
        JOIN StringIds AS strings ON strings.id = runtime.nameId
        WHERE runtime.start >= ? AND runtime.end <= ?
        ORDER BY runtime.start
        """,
        bounds,
    )

    h2d_rows = [row for row in copies if row[3] == "Host-to-Device"]
    h2d_correlations = {int(row[2]) for row in h2d_rows}
    h2d_dma = [(row[0], row[1]) for row in h2d_rows]
    h2d_api = [(row[0], row[1]) for row in api if int(row[2]) in h2d_correlations]
    other_copies = [(row[0], row[1]) for row in copies if row[3] != "Host-to-Device"] + memsets

    totals = dict.fromkeys(
        (
            "wall",
            "native_segment",
            "input_validation",
            "update_entry",
            "update_exit",
            "commit",
            "final_synchronize",
            *_BREAKDOWN_CLASSES,
        ),
        0,
    )
    counts = dict.fromkeys(
        (
            "kernels_executed",
            "launch_calls",
            "graph_launches",
            "stream_synchronizations",
            "memcpy_calls",
            "capture_info_polls",
        ),
        0,
    )
    h2d = {"bytes": 0, "dma": 0, "api": 0}
    h2d_source_kinds: dict[str, int] = {}

    for low, high in batches:
        wall = high - low
        totals["wall"] += wall
        totals["input_validation"] += sum(
            end - start for start, end in _within(prepares, low, high)
        )
        totals["commit"] += sum(end - start for start, end in _within(commits, low, high))
        update = next(iter(_within(updates, low, high)), (low, high))
        calls = [row for row in api if row[0] >= update[0] and row[1] <= update[1]]
        if not calls:
            totals["python_and_binding"] += wall
            continue
        first = calls[0][0]
        last = max(row[1] for row in calls)
        segment = last - first
        totals["native_segment"] += segment
        totals["python_and_binding"] += wall - segment
        totals["update_entry"] += first - update[0]
        totals["update_exit"] += update[1] - last

        compute = _covered(kernels, first, last)
        through_h2d = _covered(kernels + h2d_dma + h2d_api, first, last)
        through_copies = _covered(kernels + h2d_dma + h2d_api + other_copies, first, last)
        totals["gpu_compute"] += compute
        totals["h2d"] += through_h2d - compute
        totals["other_copies"] += through_copies - through_h2d
        totals["launch_and_sync"] += segment - through_copies

        synchronizations = [row for row in calls if str(row[3]).startswith("cudaStreamSynchronize")]
        if synchronizations:
            totals["final_synchronize"] += synchronizations[-1][1] - synchronizations[-1][0]
        counts["stream_synchronizations"] += len(synchronizations)
        counts["kernels_executed"] += len(_within(kernels, first, last))
        for start, end, correlation, name in calls:
            name = str(name)
            counts["launch_calls"] += name.startswith(_LAUNCH_NAMES)
            counts["graph_launches"] += name.startswith(_GRAPH_LAUNCH_NAMES)
            counts["memcpy_calls"] += name.startswith("cudaMemcpyAsync")
            counts["capture_info_polls"] += name.startswith("cuStreamGetCaptureInfo")
            if int(correlation) in h2d_correlations:
                h2d["api"] += end - start
        for start, end, _, _, byte_count, source in h2d_rows:
            if start >= first and end <= last:
                h2d["bytes"] += int(byte_count or 0)
                h2d["dma"] += end - start
                kind = str(source or "unknown")
                h2d_source_kinds[kind] = h2d_source_kinds.get(kind, 0) + int(byte_count or 0)

    batch_count = len(batches)
    wall = totals["wall"]
    iterations = None
    if metadata is not None and metadata.get("batch_iterations"):
        iterations = sum(sum(repeat) for repeat in metadata["batch_iterations"])
    result: dict[str, Any] = {
        "batch_count": batch_count,
        "iterations": iterations,
        "phase_ranges": bool(updates),
        "wall_milliseconds": _milliseconds(wall),
        "classes_milliseconds": {name: _milliseconds(totals[name]) for name in _BREAKDOWN_CLASSES},
        "classes_fraction": {
            name: (totals[name] / wall if wall else 0.0) for name in _BREAKDOWN_CLASSES
        },
        "python_and_binding_detail_milliseconds": {
            name: _milliseconds(totals[name])
            for name in ("input_validation", "update_entry", "update_exit", "commit")
        },
        "native_segment_milliseconds": _milliseconds(totals["native_segment"]),
        "final_synchronize_milliseconds": _milliseconds(totals["final_synchronize"]),
        "per_batch": {
            "wall_milliseconds": _milliseconds(wall / batch_count) if batch_count else 0.0,
            **{
                name: (count / batch_count if batch_count else 0.0)
                for name, count in counts.items()
            },
        },
        "h2d": {
            "bytes": h2d["bytes"],
            "dma_milliseconds": _milliseconds(h2d["dma"]),
            "api_milliseconds": _milliseconds(h2d["api"]),
            # bytes per nanosecond is gigabytes per second
            "dma_gigabytes_per_second": h2d["bytes"] / h2d["dma"] if h2d["dma"] else 0.0,
            "api_gigabytes_per_second": h2d["bytes"] / h2d["api"] if h2d["api"] else 0.0,
            "source_kind_bytes": h2d_source_kinds,
        },
    }
    if iterations:
        result["per_iteration"] = {
            "wall_milliseconds": _milliseconds(wall / iterations),
            **{name: count / iterations for name, count in counts.items()},
        }
        configuration = (metadata or {}).get("configuration", {})
        if configuration.get("engine") == "native_cuda":
            # The native solvers synchronize once per line-search candidate,
            # once for the initial objective and once at completion. A
            # resident-engine repeat restores empty state on its first batch,
            # which adds one synchronization per repeat. The LU and SVD
            # fallbacks add their own, so this is exact for the Cholesky path.
            restores = int(configuration.get("repeats", 0))
            candidates = counts["stream_synchronizations"] - 2 * batch_count - restores
            result["line_search_candidates_per_iteration"] = candidates / iterations
    return result


def summarize(
    report: Path,
    range_prefix: str,
    top: int,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
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
    breakdown = _breakdown(database, bounds, metadata)
    database.close()

    return {
        "schema": "renewable-huber-nsys-summary",
        "schema_version": 2,
        "source": str(report),
        "nvtx": {
            "range_prefix": range_prefix,
            "range_count": range_count,
            "window_milliseconds": _milliseconds(end - start),
        },
        "breakdown": breakdown,
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
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
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

    metadata = None
    if args.metadata is not None:
        if not args.metadata.is_file():
            parser.error(f"metadata does not exist: {args.metadata}")
        metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    try:
        summary = summarize(args.report, args.range_prefix, args.top, metadata)
    except (sqlite3.DatabaseError, ValueError) as error:
        parser.error(str(error))
    if metadata is not None:
        summary["workload_metadata"] = metadata
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
