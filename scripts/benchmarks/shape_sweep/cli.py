"""Argument parsing, sweep orchestration and schema-v2 JSON output.

Every unsupported combination is written into ``skipped`` with an explicit
reason. Silence is not an acceptable way to report that a case did not run:
a consumer comparing two records must be able to see that native CUDA declined
an L1 case rather than infer it from an absence.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

from scripts.benchmarks.shape_sweep.source_root import (
    assert_measuring,
    is_overridden,
    put_source_on_path,
)

# ``src`` must be importable before ``renewable_huber``; see ``source_root``.
put_source_on_path()

import renewable_huber  # noqa: E402
from renewable_huber import BackendUnavailableError  # noqa: E402
from scripts.benchmarks.shape_sweep.environment import environment_metadata  # noqa: E402
from scripts.benchmarks.shape_sweep.runners import (  # noqa: E402
    benchmark_cupy,
    benchmark_native_cpu,
    benchmark_native_cuda,
    benchmark_numpy,
)
from scripts.benchmarks.shape_sweep.sampling_plan import (  # noqa: E402
    load_plan,
    plan_key,
    planned_repetitions,
)
from scripts.benchmarks.shape_sweep.shapes import (  # noqa: E402
    PROFILES,
    SCHEMA,
    SCHEMA_VERSION,
    _dataset_checksum,
    _estimated_host_mib,
    make_batches,
)

DESCRIPTION = "Run reproducible shape sweeps across NumPy and native CPU/CUDA engines."


def _add_throughput(result: dict[str, Any], samples: int) -> None:
    result["median_samples_per_second"] = samples / result["median_seconds"]


def _print_result(case: dict[str, Any]) -> None:
    result = case["result"]
    print(
        f"{case['shape']['name']} {case['penalty']} {case['dtype']} "
        f"{result['lifecycle']}/{result['operation']} {case['engine']}: "
        f"{result['median_seconds']:.4f}s, "
        f"{result['median_samples_per_second']:,.0f} samples/s"
    )


def _record_skip(
    record: dict[str, Any],
    base: dict[str, Any],
    *,
    engine: str,
    lifecycle: str,
    operation: str,
    input_location: str,
    reason: str,
) -> None:
    record.setdefault("skipped", []).append(
        {
            **base,
            "engine": engine,
            "lifecycle": lifecycle,
            "operation": operation,
            "input_location": input_location,
            "reason": reason,
        }
    )
    print(
        f"Skipped {base['shape']['name']} {base['penalty']} {base['dtype']} "
        f"{lifecycle}/{operation} {engine}: {reason}"
    )


def _planned_repetitions(
    plan: dict[str, int] | None,
    *,
    shape: str,
    dtype: str,
    penalty: str,
    lifecycle: str,
    operation: str,
    engine: str,
) -> int | None:
    """Return a case's planned sample block size, or ``None`` without a plan."""

    return planned_repetitions(
        plan,
        plan_key(
            shape=shape,
            dtype=dtype,
            penalty=penalty,
            lifecycle=lifecycle,
            operation=operation,
            engine=engine,
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="smoke")
    parser.add_argument("--case", action="append", help="Run only a named shape; repeatable")
    parser.add_argument(
        "--backend",
        choices=(
            "numpy",
            "native-cpu",
            "cpu",
            "cupy",
            "gpu",
            "native_cuda",
            "both",
            "all",
        ),
        default="both",
    )
    parser.add_argument("--penalty", choices=("none", "l1", "both"), default="both")
    parser.add_argument("--dtype", choices=("float32", "float64", "both"), default="both")
    parser.add_argument(
        "--lifecycle",
        choices=("cold", "steady", "both"),
        default="cold",
        help=(
            "Cold includes a new estimator/native engine in every repeat; steady reuses a "
            "primed model and restores empty state outside timing."
        ),
    )
    parser.add_argument(
        "--operation",
        choices=("fit", "partial-fit", "both"),
        default="partial-fit",
        help="Measure public fit (one full batch) or streaming partial_fit calls.",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--minimum-sample-seconds",
        type=float,
        default=0.1,
        help=(
            "Target aggregate timed work per statistical sample. Short operations are "
            "repeated independently and reported as a per-operation arithmetic mean."
        ),
    )
    parser.add_argument(
        "--max-sample-repetitions",
        type=int,
        default=64,
        help="Safety cap for independent operations aggregated into one timing sample.",
    )
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-host-memory-mib", type=float, default=2048.0)
    parser.add_argument(
        "--sample-repetitions-plan",
        type=Path,
        help=(
            "Use the fixed per-case sample block sizes in this plan instead of sizing each "
            "block from this run's warmup. Written by run_interleaved_benchmark.py "
            "--freeze-sample-repetitions; every measured case must have an entry."
        ),
    )
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if (
        args.warmup < 0
        or args.repeats < 1
        or args.max_iter < 1
        or args.minimum_sample_seconds < 0
        or args.max_sample_repetitions < 1
    ):
        parser.error(
            "warmup and minimum-sample-seconds must be non-negative; repeats, "
            "max-iter, and max-sample-repetitions must be positive"
        )

    if is_overridden():
        assert_measuring(renewable_huber.__file__)
    plan: dict[str, int] | None = None
    plan_sha256: str | None = None
    if args.sample_repetitions_plan is not None:
        try:
            plan, plan_sha256 = load_plan(args.sample_repetitions_plan)
        except (OSError, ValueError) as error:
            parser.error(f"--sample-repetitions-plan: {error}")

    shapes = list(PROFILES[args.profile])
    if args.case:
        requested = set(args.case)
        known = {shape.name for shape in shapes}
        if unknown := requested - known:
            parser.error(f"unknown case(s) for {args.profile}: {', '.join(sorted(unknown))}")
        shapes = [shape for shape in shapes if shape.name in requested]
    penalties = ("none", "l1") if args.penalty == "both" else (args.penalty,)
    dtypes = ("float32", "float64") if args.dtype == "both" else (args.dtype,)
    lifecycles = ("cold", "steady") if args.lifecycle == "both" else (args.lifecycle,)
    operations = (
        ("fit", "partial_fit") if args.operation == "both" else (args.operation.replace("-", "_"),)
    )
    run_numpy = args.backend in ("numpy", "cpu", "both", "all")
    run_native_cpu = args.backend in ("native-cpu", "cpu", "all")
    run_cupy = args.backend in ("cupy", "gpu", "both", "all")
    run_native_cuda = args.backend in ("native_cuda", "gpu", "all")

    record: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "profile": args.profile,
        "environment": environment_metadata(),
        "arguments": {
            "warmup": args.warmup,
            "repeats": args.repeats,
            "max_iter": args.max_iter,
            "tol": args.tol,
            "seed": args.seed,
            "lifecycle": list(lifecycles),
            "operation": list(operations),
            "minimum_sample_seconds": args.minimum_sample_seconds,
            "max_sample_repetitions": args.max_sample_repetitions,
        },
        "cases": [],
    }
    if plan_sha256 is not None:
        # Absent without a plan, so a default record is unchanged.
        record["arguments"]["sample_repetitions_plan_sha256"] = plan_sha256

    profile_shape_indexes = {
        shape.name: index for index, shape in enumerate(PROFILES[args.profile])
    }
    for shape in shapes:
        for dtype in dtypes:
            estimated_mib = _estimated_host_mib(shape, dtype)
            if estimated_mib > args.max_host_memory_mib:
                parser.error(
                    f"{shape.name}/{dtype} needs about {estimated_mib:.1f} MiB host memory; "
                    "raise --max-host-memory-mib explicitly to proceed"
                )
            shape_seed = args.seed + profile_shape_indexes[shape.name]
            batches = make_batches(shape, seed=shape_seed, dtype=dtype)
            dataset_sha256 = _dataset_checksum(batches)
            for penalty in penalties:
                base = {
                    "shape": asdict(shape),
                    "dataset_seed": shape_seed,
                    "dataset_sha256": dataset_sha256,
                    "dtype": dtype,
                    "penalty": penalty,
                    "max_iter": args.max_iter,
                    "tol": args.tol,
                    "estimated_host_data_mib": estimated_mib,
                }
                for lifecycle in lifecycles:
                    for operation in operations:
                        planned = partial(
                            _planned_repetitions,
                            plan,
                            shape=shape.name,
                            dtype=dtype,
                            penalty=penalty,
                            lifecycle=lifecycle,
                            operation=operation,
                        )
                        # ``fit`` concatenates the generated stream before the
                        # timed call, so its real public-API batch contains all
                        # samples. Keep the comparison key honest instead of
                        # retaining the generation chunk size used by
                        # ``partial_fit``.
                        case_base = (
                            base
                            if operation == "partial_fit"
                            else {
                                **base,
                                "shape": {
                                    **base["shape"],
                                    "batch_size": shape.samples,
                                },
                            }
                        )
                        # ``fit`` deliberately calls reset(), so a resident-engine
                        # steady-state variant would not describe the public API.
                        if lifecycle == "steady" and operation == "fit":
                            if run_numpy:
                                _record_skip(
                                    record,
                                    case_base,
                                    engine="numpy_cpu",
                                    lifecycle=lifecycle,
                                    operation=operation,
                                    input_location="host",
                                    reason=(
                                        "fit resets the estimator; steady state is partial_fit only"
                                    ),
                                )
                            if run_native_cpu:
                                _record_skip(
                                    record,
                                    case_base,
                                    engine="rust_native_cpu",
                                    lifecycle=lifecycle,
                                    operation=operation,
                                    input_location="host",
                                    reason=(
                                        "fit resets the estimator; steady state is partial_fit only"
                                    ),
                                )
                            if run_cupy:
                                for engine, input_location in (
                                    ("cupy_cuda_host_input", "host"),
                                    ("cupy_cuda_device_input", "device"),
                                ):
                                    _record_skip(
                                        record,
                                        case_base,
                                        engine=engine,
                                        lifecycle=lifecycle,
                                        operation=operation,
                                        input_location=input_location,
                                        reason=(
                                            "fit resets the estimator; "
                                            "steady state is partial_fit only"
                                        ),
                                    )
                            if run_native_cuda:
                                _record_skip(
                                    record,
                                    case_base,
                                    engine="native_cuda_host_input",
                                    lifecycle=lifecycle,
                                    operation=operation,
                                    input_location="host",
                                    reason=(
                                        "fit resets the estimator; steady state is partial_fit only"
                                    ),
                                )
                            continue
                        if run_numpy:
                            result = benchmark_numpy(
                                batches,
                                dtype=dtype,
                                penalty=penalty,
                                lifecycle=lifecycle,
                                operation=operation,
                                warmup=args.warmup,
                                repeats=args.repeats,
                                max_iter=args.max_iter,
                                tol=args.tol,
                                minimum_sample_seconds=args.minimum_sample_seconds,
                                max_sample_repetitions=args.max_sample_repetitions,
                                sample_repetitions=planned(engine="numpy_cpu"),
                            )
                            _add_throughput(result, shape.samples)
                            case = {**case_base, "engine": "numpy_cpu", "result": result}
                            record["cases"].append(case)
                            _print_result(case)
                        if run_native_cpu:
                            try:
                                result = benchmark_native_cpu(
                                    batches,
                                    dtype=dtype,
                                    penalty=penalty,
                                    lifecycle=lifecycle,
                                    operation=operation,
                                    warmup=args.warmup,
                                    repeats=args.repeats,
                                    max_iter=args.max_iter,
                                    tol=args.tol,
                                    minimum_sample_seconds=args.minimum_sample_seconds,
                                    max_sample_repetitions=args.max_sample_repetitions,
                                    sample_repetitions=planned(engine="rust_native_cpu"),
                                )
                            except (BackendUnavailableError, ImportError) as error:
                                record.setdefault("unavailable", {})["native_cpu"] = str(error)
                                print(f"Native CPU unavailable: {error}")
                                run_native_cpu = False
                            else:
                                _add_throughput(result, shape.samples)
                                case = {
                                    **case_base,
                                    "engine": "rust_native_cpu",
                                    "result": result,
                                }
                                record["cases"].append(case)
                                _print_result(case)
                        if run_cupy:
                            try:
                                host_result, device_result = benchmark_cupy(
                                    batches,
                                    dtype=dtype,
                                    penalty=penalty,
                                    lifecycle=lifecycle,
                                    operation=operation,
                                    warmup=args.warmup,
                                    repeats=args.repeats,
                                    max_iter=args.max_iter,
                                    tol=args.tol,
                                    minimum_sample_seconds=args.minimum_sample_seconds,
                                    max_sample_repetitions=args.max_sample_repetitions,
                                    host_sample_repetitions=planned(engine="cupy_cuda_host_input"),
                                    device_sample_repetitions=planned(
                                        engine="cupy_cuda_device_input"
                                    ),
                                )
                            except (BackendUnavailableError, ImportError) as error:
                                record.setdefault("unavailable", {})["cupy_cuda"] = str(error)
                                print(f"CuPy CUDA unavailable: {error}")
                                run_cupy = False
                            else:
                                for engine, result in (
                                    ("cupy_cuda_host_input", host_result),
                                    ("cupy_cuda_device_input", device_result),
                                ):
                                    _add_throughput(result, shape.samples)
                                    case = {**case_base, "engine": engine, "result": result}
                                    record["cases"].append(case)
                                    _print_result(case)
                        if run_native_cuda:
                            try:
                                host_result, device_result = benchmark_native_cuda(
                                    batches,
                                    dtype=dtype,
                                    penalty=penalty,
                                    lifecycle=lifecycle,
                                    operation=operation,
                                    warmup=args.warmup,
                                    repeats=args.repeats,
                                    max_iter=args.max_iter,
                                    tol=args.tol,
                                    minimum_sample_seconds=args.minimum_sample_seconds,
                                    max_sample_repetitions=args.max_sample_repetitions,
                                    host_sample_repetitions=planned(
                                        engine="native_cuda_host_input"
                                    ),
                                    device_sample_repetitions=planned(
                                        engine="native_cuda_device_input"
                                    ),
                                )
                            except (BackendUnavailableError, ImportError, OSError) as error:
                                record.setdefault("unavailable", {})["native_cuda"] = str(error)
                                print(f"Native CUDA unavailable: {error}")
                                run_native_cuda = False
                            else:
                                for engine, result in (
                                    ("native_cuda_host_input", host_result),
                                    ("native_cuda_device_input", device_result),
                                ):
                                    _add_throughput(result, shape.samples)
                                    case = {**case_base, "engine": engine, "result": result}
                                    record["cases"].append(case)
                                    _print_result(case)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {len(record['cases'])} measurements to {args.output}")
    return 0
