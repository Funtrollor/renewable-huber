"""Compare ``backend="auto"`` against both explicit CPU backends.

The question this answers is not "is the native engine fast here" -- the shape
sweep already answers that -- but "does the runtime policy pick the faster of
the two, and what does asking cost".  Those are separate numbers, so every case
reports three engines and two auto lifecycles:

``numpy``            explicit ``backend="numpy"``
``native_cpu``       explicit ``backend="native_cpu"``
``auto_cold``        ``backend="auto"`` with the process dispatch cache cleared
                     first, so each sample pays for a full probe ladder
``auto_warm``        ``backend="auto"`` with the cache already populated, which
                     is what every call after the first one in a process sees

The policy's own calibration timer is the isolated calibration cost.
``auto_warm`` against the better of the two explicit engines is the
steady-state quality of the decision. ``regret`` states that directly: below
1.0 the policy beat the engine a caller would have had by default, above 1.0
it cost something. Cold samples are measured in a separate phase so repeatedly
calibrating cannot heat the CPU immediately before a steady sample.

The two operations follow the shape sweep's contract, so a case here and a case
there with the same shape describe the same work: ``fit`` is one call over the
**whole** dataset, concatenated outside every timed region by the sweep's own
``_fit_batch``, and ``stream`` is one ``partial_fit`` per batch.  That
distinction is also what the dispatch policy sees, so ``summary.work_units``
counts all the samples for ``fit`` and only the first batch for ``stream`` --
the batch the decision is actually made on.

The record is deliberately a different schema from the shape sweep so that
``performance_policy.validate_record`` can never be handed one by mistake.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _entry in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from renewable_huber import RenewableHuberRegressor  # noqa: E402
from renewable_huber.backends.cpu_dispatch import (  # noqa: E402
    DEFAULT_POLICY,
    current_runtime_signature,
    reset_cpu_dispatch_cache,
)
from scripts.benchmarks.shape_sweep.environment import environment_metadata  # noqa: E402
from scripts.benchmarks.shape_sweep.shapes import (  # noqa: E402
    PROFILES,
    Shape,
    _dataset_checksum,
    make_batches,
)
from scripts.benchmarks.shape_sweep.timing import _fit_batch  # noqa: E402

SCHEMA = "renewable-huber-auto-dispatch"
SCHEMA_VERSION = 1

#: ``auto_cold`` and ``auto_warm`` are the same estimator configuration; they
#: differ only in whether the process has already measured this host.
ENGINES = ("numpy", "native_cpu", "auto_cold", "auto_warm")
STEADY_ENGINES = ("numpy", "native_cpu", "auto_warm")


def _engine_order(round_index: int) -> tuple[str, ...]:
    """Alternate forward/reverse order so thermal position is balanced."""

    return STEADY_ENGINES if round_index % 2 == 0 else tuple(reversed(STEADY_ENGINES))


@dataclass(frozen=True, slots=True)
class Case:
    shape: Shape
    profile: str
    dtype: str
    penalty: str
    operation: str


def _configured_backend(engine: str) -> str:
    return "auto" if engine.startswith("auto") else engine


def _work_units(case: Case) -> float:
    """Batch work as the dispatch policy counts it, for the deciding batch.

    The width is the *design* width, ``features + 1``, because the estimator
    fits an intercept and the policy is asked about the prepared design matrix.
    The row count is where the two operations part: ``fit`` hands the whole
    dataset over in one call, while a stream decides on its first batch and
    never revisits, so it is judged on ``batch_size`` rows -- or on the whole
    dataset when that is shorter than one batch.
    """

    rows = (
        case.shape.samples
        if case.operation == "fit"
        else min(case.shape.batch_size, case.shape.samples)
    )
    return float(rows) * float(case.shape.features + 1) ** 2


def _one_run(
    engine: str,
    case: Case,
    batches: list[tuple[np.ndarray, np.ndarray]],
    fit_batch: tuple[np.ndarray, np.ndarray],
) -> tuple[float, RenewableHuberRegressor]:
    """Time one complete lifecycle: build an estimator, then consume data.

    The estimator is constructed inside the timed region on purpose. A one-shot
    caller pays construction, dispatch and the solve together, and separating
    them here would hide exactly the cost this tool exists to expose.

    ``fit_batch`` is the concatenation of ``batches``, built by the caller
    before any clock starts, so a ``fit`` case measures a single call over the
    whole dataset the header and the shape record describe -- not its first
    batch -- without paying for the concatenation. ``stream`` keeps the batches
    separate, because per-batch updates are the workload it exists to measure.
    """

    started = perf_counter()
    model = RenewableHuberRegressor(
        backend=_configured_backend(engine),  # type: ignore[arg-type]
        dtype=case.dtype,  # type: ignore[arg-type]
        penalty=case.penalty,  # type: ignore[arg-type]
    )
    if case.operation == "fit":
        model.fit(*fit_batch)
    else:
        for features, target in batches:
            model.partial_fit(features, target)
    return perf_counter() - started, model


def _sample(
    engine: str,
    case: Case,
    batches: list[tuple[np.ndarray, np.ndarray]],
    fit_batch: tuple[np.ndarray, np.ndarray],
    *,
    warmup: int,
) -> tuple[float, RenewableHuberRegressor]:
    """Take one timed sample behind ``warmup`` untimed runs of the same engine.

    Every engine gets the same treatment, which is the point. Measuring four
    engines back to back in one process without it compares the first one
    against cold pages and an unspun BLAS pool and the last against a fully
    warm machine; an early version of this script reported the *same* NumPy
    work as 335 ms under one label and 69 ms under another for exactly that
    reason.

    ``auto_cold`` is the one asymmetry: its warmup necessarily fills the
    dispatch cache, so the cache is cleared again between the warmup and the
    timed run. The timed run therefore pays a full probe ladder on a warm
    machine, which is the isolated calibration cost and nothing else.
    """

    for _ in range(warmup):
        _one_run(engine, case, batches, fit_batch)
    if engine == "auto_cold":
        reset_cpu_dispatch_cache()
    return _one_run(engine, case, batches, fit_batch)


def _measure(
    case: Case,
    batches: list[tuple[np.ndarray, np.ndarray]],
    fit_batch: tuple[np.ndarray, np.ndarray],
    *,
    warmup: int,
    repeats: int,
) -> dict[str, dict[str, Any]]:
    """Interleave every engine across rounds and return one entry per engine.

    The steady engines run first, alternating forward and reverse order by
    round. ``auto_cold`` runs only after that phase: otherwise every cold
    calibration perturbs the immediately following warm measurement and the
    benchmark attributes calibration heat to dispatch overhead.
    """

    if repeats < 2 or repeats % 2:
        raise ValueError("repeats must be a positive even number for balanced A/B ordering")

    samples: dict[str, list[float]] = {engine: [] for engine in ENGINES}
    selected: dict[str, str | None] = {engine: None for engine in ENGINES}
    dispatch: dict[str, dict[str, Any] | None] = {engine: None for engine in ENGINES}
    skipped: dict[str, str] = {}

    # Prime the host model outside every steady timed region. Small cases still
    # take the documented no-calibration NumPy path.
    reset_cpu_dispatch_cache()
    try:
        _one_run("auto_warm", case, batches, fit_batch)
    except Exception as error:
        skipped["auto_warm"] = f"{type(error).__name__}: {error}"

    for round_index in range(repeats):
        for engine in _engine_order(round_index):
            if engine in skipped:
                continue
            try:
                seconds, model = _sample(engine, case, batches, fit_batch, warmup=warmup)
            except Exception as error:  # an unavailable engine is data, not a crash
                skipped[engine] = f"{type(error).__name__}: {error}"
                continue
            samples[engine].append(seconds)
            selected[engine] = model.backend_
            record = getattr(model, "auto_dispatch_", None)
            if record is not None:
                dispatch[engine] = dict(record)

    # Cold lifecycle is evidence about end-to-end first use, but deliberately
    # not interleaved with the steady comparison above.
    for _ in range(repeats):
        try:
            seconds, model = _sample("auto_cold", case, batches, fit_batch, warmup=warmup)
        except Exception as error:
            skipped["auto_cold"] = f"{type(error).__name__}: {error}"
            break
        samples["auto_cold"].append(seconds)
        selected["auto_cold"] = model.backend_
        record = getattr(model, "auto_dispatch_", None)
        if record is not None:
            dispatch["auto_cold"] = dict(record)

    results: dict[str, dict[str, Any]] = {}
    for engine in ENGINES:
        if engine in skipped or not samples[engine]:
            results[engine] = {
                "engine": engine,
                "skipped": skipped.get(engine, "no sample completed"),
            }
            continue
        results[engine] = {
            "engine": engine,
            "seconds": samples[engine],
            "median_seconds": statistics.median(samples[engine]),
            "minimum_seconds": min(samples[engine]),
            "selected_backend": selected[engine],
            "auto_dispatch": dispatch[engine],
        }
    return results


def _cases(
    profiles: tuple[str, ...],
    dtypes: tuple[str, ...],
    penalties: tuple[str, ...],
    operations: tuple[str, ...],
) -> list[Case]:
    cases = []
    for profile in profiles:
        for shape in PROFILES[profile]:
            for dtype in dtypes:
                for penalty in penalties:
                    for operation in operations:
                        cases.append(Case(shape, profile, dtype, penalty, operation))
    return cases


def _summarise(results: dict[str, dict[str, Any]], case: Case) -> dict[str, Any]:
    """Derive the two numbers a reviewer actually reads."""

    summary: dict[str, Any] = {
        "work_units": _work_units(case),
        "calibration_seconds": None,
        "calibration_overhead_seconds": None,
        "regret": None,
        "best_explicit": None,
    }
    warm = results.get("auto_warm", {})
    cold = results.get("auto_cold", {})
    dispatch = warm.get("auto_dispatch") or cold.get("auto_dispatch")
    if dispatch is not None:
        summary["calibration_seconds"] = dispatch.get("calibration_seconds")
        summary["dispatch_reason"] = dispatch.get("reason")
        summary["predicted_ratio"] = dispatch.get("predicted_ratio")
    if dispatch is not None and dispatch.get("calibrated"):
        summary["calibration_overhead_seconds"] = dispatch.get("calibration_seconds")
    explicit = {
        name: results[name]
        for name in ("numpy", "native_cpu")
        if "seconds" in results.get(name, {})
    }
    if explicit and "seconds" in warm:
        best = min(explicit, key=lambda name: explicit[name]["median_seconds"])
        summary["best_explicit"] = best
        sample_count = len(warm["seconds"])
        if all(len(result["seconds"]) == sample_count for result in explicit.values()):
            summary["regret"] = statistics.median(
                warm["seconds"][index]
                / min(result["seconds"][index] for result in explicit.values())
                for index in range(sample_count)
            )
            summary["regret_statistic"] = "median aligned ratio to per-round best explicit"
    return summary


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    cases = _cases(
        arguments.profiles,
        arguments.dtypes,
        arguments.penalties,
        arguments.operations,
    )
    emitted: list[dict[str, Any]] = []
    for case in cases:
        batches = make_batches(case.shape, seed=arguments.seed, dtype=case.dtype)
        # Concatenated here, before any clock starts, and reused by every
        # engine and every repeat: a `fit` case must measure one call over the
        # whole dataset, not the cost of assembling it.
        fit_batch = _fit_batch(batches, xp=np)
        results = _measure(
            case,
            batches,
            fit_batch,
            warmup=arguments.warmup,
            repeats=arguments.repeats,
        )
        record = {
            "shape": {
                "name": case.shape.name,
                "profile": case.profile,
                "samples": case.shape.samples,
                "features": case.shape.features,
                "batch_size": case.shape.batch_size,
            },
            "dtype": case.dtype,
            "penalty": case.penalty,
            "operation": case.operation,
            "dataset_sha256": _dataset_checksum(batches),
            "engines": results,
            "summary": _summarise(results, case),
        }
        emitted.append(record)
        _print_case(record)
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "arguments": {
            "profiles": list(arguments.profiles),
            "dtypes": list(arguments.dtypes),
            "penalties": list(arguments.penalties),
            "operations": list(arguments.operations),
            "warmup": arguments.warmup,
            "repeats": arguments.repeats,
            "seed": arguments.seed,
        },
        # The runtime signature is recorded because a record captured under a
        # different affinity mask or thread environment is not comparable with
        # this one.
        "runtime_signature": asdict(current_runtime_signature()),
        "policy": {
            "minimum_work_units": DEFAULT_POLICY.minimum_work_units,
            "calibration_work_units": DEFAULT_POLICY.calibration_work_units,
            "ladder_work_units": DEFAULT_POLICY.ladder_work_units(),
            "enter_margin": DEFAULT_POLICY.enter_margin,
            "uncertainty_multiplier": DEFAULT_POLICY.uncertainty_multiplier,
            "probe_start_deadline_seconds": DEFAULT_POLICY.probe_start_deadline_seconds,
            "probes": [
                {"samples": probe.samples, "parameters": probe.parameters}
                for probe in DEFAULT_POLICY.probe_ladder
            ],
        },
        "environment": environment_metadata(),
        "cases": emitted,
    }


def _print_case(record: dict[str, Any]) -> None:
    shape = record["shape"]
    header = (
        f"{shape['name']:<16} {shape['samples']}x{shape['features']} "
        f"{record['dtype']} {record['penalty']} {record['operation']}"
    )
    print(header)
    for engine in ENGINES:
        result = record["engines"][engine]
        if "skipped" in result:
            print(f"  {engine:<12} skipped: {result['skipped']}")
            continue
        print(
            f"  {engine:<12} median {result['median_seconds'] * 1e3:9.3f} ms"
            f"   selected {result['selected_backend']}"
        )
    summary = record["summary"]
    if summary["regret"] is not None:
        print(
            f"  -> best explicit {summary['best_explicit']}, regret "
            f"{summary['regret']:.3f}x, calibration "
            f"{(summary['calibration_seconds'] or 0.0) * 1e3:.1f} ms"
        )
    if summary.get("dispatch_reason"):
        print(f"  -> {summary['dispatch_reason']}")


def _plural(values: str, allowed: tuple[str, ...]) -> tuple[str, ...]:
    return allowed if values == "both" or values == "all" else (values,)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--profile", choices=("smoke", "standard", "both"), default="smoke")
    parser.add_argument("--dtype", choices=("float32", "float64", "both"), default="float64")
    parser.add_argument("--penalty", choices=("none", "l1", "both"), default="none")
    parser.add_argument("--operation", choices=("fit", "stream", "both"), default="fit")
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="untimed runs of the same engine before every timed sample (default: 1)",
    )
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20_260_809)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)

    if arguments.repeats < 2 or arguments.repeats % 2:
        parser.error("--repeats must be a positive even number for balanced A/B ordering")
    if arguments.warmup < 0:
        parser.error("--warmup must not be negative")

    arguments.profiles = _plural(arguments.profile, ("smoke", "standard"))
    arguments.dtypes = _plural(arguments.dtype, ("float32", "float64"))
    arguments.penalties = _plural(arguments.penalty, ("none", "l1"))
    arguments.operations = _plural(arguments.operation, ("fit", "stream"))

    record = run(arguments)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
