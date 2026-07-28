"""Generate the versioned native-core differential-testing corpus."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from renewable_huber import RenewableHuberRegressor  # noqa: E402

DEFAULT_OUTPUT = PROJECT_ROOT / "tests" / "golden" / "native_core_v1.json"


def _array(value: Any) -> list[Any]:
    return np.asarray(value).tolist()


def _snapshot(model: RenewableHuberRegressor) -> dict[str, Any]:
    state = model.state_
    diagnostics = model.diagnostics_
    return {
        "coefficients": _array(state.coefficients),
        "information": _array(state.information),
        "n_samples_seen": state.n_samples_seen,
        "batch_count": state.batch_count,
        "previous_lambda": state.previous_lambda,
        "weight_sum": state.effective_weight,
        "diagnostics": {
            "iterations": diagnostics.iterations,
            "converged": diagnostics.converged,
            "objective": diagnostics.objective,
            "lambda_value": diagnostics.lambda_value,
            "bandwidth": diagnostics.bandwidth,
        },
    }


def _run_case(
    *,
    case_id: str,
    description: str,
    config: dict[str, Any],
    batches: list[tuple[np.ndarray, np.ndarray, np.ndarray | None]],
    probe_X: np.ndarray,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    model = RenewableHuberRegressor(backend="numpy", device="cpu", **config)
    batch_records = []
    expected_states = []
    for X, y, sample_weight in batches:
        model.partial_fit(X, y, sample_weight=sample_weight)
        batch_records.append(
            {
                "X": _array(X),
                "y": _array(y),
                "sample_weight": None if sample_weight is None else _array(sample_weight),
            }
        )
        expected_states.append(_snapshot(model))

    return {
        "id": case_id,
        "description": description,
        "rtol": rtol,
        "atol": atol,
        "config": model.get_params(),
        "batches": batch_records,
        "probe_X": _array(probe_X),
        "expected": {
            "states": expected_states,
            "predictions": _array(model.predict(probe_X)),
        },
    }


def _weighted_stream_case() -> dict[str, Any]:
    rng = np.random.default_rng(4101)
    X = rng.normal(size=(64, 4))
    y = X @ np.asarray([1.25, -0.75, 0.4, 0.0]) + 0.3
    y += rng.normal(scale=0.08, size=X.shape[0])
    y[[7, 41]] += np.asarray([6.0, -5.0])
    weights = np.tile(np.asarray([0.0, 0.5, 1.0, 2.0]), 16)
    probe = np.asarray([[0.0, 0.0, 0.0, 0.0], [1.0, -1.0, 0.5, 2.0]])
    return _run_case(
        case_id="weighted_unpenalized_stream_f64",
        description="Two weighted batches with an intercept and response outliers.",
        config={"tau": 1.2, "max_iter": 200, "tol": 1e-10, "dtype": "float64"},
        batches=[
            (X[:29], y[:29], weights[:29]),
            (X[29:], y[29:], weights[29:]),
        ],
        probe_X=probe,
        rtol=2e-9,
        atol=2e-10,
    )


def _l1_stream_case() -> dict[str, Any]:
    rng = np.random.default_rng(4102)
    X = rng.normal(size=(80, 6))
    y = X @ np.asarray([1.4, -0.9, 0.0, 0.45, 0.0, 0.0]) - 0.2
    y += rng.normal(scale=0.06, size=X.shape[0])
    probe = np.asarray(
        [
            [0.5, -0.25, 0.0, 1.0, 0.0, -0.5],
            [-1.0, 0.25, 0.5, 0.0, 1.0, 0.0],
        ]
    )
    return _run_case(
        case_id="l1_stream_f64",
        description="Two L1 batches exercising previous-lambda historical subgradients.",
        config={
            "penalty": "l1",
            "lambda_scale": 0.55,
            "max_iter": 300,
            "tol": 1e-8,
            "dtype": "float64",
        },
        batches=[
            (X[:31], y[:31], None),
            (X[31:], y[31:], None),
        ],
        probe_X=probe,
        rtol=3e-7,
        atol=3e-8,
    )


def _float32_no_intercept_case() -> dict[str, Any]:
    rng = np.random.default_rng(4103)
    X = rng.normal(size=(48, 3)).astype(np.float32)
    y = (X @ np.asarray([0.8, -1.1, 0.35], dtype=np.float32)).astype(np.float32)
    y += rng.normal(scale=0.04, size=X.shape[0]).astype(np.float32)
    y[[5, 33]] += np.asarray([4.0, -3.5], dtype=np.float32)
    probe = np.asarray([[1.0, 0.0, -1.0], [-0.5, 0.25, 0.75]], dtype=np.float32)
    return _run_case(
        case_id="outliers_no_intercept_f32",
        description="Float32 outliers without an intercept.",
        config={
            "tau": 1.1,
            "fit_intercept": False,
            "max_iter": 150,
            "tol": 1e-6,
            "dtype": "float32",
        },
        batches=[(X, y, None)],
        probe_X=probe,
        rtol=3e-4,
        atol=3e-5,
    )


def _rank_deficient_case() -> dict[str, Any]:
    base = np.linspace(-2.0, 2.0, 24)
    X = np.column_stack((base, 2.0 * base, np.ones_like(base)))
    y = 1.75 * base + 0.4
    probe = np.asarray([[0.5, 1.0, 1.0], [-1.5, -3.0, 1.0]])
    return _run_case(
        case_id="rank_deficient_lstsq_f64",
        description="Rank-deficient quadratic-region fit exercising least-squares fallback.",
        config={"tau": 100.0, "ridge": 0.0, "max_iter": 40, "tol": 1e-11, "dtype": "float64"},
        batches=[(X, y, None)],
        probe_X=probe,
        rtol=2e-8,
        atol=2e-9,
    )


def generate_corpus() -> dict[str, Any]:
    """Return the complete deterministic corpus."""

    return {
        "schema": "renewable-huber-native-golden",
        "schema_version": 1,
        "oracle": {
            "implementation": "renewable_huber NumPy backend",
            "role": "Pre-native reference for differential testing",
        },
        "cases": [
            _weighted_stream_case(),
            _l1_stream_case(),
            _float32_no_intercept_case(),
            _rank_deficient_case(),
        ],
    }


def _assert_equivalent(expected: Any, actual: Any, path: str = "$") -> None:
    if isinstance(expected, dict) and isinstance(actual, dict):
        if expected.keys() != actual.keys():
            raise AssertionError(f"{path}: object keys differ")
        for key in expected:
            _assert_equivalent(expected[key], actual[key], f"{path}.{key}")
        return
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            raise AssertionError(f"{path}: list lengths differ")
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual, strict=True)):
            _assert_equivalent(expected_item, actual_item, f"{path}[{index}]")
        return
    if (
        isinstance(expected, (int, float))
        and not isinstance(expected, bool)
        and isinstance(actual, (int, float))
        and not isinstance(actual, bool)
    ):
        if not math.isclose(float(expected), float(actual), rel_tol=5e-5, abs_tol=5e-6):
            raise AssertionError(f"{path}: expected {expected!r}, generated {actual!r}")
        return
    if expected != actual:
        raise AssertionError(f"{path}: expected {expected!r}, generated {actual!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true", help="Write the generated corpus")
    action.add_argument("--check", action="store_true", help="Compare with the committed corpus")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    generated = generate_corpus()
    if args.write:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(generated, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {len(generated['cases'])} cases to {args.output}")
        return 0

    if not args.output.is_file():
        parser.error(f"committed corpus does not exist: {args.output}")
    committed = json.loads(args.output.read_text(encoding="utf-8"))
    try:
        _assert_equivalent(committed, generated)
    except AssertionError as error:
        print(f"Golden corpus check failed: {error}", file=sys.stderr)
        return 1
    print(f"Golden corpus matches {len(generated['cases'])} generated cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
