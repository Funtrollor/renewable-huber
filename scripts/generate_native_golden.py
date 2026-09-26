"""Generate the versioned native-core differential-testing corpora.

``v1`` is frozen: it must stay byte-identical, so ``--check`` is the only
operation anyone should run against it.  ``v2`` adds the L1 cases that
native CUDA ABI 2 is accepted against.  Both are generated from the NumPy
reference, never from a native engine.
"""

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

CORPUS_OUTPUTS = {
    "v1": PROJECT_ROOT / "tests" / "golden" / "native_core_v1.json",
    "v2": PROJECT_ROOT / "tests" / "golden" / "native_core_v2.json",
}


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


# The v2 L1 cases converge at tol=1e-6 in tens of iterations.  Tighter
# tolerances make LAMM stall at max_iter, and a non-converged trajectory is a
# poor cross-engine oracle.  Their float64 tolerances admit one iteration's
# difference in where an engine's reduction order stops the solver.


def _weighted_three_batch_l1_case() -> dict[str, Any]:
    rng = np.random.default_rng(5201)
    X = rng.normal(size=(90, 5))
    y = X @ np.asarray([1.1, 0.0, -0.8, 0.0, 0.35]) + 0.45
    y += rng.normal(scale=0.07, size=X.shape[0])
    y[[11, 52, 77]] += np.asarray([5.0, -4.5, 3.5])
    weights = np.tile(np.asarray([1.0, 0.0, 2.5, 0.5, 1.0, 3.0]), 15)
    probe = np.asarray([[0.0] * 5, [1.0, -0.5, 0.25, 2.0, -1.0], [-1.5, 1.0, 0.0, 0.5, 0.75]])
    return _run_case(
        case_id="weighted_three_batch_l1_f64",
        description="Three weighted L1 batches, including zero and non-unit weights and outliers.",
        config={
            "penalty": "l1",
            "lambda_scale": 0.6,
            "max_iter": 2000,
            "tol": 1e-6,
            "dtype": "float64",
        },
        batches=[
            (X[:25], y[:25], weights[:25]),
            (X[25:58], y[25:58], weights[25:58]),
            (X[58:], y[58:], weights[58:]),
        ],
        probe_X=probe,
        rtol=2e-5,
        atol=2e-6,
    )


def _float32_no_intercept_l1_case() -> dict[str, Any]:
    rng = np.random.default_rng(5202)
    X = rng.normal(size=(96, 4)).astype(np.float32)
    beta = np.asarray([1.3, 0.0, -0.9, 0.0], dtype=np.float32)
    y = (X @ beta + rng.normal(scale=0.05, size=X.shape[0])).astype(np.float32)
    probe = np.asarray([[1.0, 1.0, -1.0, 0.5], [-0.5, 2.0, 0.25, -1.0]], dtype=np.float32)
    return _run_case(
        case_id="multi_batch_l1_no_intercept_f32",
        description="Float32 L1 stream without an intercept whose truth has exact zeros.",
        config={
            "penalty": "l1",
            "lambda_scale": 0.8,
            "fit_intercept": False,
            "max_iter": 2000,
            "tol": 1e-6,
            "dtype": "float32",
        },
        batches=[(X[:40], y[:40], None), (X[40:], y[40:], None)],
        probe_X=probe,
        rtol=3e-4,
        atol=3e-5,
    )


def _sparse_l1_intercept_case() -> dict[str, Any]:
    rng = np.random.default_rng(5203)
    X = rng.normal(size=(70, 8))
    beta = np.zeros(8)
    beta[[0, 5]] = [2.0, -1.25]
    y = X @ beta + 3.0 + rng.normal(scale=0.05, size=X.shape[0])
    probe = np.asarray([[0.0] * 8, [1.0] * 8])
    return _run_case(
        case_id="sparse_l1_intercept_f64",
        description="Sparse L1 truth with a large intercept that must stay unpenalized.",
        config={
            "penalty": "l1",
            "lambda_scale": 1.5,
            "max_iter": 2000,
            "tol": 1e-6,
            "dtype": "float64",
        },
        batches=[(X[:35], y[:35], None), (X[35:], y[35:], None)],
        probe_X=probe,
        rtol=2e-5,
        atol=2e-6,
    )


def _zero_lambda_l1_case() -> dict[str, Any]:
    rng = np.random.default_rng(5204)
    X = rng.normal(size=(60, 3))
    y = X @ np.asarray([0.7, -1.1, 0.3]) - 0.15 + rng.normal(scale=0.1, size=X.shape[0])
    probe = np.asarray([[0.5, 0.5, 0.5], [-1.0, 0.0, 1.0]])
    return _run_case(
        case_id="streaming_l1_zero_lambda_f64",
        description="L1 solver with lambda_scale=0: proximal steps without shrinkage.",
        config={
            "penalty": "l1",
            "lambda_scale": 0.0,
            "max_iter": 2000,
            "tol": 1e-6,
            "dtype": "float64",
        },
        batches=[(X[:20], y[:20], None), (X[20:40], y[20:40], None), (X[40:], y[40:], None)],
        probe_X=probe,
        rtol=2e-5,
        atol=2e-6,
    )


def generate_v2_corpus() -> dict[str, Any]:
    """Return the deterministic L1 corpus native CUDA ABI 2 is accepted against."""

    return {
        "schema": "renewable-huber-native-golden",
        "schema_version": 2,
        "oracle": {
            "implementation": "renewable_huber NumPy backend",
            "role": "Pre-native reference for L1 differential testing",
        },
        "cases": [
            _weighted_three_batch_l1_case(),
            _float32_no_intercept_l1_case(),
            _sparse_l1_intercept_case(),
            _zero_lambda_l1_case(),
        ],
    }


def generate_corpus() -> dict[str, Any]:
    """Return the complete deterministic v1 corpus."""

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
    parser.add_argument("--corpus", choices=sorted(CORPUS_OUTPUTS), default="v1")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is None:
        args.output = CORPUS_OUTPUTS[args.corpus]

    generated = generate_corpus() if args.corpus == "v1" else generate_v2_corpus()
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
