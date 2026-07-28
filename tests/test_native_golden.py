from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any

import numpy as np

from renewable_huber import RenewableHuberRegressor

CORPUS_PATH = Path(__file__).parent / "golden" / "native_core_v1.json"


class NativeGoldenCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))

    def test_schema_and_case_ids_are_stable(self) -> None:
        self.assertEqual(self.corpus["schema"], "renewable-huber-native-golden")
        self.assertEqual(self.corpus["schema_version"], 1)
        self.assertEqual(
            [case["id"] for case in self.corpus["cases"]],
            [
                "weighted_unpenalized_stream_f64",
                "l1_stream_f64",
                "outliers_no_intercept_f32",
                "rank_deficient_lstsq_f64",
            ],
        )

    def test_numpy_oracle_replays_every_case(self) -> None:
        for case in self.corpus["cases"]:
            with self.subTest(case=case["id"]):
                self._replay_case(case)

    def _replay_case(self, case: dict[str, Any]) -> None:
        config = dict(case["config"])
        config["backend"] = "numpy"
        config["device"] = "cpu"
        model = RenewableHuberRegressor(**config)
        dtype = np.dtype(config["dtype"])
        rtol = float(case["rtol"])
        atol = float(case["atol"])

        for batch, expected in zip(case["batches"], case["expected"]["states"], strict=True):
            X = np.asarray(batch["X"], dtype=dtype)
            y = np.asarray(batch["y"], dtype=dtype)
            sample_weight = batch["sample_weight"]
            if sample_weight is not None:
                sample_weight = np.asarray(sample_weight, dtype=dtype)
            model.partial_fit(X, y, sample_weight=sample_weight)
            self._assert_state(model, expected, rtol=rtol, atol=atol)

        probe_X = np.asarray(case["probe_X"], dtype=dtype)
        np.testing.assert_allclose(
            model.predict(probe_X),
            np.asarray(case["expected"]["predictions"], dtype=dtype),
            rtol=rtol,
            atol=atol,
        )

    def _assert_state(
        self,
        model: RenewableHuberRegressor,
        expected: dict[str, Any],
        *,
        rtol: float,
        atol: float,
    ) -> None:
        state = model.state_
        diagnostics = model.diagnostics_
        np.testing.assert_allclose(
            state.coefficients,
            np.asarray(expected["coefficients"], dtype=state.coefficients.dtype),
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            state.information,
            np.asarray(expected["information"], dtype=state.information.dtype),
            rtol=rtol,
            atol=atol,
        )
        self.assertEqual(state.n_samples_seen, expected["n_samples_seen"])
        self.assertEqual(state.batch_count, expected["batch_count"])
        self.assertAlmostEqual(
            state.previous_lambda,
            expected["previous_lambda"],
            delta=atol + rtol * abs(expected["previous_lambda"]),
        )
        self.assertAlmostEqual(
            state.effective_weight,
            expected["weight_sum"],
            delta=atol + rtol * abs(expected["weight_sum"]),
        )

        expected_diagnostics = expected["diagnostics"]
        self.assertEqual(diagnostics.converged, expected_diagnostics["converged"])
        for name in ("objective", "lambda_value", "bandwidth"):
            expected_value = expected_diagnostics[name]
            self.assertAlmostEqual(
                getattr(diagnostics, name),
                expected_value,
                delta=atol + rtol * abs(expected_value),
            )


if __name__ == "__main__":
    unittest.main()
