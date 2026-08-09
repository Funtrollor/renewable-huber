"""Contract for the checkpoint payload boundary between estimator and codec.

The point of the boundary is that ``renewable_huber.serialization`` encodes and
decodes data, while every rule about *when* an estimator may grow a fitted
attribute stays in the estimator layer. A violation of that split does not
break any numerical result, so the separation is asserted structurally here.
"""

from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np

from renewable_huber import RenewableHuberRegressor, ValidationError, serialization
from renewable_huber.core import UpdateDiagnostics
from renewable_huber.serialization import (
    FORMAT_VERSION,
    SUPPORTED_FORMAT_VERSIONS,
    CheckpointPayload,
    read_checkpoint,
    write_checkpoint,
)
from renewable_huber.state import RenewableHuberState


def _fitted_model(**settings: Any) -> tuple[RenewableHuberRegressor, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260809)
    X = rng.normal(size=(180, 4))
    y = X @ np.asarray([1.4, -0.6, 0.0, 0.9]) + 0.35 + rng.normal(scale=0.05, size=180)
    return RenewableHuberRegressor(**settings).fit(X, y), X, y


def _write_archive(path: Path, *, coefficients: Any, information: Any, metadata: Any) -> None:
    """Write a checkpoint archive directly, bypassing the encoder."""

    np.savez_compressed(
        path,
        coefficients=coefficients,
        information=information,
        metadata=np.asarray(json.dumps(metadata)),
    )


def _legacy_metadata(model: RenewableHuberRegressor, *, format_version: int) -> dict[str, Any]:
    payload = model.state_dict()
    metadata = {
        "format_version": format_version,
        "config": payload["config"],
        "n_samples_seen": payload["n_samples_seen"],
        "batch_count": payload["batch_count"],
        "previous_lambda": payload["previous_lambda"],
        "n_features_in": payload["n_features_in"],
        "fit_intercept": payload["fit_intercept"],
    }
    if format_version >= 2:
        metadata["weight_sum"] = payload["weight_sum"]
        metadata["feature_names_in"] = payload["feature_names_in"]
    return metadata


class SerializationBoundaryTests(unittest.TestCase):
    """``serialization.py`` must not depend on the estimator's private lifecycle.

    A ``sys.modules`` check cannot express this: importing any submodule runs
    ``renewable_huber/__init__.py``, which imports the estimator for its own
    public re-export. The dependency is therefore asserted against the module's
    source and its own namespace.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.path = Path(serialization.__file__)
        cls.tree = ast.parse(cls.path.read_text(encoding="utf-8"))

    def test_the_parsed_module_is_the_one_under_test(self) -> None:
        # Anti-vacuity only: without it every check below would also pass
        # against an empty file. The codec is free to grow private helpers, so
        # this asserts that the three names exist and are the right kind of
        # object, not that they are the only things defined.
        defined = {
            node.name: node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef | ast.ClassDef)
        }
        self.assertLessEqual(
            {"CheckpointPayload", "write_checkpoint", "read_checkpoint"}, set(defined)
        )
        self.assertIsInstance(defined["CheckpointPayload"], ast.ClassDef)
        self.assertIsInstance(defined["write_checkpoint"], ast.FunctionDef)
        self.assertIsInstance(defined["read_checkpoint"], ast.FunctionDef)

    def test_a_new_private_helper_does_not_break_the_boundary_checks(self) -> None:
        # The lock above used to be an equality assertion, which turned any
        # added helper into a failure. Prove the relaxed form still holds when
        # the module gains one.
        grown = ast.parse(
            self.path.read_text(encoding="utf-8") + "\n\ndef _pack_metadata() -> None:\n    pass\n"
        )
        defined = {
            node.name for node in grown.body if isinstance(node, ast.FunctionDef | ast.ClassDef)
        }
        self.assertIn("_pack_metadata", defined)
        self.assertLessEqual({"CheckpointPayload", "write_checkpoint", "read_checkpoint"}, defined)

    def test_nothing_imports_the_estimator_module(self) -> None:
        imported: set[str] = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
                imported.update(alias.name for alias in node.names)
        self.assertTrue(imported, "the module imports nothing at all")
        self.assertEqual([name for name in imported if "estimator" in name], [])

    def test_no_reference_to_estimator_lifecycle_members(self) -> None:
        referenced: set[str] = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Name):
                referenced.add(node.id)
            elif isinstance(node, ast.Attribute):
                referenced.add(node.attr)
        # Docstrings are ``Constant`` nodes, so naming these in prose is safe.
        forbidden = {
            "RenewableHuberRegressor",
            "_restore_state",
            "_sync_public_coefficients",
            "state_dict",
            "_backend",
        }
        self.assertEqual(sorted(referenced & forbidden), [])

    def test_the_forbidden_names_still_exist_in_the_estimator(self) -> None:
        # Guards against the check above passing because a member was renamed.
        self.assertTrue(hasattr(RenewableHuberRegressor, "_restore_state"))
        self.assertTrue(hasattr(RenewableHuberRegressor, "_sync_public_coefficients"))
        self.assertTrue(hasattr(RenewableHuberRegressor, "state_dict"))

    def test_the_module_namespace_holds_no_estimator_class(self) -> None:
        self.assertNotIn("RenewableHuberRegressor", vars(serialization))
        for name, value in vars(serialization).items():
            if isinstance(value, type):
                self.assertNotEqual(value, RenewableHuberRegressor, name)


class CheckpointFormatTests(unittest.TestCase):
    def test_the_written_format_version_is_still_two(self) -> None:
        # A structural refactor must not bump the on-disk format.
        self.assertEqual(FORMAT_VERSION, 2)
        self.assertEqual(SUPPORTED_FORMAT_VERSIONS, (1, 2))
        model, _, _ = _fitted_model()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "version.npz"
            model.save(checkpoint)
            with np.load(checkpoint, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata"].item()))
                stored = sorted(archive.files)
        self.assertEqual(metadata["format_version"], 2)
        self.assertEqual(stored, ["coefficients", "information", "metadata"])

    def test_codec_round_trips_a_payload_without_any_estimator(self) -> None:
        state = RenewableHuberState(
            coefficients=np.asarray([1.5, -2.5, 0.25]),
            information=np.eye(3) * 3.0,
            n_samples_seen=41,
            batch_count=3,
            previous_lambda=0.75,
            n_features_in=2,
            fit_intercept=True,
            weight_sum=52.5,
        )
        payload = CheckpointPayload(
            config={"tau": 1.345},
            state=state,
            feature_names=["b", "a"],
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = write_checkpoint(payload, Path(directory) / "payload.npz")
            decoded = read_checkpoint(checkpoint)

        self.assertEqual(decoded.format_version, FORMAT_VERSION)
        self.assertEqual(decoded.config, {"tau": 1.345})
        self.assertEqual(decoded.feature_names, ["b", "a"])
        np.testing.assert_array_equal(decoded.state.coefficients, state.coefficients)
        np.testing.assert_array_equal(decoded.state.information, state.information)
        self.assertEqual(decoded.state.n_samples_seen, 41)
        self.assertEqual(decoded.state.batch_count, 3)
        self.assertEqual(decoded.state.previous_lambda, 0.75)
        self.assertEqual(decoded.state.effective_weight, 52.5)

    def test_decoded_arrays_are_float64_whatever_the_producing_dtype_was(self) -> None:
        model, _, _ = _fitted_model(dtype="float32")
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "float32.npz"
            model.save(checkpoint)
            with np.load(checkpoint, allow_pickle=False) as archive:
                self.assertEqual(archive["coefficients"].dtype, np.dtype("float32"))
            decoded = read_checkpoint(checkpoint)
        self.assertEqual(decoded.state.coefficients.dtype, np.dtype("float64"))
        self.assertEqual(decoded.state.information.dtype, np.dtype("float64"))

    def test_version_one_falls_back_to_unit_weights(self) -> None:
        model, X, _ = _fitted_model()
        payload = model.state_dict()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "legacy-v1.npz"
            _write_archive(
                checkpoint,
                coefficients=payload["coefficients"],
                information=payload["information"],
                metadata=_legacy_metadata(model, format_version=1),
            )
            decoded = read_checkpoint(checkpoint)
            restored = RenewableHuberRegressor.load(checkpoint)

        self.assertEqual(decoded.format_version, 1)
        self.assertEqual(decoded.state.effective_weight, float(decoded.state.n_samples_seen))
        self.assertIsNone(decoded.feature_names)
        np.testing.assert_allclose(restored.predict(X), model.predict(X))

    def test_an_unsupported_format_version_is_named_as_such(self) -> None:
        model, _, _ = _fitted_model()
        payload = model.state_dict()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "future.npz"
            metadata = _legacy_metadata(model, format_version=99)
            _write_archive(
                checkpoint,
                coefficients=payload["coefficients"],
                information=payload["information"],
                metadata=metadata,
            )
            with self.assertRaisesRegex(ValidationError, "Unsupported"):
                read_checkpoint(checkpoint)

    def test_writing_refuses_a_payload_that_is_not_the_current_format(self) -> None:
        payload = CheckpointPayload(
            format_version=1,
            config={},
            state=RenewableHuberState.empty(2, fit_intercept=True, xp=np, dtype=np.float64),
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValidationError, "Unsupported"):
                write_checkpoint(payload, Path(directory) / "downgrade.npz")

    def test_a_missing_file_is_not_disguised_as_a_validation_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                read_checkpoint(Path(directory) / "absent.npz")

    def test_corrupted_archives_and_metadata_are_rejected(self) -> None:
        model, _, _ = _fitted_model()
        payload = model.state_dict()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            not_an_archive = root / "garbage.npz"
            not_an_archive.write_bytes(b"PK\x03\x04 definitely not an npz")
            with self.assertRaisesRegex(ValidationError, "Invalid or corrupted"):
                read_checkpoint(not_an_archive)

            truncated = root / "truncated.npz"
            model.save(truncated)
            content = truncated.read_bytes()
            truncated.write_bytes(content[: len(content) // 2])
            with self.assertRaisesRegex(ValidationError, "Invalid or corrupted"):
                read_checkpoint(truncated)

            bad_json = root / "bad-json.npz"
            np.savez_compressed(
                bad_json,
                coefficients=payload["coefficients"],
                information=payload["information"],
                metadata=np.asarray("{not json"),
            )
            with self.assertRaisesRegex(ValidationError, "Invalid or corrupted"):
                read_checkpoint(bad_json)

            missing_key = root / "missing-key.npz"
            metadata = _legacy_metadata(model, format_version=2)
            del metadata["n_features_in"]
            _write_archive(
                missing_key,
                coefficients=payload["coefficients"],
                information=payload["information"],
                metadata=metadata,
            )
            with self.assertRaisesRegex(ValidationError, "Invalid or corrupted"):
                read_checkpoint(missing_key)

    def test_pickled_content_stays_unreadable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "pickled.npz"
            np.savez_compressed(
                checkpoint,
                coefficients=np.asarray([{"arbitrary": "object"}], dtype=object),
                information=np.eye(2),
                metadata=np.asarray(json.dumps({"format_version": 2})),
            )
            with self.assertRaises(ValidationError):
                read_checkpoint(checkpoint)

    def test_a_configuration_that_is_not_a_mapping_has_its_own_message(self) -> None:
        model, _, _ = _fitted_model()
        payload = model.state_dict()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "bad-config.npz"
            metadata = _legacy_metadata(model, format_version=2)
            metadata["config"] = ["not", "a", "mapping"]
            _write_archive(
                checkpoint,
                coefficients=payload["coefficients"],
                information=payload["information"],
                metadata=metadata,
            )
            with self.assertRaisesRegex(ValidationError, "checkpoint configuration"):
                read_checkpoint(checkpoint)

    def test_an_unknown_configuration_key_is_reported_as_a_configuration_error(self) -> None:
        model, _, _ = _fitted_model()
        payload = model.state_dict()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "unknown-config.npz"
            metadata = _legacy_metadata(model, format_version=2)
            metadata["config"] = {**metadata["config"], "not_a_parameter": 1}
            _write_archive(
                checkpoint,
                coefficients=payload["coefficients"],
                information=payload["information"],
                metadata=metadata,
            )
            # Decoding succeeds; only construction can know the keyword is wrong.
            self.assertIn("not_a_parameter", read_checkpoint(checkpoint).config)
            with self.assertRaisesRegex(ValidationError, "checkpoint configuration"):
                RenewableHuberRegressor.load(checkpoint)


class CheckpointStateValidationTests(unittest.TestCase):
    """State invariants are enforced when the payload reaches the estimator."""

    def _load_mutated(self, mutate: Any, **overrides: Any) -> RenewableHuberRegressor:
        model, _, _ = _fitted_model()
        payload = model.state_dict()
        metadata = _legacy_metadata(model, format_version=2)
        metadata.update(overrides)
        coefficients = np.asarray(payload["coefficients"], dtype=np.float64).copy()
        information = np.asarray(payload["information"], dtype=np.float64).copy()
        coefficients, information, metadata = mutate(coefficients, information, metadata)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "mutated.npz"
            _write_archive(
                checkpoint,
                coefficients=coefficients,
                information=information,
                metadata=metadata,
            )
            return RenewableHuberRegressor.load(checkpoint)

    def test_non_finite_coefficients_are_rejected(self) -> None:
        def mutate(coefficients: Any, information: Any, metadata: Any) -> Any:
            coefficients[0] = np.nan
            return coefficients, information, metadata

        with self.assertRaisesRegex(ValidationError, "coefficients must contain only finite"):
            self._load_mutated(mutate)

    def test_non_finite_information_is_rejected(self) -> None:
        def mutate(coefficients: Any, information: Any, metadata: Any) -> Any:
            information[0, 0] = np.inf
            return coefficients, information, metadata

        with self.assertRaisesRegex(ValidationError, "information must contain only finite"):
            self._load_mutated(mutate)

    def test_a_shape_inconsistent_with_the_metadata_is_rejected(self) -> None:
        def mutate(coefficients: Any, information: Any, metadata: Any) -> Any:
            metadata["n_features_in"] = int(metadata["n_features_in"]) + 1
            return coefficients, information, metadata

        with self.assertRaisesRegex(ValidationError, "coefficient shape does not match"):
            self._load_mutated(mutate)

    def test_a_fit_intercept_disagreement_is_rejected(self) -> None:
        def mutate(coefficients: Any, information: Any, metadata: Any) -> Any:
            metadata["config"] = {**metadata["config"], "fit_intercept": False}
            return coefficients, information, metadata

        with self.assertRaisesRegex(ValidationError, "fit_intercept does not match"):
            self._load_mutated(mutate)

    def test_a_negative_counter_is_rejected(self) -> None:
        def mutate(coefficients: Any, information: Any, metadata: Any) -> Any:
            metadata["batch_count"] = -1
            return coefficients, information, metadata

        with self.assertRaisesRegex(ValidationError, "counters must be non-negative"):
            self._load_mutated(mutate)

    def test_feature_names_of_the_wrong_length_are_rejected(self) -> None:
        def mutate(coefficients: Any, information: Any, metadata: Any) -> Any:
            metadata["feature_names_in"] = ["only_one"]
            return coefficients, information, metadata

        with self.assertRaisesRegex(ValidationError, "feature names do not match"):
            self._load_mutated(mutate)

    def test_non_string_feature_names_are_rejected(self) -> None:
        def mutate(coefficients: Any, information: Any, metadata: Any) -> Any:
            metadata["feature_names_in"] = [0, 1, 2, 3]
            return coefficients, information, metadata

        with self.assertRaisesRegex(ValidationError, "feature names do not match"):
            self._load_mutated(mutate)


class CheckpointDiagnosticsTests(unittest.TestCase):
    """No released format stores diagnostics, and none may be invented."""

    def test_decoding_any_supported_version_yields_no_diagnostics(self) -> None:
        model, _, _ = _fitted_model()
        payload = model.state_dict()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current = root / "current.npz"
            model.save(current)
            self.assertIsNone(read_checkpoint(current).diagnostics)

            legacy = root / "legacy.npz"
            _write_archive(
                legacy,
                coefficients=payload["coefficients"],
                information=payload["information"],
                metadata=_legacy_metadata(model, format_version=1),
            )
            self.assertIsNone(read_checkpoint(legacy).diagnostics)

    def test_a_loaded_model_reports_no_batch_rather_than_a_fabricated_one(self) -> None:
        model, _, _ = _fitted_model()
        self.assertTrue(model.diagnostics_.converged)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "no-diagnostics.npz"
            model.save(checkpoint)
            restored = RenewableHuberRegressor.load(checkpoint)
        with self.assertRaises(Exception) as caught:
            restored.diagnostics_
        self.assertEqual(type(caught.exception).__name__, "NotFittedError")

    def test_an_in_memory_payload_carries_diagnostics_through_restoration(self) -> None:
        # The field is part of the boundary type even though the codec never
        # produces it, so a caller assembling a payload is not silently ignored.
        model, _, _ = _fitted_model()
        payload = model._checkpoint_payload()
        diagnostics = UpdateDiagnostics(
            iterations=7,
            converged=True,
            objective=1.25,
            lambda_value=0.0,
            bandwidth=0.5,
        )
        carried = CheckpointPayload(
            format_version=payload.format_version,
            config=payload.config,
            state=payload.state,
            feature_names=payload.feature_names,
            diagnostics=diagnostics,
        )
        restored = RenewableHuberRegressor._from_checkpoint_payload(carried)
        self.assertEqual(restored.diagnostics_, diagnostics)


class CheckpointRestorationTests(unittest.TestCase):
    def test_feature_name_order_survives_a_round_trip(self) -> None:
        class _Frame:
            def __init__(self, values: np.ndarray, columns: list[str]) -> None:
                self._values = values
                self.columns = columns

            def to_numpy(self) -> np.ndarray:
                return self._values

        rng = np.random.default_rng(7)
        X = rng.normal(size=(90, 3))
        y = X @ np.asarray([0.5, -1.0, 2.0])
        names = ["zeta", "alpha", "mid"]
        model = RenewableHuberRegressor().fit(_Frame(X, names), y)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "named.npz"
            model.save(checkpoint)
            self.assertEqual(read_checkpoint(checkpoint).feature_names, names)
            restored = RenewableHuberRegressor.load(checkpoint)
        np.testing.assert_array_equal(restored.feature_names_in_, names)
        with self.assertRaisesRegex(ValidationError, "feature names must match"):
            restored.predict(_Frame(X, ["alpha", "mid", "zeta"]))

    def test_weighted_state_survives_a_round_trip(self) -> None:
        rng = np.random.default_rng(11)
        X = rng.normal(size=(150, 3))
        y = X @ np.asarray([1.0, 0.5, -0.5])
        weights = np.linspace(0.25, 2.75, X.shape[0])
        model = RenewableHuberRegressor().fit(X, y, sample_weight=weights)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "weighted.npz"
            model.save(checkpoint)
            decoded = read_checkpoint(checkpoint)
            restored = RenewableHuberRegressor.load(checkpoint)
        self.assertAlmostEqual(decoded.state.effective_weight, float(weights.sum()))
        self.assertEqual(restored.state_.effective_weight, model.state_.effective_weight)
        self.assertNotEqual(restored.state_.effective_weight, restored.state_.n_samples_seen)

    def test_l1_previous_lambda_survives_a_round_trip(self) -> None:
        rng = np.random.default_rng(915)
        X = rng.normal(size=(160, 5))
        y = X @ np.asarray([1.5, -0.8, 0.0, 0.4, 0.0]) + 0.25
        model = RenewableHuberRegressor(penalty="l1", lambda_scale=0.6, max_iter=250)
        model.partial_fit(X[:70], y[:70])
        previous_lambda = model.state_.previous_lambda
        self.assertGreater(previous_lambda, 0.0)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "l1.npz"
            model.save(checkpoint)
            decoded = read_checkpoint(checkpoint)
            restored = RenewableHuberRegressor.load(checkpoint)
        self.assertEqual(decoded.state.previous_lambda, previous_lambda)
        self.assertEqual(restored.state_.previous_lambda, previous_lambda)

    def test_overrides_are_applied_to_the_decoded_configuration(self) -> None:
        model, X, _ = _fitted_model(dtype="float64")
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "override.npz"
            model.save(checkpoint)
            payload = read_checkpoint(checkpoint)

        self.assertEqual(payload.config["dtype"], "float64")
        restored = RenewableHuberRegressor._from_checkpoint_payload(
            payload, backend="numpy", dtype="float32"
        )
        self.assertEqual(restored.backend_, "numpy")
        # Omitting an explicit device must re-resolve it for the new backend.
        self.assertEqual(restored.device, "auto")
        self.assertEqual(restored.coef_.dtype, np.dtype("float32"))
        np.testing.assert_allclose(restored.predict(X), model.predict(X), rtol=3e-5, atol=3e-5)

    def test_an_explicit_device_is_not_overwritten_by_the_backend_override(self) -> None:
        model, _, _ = _fitted_model()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "device.npz"
            model.save(checkpoint)
            restored = RenewableHuberRegressor.load(checkpoint, backend="numpy", device="cpu")
        self.assertEqual(restored.device, "cpu")
        self.assertEqual(restored.device_, "cpu")

    def test_a_subclass_loads_as_itself(self) -> None:
        class TaggedRegressor(RenewableHuberRegressor):
            """A subclass adding behaviour that must survive ``load``."""

            def tag(self) -> str:
                return "tagged"

        model, X, y = _fitted_model()
        subclass_model = TaggedRegressor().fit(X, y)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "subclass.npz"
            subclass_model.save(checkpoint)
            restored = TaggedRegressor.load(checkpoint)
            base = RenewableHuberRegressor.load(checkpoint)

        self.assertIsInstance(restored, TaggedRegressor)
        self.assertEqual(restored.tag(), "tagged")
        # The same archive still loads into the base class on request.
        self.assertIs(type(base), RenewableHuberRegressor)
        np.testing.assert_array_equal(restored.coef_, base.coef_)

    def test_resume_after_load_matches_an_uninterrupted_stream(self) -> None:
        rng = np.random.default_rng(2026)
        X = rng.normal(size=(200, 3))
        y = X @ np.asarray([0.9, -1.1, 0.3]) + 0.2
        weights = np.linspace(0.4, 1.6, X.shape[0])

        uninterrupted = RenewableHuberRegressor()
        uninterrupted.partial_fit(X[:80], y[:80], sample_weight=weights[:80])
        uninterrupted.partial_fit(X[80:], y[80:], sample_weight=weights[80:])

        resumable = RenewableHuberRegressor()
        resumable.partial_fit(X[:80], y[:80], sample_weight=weights[:80])
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "resume.npz"
            resumable.save(checkpoint)
            resumed = RenewableHuberRegressor.load(checkpoint)
            resumed.partial_fit(X[80:], y[80:], sample_weight=weights[80:])

        np.testing.assert_array_equal(resumed.coef_, uninterrupted.coef_)
        self.assertEqual(resumed.state_.effective_weight, uninterrupted.state_.effective_weight)
        self.assertEqual(resumed.state_.batch_count, 2)


if __name__ == "__main__":
    unittest.main()
