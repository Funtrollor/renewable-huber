from __future__ import annotations

import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

import numpy as np

from renewable_huber import RenewableHuberRegressor


def _tensorflow_ready() -> bool:
    return find_spec("tensorflow") is not None


@unittest.skipUnless(_tensorflow_ready(), "TensorFlow is required")
class TensorFlowBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import tensorflow as tf

        if not tf.executing_eagerly():
            raise unittest.SkipTest("TensorFlow eager execution is required")
        cls.tf = tf
        rng = np.random.default_rng(37)
        cls.X = rng.normal(size=(160, 5))
        cls.y = cls.X @ np.asarray([0.5, -1.4, 0.0, 2.0, -0.2]) + 0.1

    def test_cpu_tensors_match_numpy_streamed_updates(self) -> None:
        numpy_model = RenewableHuberRegressor(max_iter=100)
        tensorflow_model = RenewableHuberRegressor(backend="tensorflow", device="cpu", max_iter=100)
        for X_batch, y_batch in ((self.X[:80], self.y[:80]), (self.X[80:], self.y[80:])):
            numpy_model.partial_fit(X_batch, y_batch)
            tensorflow_model.partial_fit(
                self.tf.convert_to_tensor(X_batch, dtype=self.tf.float64),
                self.tf.convert_to_tensor(y_batch, dtype=self.tf.float64),
            )

        tensorflow_state = tensorflow_model.state_
        self.assertIsInstance(tensorflow_state.coefficients, self.tf.Tensor)
        self.assertEqual(tensorflow_model.backend_, "tensorflow")
        self.assertEqual(tensorflow_model.device_, "cpu")
        np.testing.assert_allclose(
            tensorflow_model.coef_.numpy(), numpy_model.coef_, rtol=2e-8, atol=2e-8
        )
        np.testing.assert_allclose(
            tensorflow_model.predict(self.tf.convert_to_tensor(self.X)).numpy(),
            numpy_model.predict(self.X),
            rtol=2e-8,
            atol=2e-8,
        )

    def test_auto_cpu_checkpoint_preserves_tensorflow_backend_and_dtype(self) -> None:
        model = RenewableHuberRegressor(backend="tensorflow", dtype="float32")
        model.fit(self.X, self.y)
        self.assertEqual(model.device_, "cpu")
        self.assertEqual(model.coef_.dtype, self.tf.float32)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "tensorflow-model.npz"
            model.save(checkpoint)
            restored = RenewableHuberRegressor.load(checkpoint)

        self.assertEqual(restored.backend_, "tensorflow")
        self.assertIsInstance(restored.state_.coefficients, self.tf.Tensor)
        np.testing.assert_allclose(
            restored.predict(self.tf.convert_to_tensor(self.X)).numpy(),
            model.predict(self.tf.convert_to_tensor(self.X)).numpy(),
            rtol=3e-5,
            atol=3e-5,
        )

    def test_cuda_tensors_match_numpy_when_available(self) -> None:
        if not self.tf.config.list_physical_devices("GPU"):
            self.skipTest("a GPU-enabled TensorFlow build is required")
        numpy_model = RenewableHuberRegressor(max_iter=100)
        tensorflow_model = RenewableHuberRegressor(
            backend="tensorflow", device="cuda", max_iter=100
        )
        numpy_model.fit(self.X, self.y)
        with self.tf.device("/GPU:0"):
            tensorflow_model.fit(
                self.tf.convert_to_tensor(self.X, dtype=self.tf.float64),
                self.tf.convert_to_tensor(self.y, dtype=self.tf.float64),
            )
        np.testing.assert_allclose(
            tensorflow_model.coef_.numpy(), numpy_model.coef_, rtol=2e-8, atol=2e-8
        )


if __name__ == "__main__":
    unittest.main()
