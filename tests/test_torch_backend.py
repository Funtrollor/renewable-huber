from __future__ import annotations

import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

import numpy as np

from renewable_huber import RenewableHuberRegressor


def _torch_ready() -> bool:
    return find_spec("torch") is not None


@unittest.skipUnless(_torch_ready(), "PyTorch is required")
class TorchBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import torch

        cls.torch = torch
        rng = np.random.default_rng(21)
        cls.X = rng.normal(size=(160, 5))
        cls.y = cls.X @ np.asarray([1.2, -1.5, 0.0, 0.4, 0.7]) - 0.3

    def test_cpu_tensors_match_numpy_streamed_updates(self) -> None:
        numpy_model = RenewableHuberRegressor(max_iter=100)
        torch_model = RenewableHuberRegressor(backend="torch", device="cpu", max_iter=100)
        for X_batch, y_batch in ((self.X[:80], self.y[:80]), (self.X[80:], self.y[80:])):
            numpy_model.partial_fit(X_batch, y_batch)
            torch_model.partial_fit(
                self.torch.as_tensor(X_batch, dtype=self.torch.float64),
                self.torch.as_tensor(y_batch, dtype=self.torch.float64),
            )

        torch_state = torch_model.state_
        self.assertIsInstance(torch_state.coefficients, self.torch.Tensor)
        self.assertEqual(torch_model.backend_, "torch")
        self.assertEqual(torch_model.device_, "cpu")
        np.testing.assert_allclose(
            torch_model.coef_.detach().numpy(), numpy_model.coef_, rtol=2e-8, atol=2e-8
        )
        np.testing.assert_allclose(
            torch_model.predict(self.torch.as_tensor(self.X)).detach().numpy(),
            numpy_model.predict(self.X),
            rtol=2e-8,
            atol=2e-8,
        )

    def test_auto_cpu_checkpoint_preserves_torch_backend_and_dtype(self) -> None:
        model = RenewableHuberRegressor(backend="torch", dtype="float32")
        model.fit(self.X, self.y)
        self.assertEqual(model.device_, "cpu")
        self.assertEqual(model.coef_.dtype, self.torch.float32)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "torch-model.npz"
            model.save(checkpoint)
            restored = RenewableHuberRegressor.load(checkpoint)

        self.assertEqual(restored.backend_, "torch")
        self.assertIsInstance(restored.state_.coefficients, self.torch.Tensor)
        np.testing.assert_allclose(
            restored.predict(self.torch.as_tensor(self.X)).detach().numpy(),
            model.predict(self.torch.as_tensor(self.X)).detach().numpy(),
            rtol=3e-5,
            atol=3e-5,
        )

    def test_weighted_tensors_are_detached_and_match_numpy(self) -> None:
        weights = np.linspace(0.2, 1.8, self.X.shape[0])
        numpy_model = RenewableHuberRegressor(max_iter=100).fit(
            self.X, self.y, sample_weight=weights
        )
        X_tensor = self.torch.tensor(self.X, dtype=self.torch.float64, requires_grad=True)
        y_tensor = self.torch.tensor(self.y, dtype=self.torch.float64, requires_grad=True)
        weight_tensor = self.torch.tensor(weights, dtype=self.torch.float64, requires_grad=True)
        torch_model = RenewableHuberRegressor(backend="torch", device="cpu", max_iter=100).fit(
            X_tensor, y_tensor, sample_weight=weight_tensor
        )
        prediction = torch_model.predict(X_tensor)

        self.assertFalse(prediction.requires_grad)
        np.testing.assert_allclose(
            prediction.numpy(), numpy_model.predict(self.X), rtol=2e-8, atol=2e-8
        )
        self.assertAlmostEqual(torch_model.state_.effective_weight, float(weights.sum()), places=12)

    def test_torch_checkpoint_can_restore_to_numpy(self) -> None:
        model = RenewableHuberRegressor(backend="torch", device="cpu").fit(self.X, self.y)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "torch-to-numpy.npz"
            model.save(checkpoint)
            restored = RenewableHuberRegressor.load(checkpoint, backend="numpy", device="cpu")

        self.assertEqual(restored.backend_, "numpy")
        np.testing.assert_allclose(
            restored.predict(self.X),
            model.predict(self.torch.as_tensor(self.X)).numpy(),
            rtol=2e-8,
            atol=2e-8,
        )

    @unittest.skipUnless(_torch_ready(), "PyTorch is required")
    def test_cuda_tensors_match_numpy_when_available(self) -> None:
        if not self.torch.cuda.is_available():
            self.skipTest("a CUDA-enabled PyTorch build is required")
        numpy_model = RenewableHuberRegressor(max_iter=100)
        torch_model = RenewableHuberRegressor(backend="torch", device="cuda", max_iter=100)
        numpy_model.fit(self.X, self.y)
        torch_model.fit(
            self.torch.as_tensor(self.X, dtype=self.torch.float64, device="cuda"),
            self.torch.as_tensor(self.y, dtype=self.torch.float64, device="cuda"),
        )
        np.testing.assert_allclose(
            torch_model.coef_.detach().cpu().numpy(), numpy_model.coef_, rtol=2e-8, atol=2e-8
        )


if __name__ == "__main__":
    unittest.main()
