from __future__ import annotations

import ast
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import renewable_huber
from renewable_huber import RenewableHuberRegressor
from renewable_huber.backends._dlpack import (
    TensorFlowCudaDlpackTensor,
    _tensorflow_device_id,
    adapt_cuda_dlpack,
)
from renewable_huber.exceptions import BackendContractError, ValidationError


def _native_cuda_ready() -> bool:
    try:
        from renewable_huber import _native_cuda

        version = _native_cuda.version()
        return bool(
            _native_cuda.is_available()
            and _native_cuda.device_count()
            and version.get("python_api_version") == 3
        )
    except (ImportError, OSError, RuntimeError):
        return False


def _cupy_native_ready() -> bool:
    if not _native_cuda_ready():
        return False
    try:
        import cupy as cp

        return bool(cp.cuda.runtime.getDeviceCount())
    except (ImportError, OSError, RuntimeError):
        return False


def _torch_native_ready() -> bool:
    if not _native_cuda_ready():
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except (ImportError, OSError, RuntimeError):
        return False


def _tensorflow_native_ready() -> bool:
    if not _native_cuda_ready():
        return False
    try:
        import tensorflow as tf

        return bool(tf.executing_eagerly() and tf.config.list_logical_devices("GPU"))
    except (ImportError, OSError, RuntimeError):
        return False


class _FakeScalar:
    def __init__(self, value: object) -> None:
        self._value = value

    def numpy(self) -> object:
        return self._value


class _FakeTensor:
    __module__ = "tensorflow.python.framework.ops"

    def __init__(
        self,
        values: object,
        *,
        device: str = "/job:localhost/replica:0/task:0/device:GPU:0",
    ) -> None:
        self.values = np.asarray(values)
        self.shape = self.values.shape
        self.dtype = self.values.dtype
        self.device = device

    def numpy(self) -> object:
        raise AssertionError("the adapter must never materialize the source tensor on the host")


def _fake_tensorflow(*, eager: bool = True, async_wait: object = True) -> types.SimpleNamespace:
    exports: list[_FakeTensor] = []
    waits: list[None] = []

    def wait() -> None:
        waits.append(None)

    experimental = types.SimpleNamespace(
        dlpack=types.SimpleNamespace(
            to_dlpack=lambda tensor: exports.append(tensor) or ("capsule", id(tensor))
        )
    )
    if async_wait is True:
        experimental.async_wait = wait
    elif async_wait is not False:
        experimental.async_wait = async_wait
    tf = types.SimpleNamespace(
        executing_eagerly=lambda: eager,
        is_tensor=lambda value: isinstance(value, _FakeTensor),
        reshape=lambda tensor, shape: _FakeTensor(
            tensor.values.reshape(shape), device=tensor.device
        ),
        math=types.SimpleNamespace(is_finite=lambda tensor: np.isfinite(tensor.values)),
        reduce_all=lambda value: _FakeScalar(np.all(value)),
        reduce_min=lambda tensor: _FakeScalar(np.min(tensor.values)),
        reduce_sum=lambda tensor: _FakeScalar(np.sum(tensor.values)),
        experimental=experimental,
        config=types.SimpleNamespace(
            experimental=types.SimpleNamespace(get_synchronous_execution=lambda: False)
        ),
        _exports=exports,
        _waits=waits,
    )
    return tf


class DeviceInputContractErrorTests(unittest.TestCase):
    """Device-input refusals must use a type the estimator lets through.

    ``RenewableHuberRegressor._validate_features`` rewrites an unrecognised
    ``TypeError`` from ``backend.asarray`` into scikit-learn's coercion message.
    A bare ``TypeError`` raised here therefore reaches the caller as "float()
    argument must be a string or a number" instead of naming the dtype or
    protocol violation, with the real message surviving only as ``__cause__``.
    The modules are parsed rather than imported, so this runs on CPU CI where
    the native CUDA extension is absent.
    """

    SOURCES = (
        "backends/_dlpack.py",
        "backends/native_cuda_backend.py",
    )

    def _raised_types(self, relative: str) -> list[str]:
        path = Path(renewable_huber.__file__).parent / relative
        raised = []
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
                function = node.exc.func
                if isinstance(function, ast.Name):
                    raised.append(function.id)
                elif isinstance(function, ast.Attribute):
                    raised.append(function.attr)
        return raised

    def test_no_device_input_module_raises_a_bare_type_error(self) -> None:
        for relative in self.SOURCES:
            with self.subTest(module=relative):
                raised = self._raised_types(relative)
                # Anti-vacuity: a path typo would otherwise pass silently.
                self.assertTrue(raised, f"no raise statements parsed from {relative}")
                self.assertNotIn("TypeError", raised)

    def test_the_contract_type_is_actually_used(self) -> None:
        used = {name for relative in self.SOURCES for name in self._raised_types(relative)}
        self.assertIn("BackendContractError", used)

    def test_the_contract_type_survives_the_estimator_translation(self) -> None:
        # The rule the two modules above depend on, asserted directly.
        self.assertTrue(issubclass(BackendContractError, TypeError))


class TensorFlowDlpackAdapterTests(unittest.TestCase):
    def test_tensorflow_device_parser_accepts_canonical_gpu_names(self) -> None:
        self.assertEqual(_tensorflow_device_id("/device:GPU:3"), 3)
        self.assertEqual(_tensorflow_device_id("/job:a/task:0/device:GPU:12"), 12)
        self.assertEqual(_tensorflow_device_id("/GPU:1"), 1)
        self.assertIsNone(_tensorflow_device_id("/device:CPU:0"))

    def test_adapter_exports_after_explicit_async_wait_without_host_copy(self) -> None:
        tf = _fake_tensorflow()
        tensor = _FakeTensor([[1.0, 2.0]], device="/device:GPU:2")
        adapter = TensorFlowCudaDlpackTensor(tensor, tf)

        self.assertEqual(adapter.__dlpack_device__(), (2, 2))
        self.assertEqual(adapter.__dlpack__(stream=0x1234), ("capsule", id(tensor)))
        self.assertEqual(tf._waits, [None])
        self.assertEqual(tf._exports, [tensor])

    def test_adapter_rejects_unsafe_legacy_async_export(self) -> None:
        tf = _fake_tensorflow(async_wait=False)
        tensor = _FakeTensor([1.0], device="/device:GPU:0")
        adapter = TensorFlowCudaDlpackTensor(tensor, tf)

        with self.assertRaisesRegex(RuntimeError, "safe CUDA stream boundary"):
            adapter.__dlpack__(stream=7)
        self.assertEqual(tf._exports, [])

    def test_synchronous_eager_mode_is_a_safe_legacy_fallback(self) -> None:
        tf = _fake_tensorflow(async_wait=False)
        tf.config.experimental.get_synchronous_execution = lambda: True
        tensor = _FakeTensor([1.0], device="/device:GPU:0")

        self.assertEqual(
            TensorFlowCudaDlpackTensor(tensor, tf).__dlpack__(stream=9),
            ("capsule", id(tensor)),
        )

    def test_adapter_rejects_graph_mode_cpu_and_copy_requests(self) -> None:
        with self.assertRaisesRegex(ValidationError, "eager execution"):
            TensorFlowCudaDlpackTensor(_FakeTensor([1.0]), _fake_tensorflow(eager=False))
        with self.assertRaisesRegex(ValidationError, "resident on a GPU"):
            TensorFlowCudaDlpackTensor(
                _FakeTensor([1.0], device="/device:CPU:0"), _fake_tensorflow()
            )

        adapter = TensorFlowCudaDlpackTensor(_FakeTensor([1.0]), _fake_tensorflow())
        with self.assertRaisesRegex(BufferError, "does not permit copies"):
            adapter.__dlpack__(copy=True)
        with self.assertRaisesRegex(BufferError, "cannot change devices"):
            adapter.__dlpack__(dl_device=(2, 1))

    def test_native_cuda_fit_keeps_all_tensorflow_inputs_device_resident(self) -> None:
        received: list[tuple[object, object, object]] = []

        class FakeEngine:
            def __init__(
                self, dtype: str, n_parameters: int, device_id: int, **kwargs: object
            ) -> None:
                del device_id, kwargs
                self.dtype = np.dtype(dtype)
                self.n_parameters = n_parameters

            def update_device(
                self,
                X: object,
                y: object,
                sample_weight: object,
                **config: object,
            ) -> dict[str, object]:
                received.append((X, y, sample_weight))
                rows = int(getattr(X, "shape")[0])
                return {
                    "coefficients": np.zeros(self.n_parameters, dtype=self.dtype),
                    "information": np.eye(self.n_parameters, dtype=self.dtype),
                    "n_samples_seen": rows,
                    "batch_count": 1,
                    "previous_lambda": 0.0,
                    "weight_sum": float(config["batch_weight"]),
                    "iterations": 1,
                    "converged": True,
                    "used_regularized_fallback": False,
                    "objective": 0.0,
                    "lambda_value": 0.0,
                    "bandwidth": 0.5,
                }

            def restore(self, *args: object) -> None:
                del args

        extension = types.SimpleNamespace(
            NativeCudaEngine=FakeEngine,
            is_available=lambda: True,
            device_count=lambda: 1,
            version=lambda: {
                "abi_version": 1,
                "python_api_version": 3,
                "initial_state": "canonical_empty",
                "device_input": "dlpack",
            },
        )
        tf = _fake_tensorflow()
        X = _FakeTensor([[1.0, 2.0], [3.0, 4.0]])
        y = _FakeTensor([1.0, 2.0])
        weight = _FakeTensor([1.0, 2.0])

        with (
            mock.patch.dict(
                sys.modules,
                {"tensorflow": tf, "renewable_huber._native_cuda": extension},
            ),
            mock.patch.object(renewable_huber, "_native_cuda", extension, create=True),
        ):
            model = RenewableHuberRegressor(
                backend="native_cuda", device="cuda", dtype="float64"
            ).fit(X, y, sample_weight=weight)

        self.assertEqual(model.n_samples_seen_, 2)
        self.assertEqual(len(received), 1)
        self.assertTrue(all(isinstance(value, TensorFlowCudaDlpackTensor) for value in received[0]))
        self.assertEqual(
            tf._exports, []
        )  # The fake engine never consumes; no eager export occurred.

    def test_existing_protocol_producer_is_not_wrapped(self) -> None:
        producer = types.SimpleNamespace(
            __dlpack__=lambda stream=None: stream,
            __dlpack_device__=lambda: (2, 0),
        )
        self.assertIs(adapt_cuda_dlpack(producer), producer)

    def test_pytorch_requires_grad_is_detached_without_copy(self) -> None:
        class FakeTorchTensor:
            __module__ = "torch"

            def __init__(self, *, requires_grad: bool, storage: object) -> None:
                self.requires_grad = requires_grad
                self.storage = storage

            def detach(self) -> FakeTorchTensor:
                return type(self)(requires_grad=False, storage=self.storage)

            def __dlpack__(self, stream: int | None = None) -> object:
                return stream

            def __dlpack_device__(self) -> tuple[int, int]:
                return 2, 0

        storage = object()
        detached = adapt_cuda_dlpack(FakeTorchTensor(requires_grad=True, storage=storage))
        self.assertFalse(detached.requires_grad)
        self.assertIs(detached.storage, storage)


@unittest.skipUnless(_cupy_native_ready(), "native CUDA API 3, CuPy, and a GPU are required")
class CuPyDlpackProtocolIntegrationTests(unittest.TestCase):
    def test_consumer_stream_is_forwarded_once_per_capsule_without_host_staging(self) -> None:
        import cupy as cp

        calls: dict[str, list[int | None]] = {"X": [], "y": []}

        class TrackingProducer:
            def __init__(self, array: object, name: str) -> None:
                self.array = array
                self.name = name

            @property
            def shape(self) -> tuple[int, ...]:
                return tuple(self.array.shape)

            @property
            def dtype(self) -> object:
                return self.array.dtype

            @property
            def flags(self) -> object:
                return self.array.flags

            def reshape(self, shape: tuple[int, ...]) -> TrackingProducer:
                return type(self)(self.array.reshape(shape), self.name)

            def __dlpack_device__(self) -> tuple[int, int]:
                return self.array.__dlpack_device__()

            def __dlpack__(self, stream: int | None = None) -> object:
                calls[self.name].append(stream)
                return self.array.__dlpack__(stream=stream)

            def is_finite(self) -> bool:
                return bool(cp.isfinite(self.array).all().item())

        rng = np.random.default_rng(3301)
        X_host = rng.normal(size=(256, 8)).astype(np.float32)
        y_host = rng.normal(size=256).astype(np.float32)
        model = RenewableHuberRegressor(
            backend="native_cuda", device="cuda", dtype="float32", max_iter=30
        ).fit(
            TrackingProducer(cp.asarray(X_host), "X"),
            TrackingProducer(cp.asarray(y_host), "y"),
        )

        self.assertEqual(model.n_samples_seen_, X_host.shape[0])
        self.assertEqual(len(calls["X"]), 1)
        self.assertEqual(len(calls["y"]), 1)
        self.assertIsInstance(calls["X"][0], int)
        self.assertGreater(int(calls["X"][0]), 0)
        self.assertEqual(calls["X"], calls["y"])


@unittest.skipUnless(_torch_native_ready(), "native CUDA API 3 and CUDA PyTorch are required")
class PyTorchDlpackIntegrationTests(unittest.TestCase):
    def test_cuda_tensor_matches_host_input_and_detaches_autograd(self) -> None:
        import torch

        rng = np.random.default_rng(3302)
        X = rng.normal(size=(384, 5)).astype(np.float32)
        y = rng.normal(size=384).astype(np.float32)
        config = dict(backend="native_cuda", device="cuda", dtype="float32", max_iter=40)
        host = RenewableHuberRegressor(**config).fit(X, y)
        device = RenewableHuberRegressor(**config).fit(
            torch.as_tensor(X, device="cuda").requires_grad_(True),
            torch.as_tensor(y, device="cuda").requires_grad_(True),
        )

        np.testing.assert_allclose(device.coef_, host.coef_, rtol=4e-4, atol=4e-5)
        self.assertAlmostEqual(device.intercept_, host.intercept_, places=4)


@unittest.skipUnless(
    _tensorflow_native_ready(), "native CUDA API 3 and eager CUDA TensorFlow are required"
)
class TensorFlowDlpackIntegrationTests(unittest.TestCase):
    def test_eager_cuda_tensor_matches_host_input_without_storage_copy(self) -> None:
        import tensorflow as tf

        rng = np.random.default_rng(3303)
        X = rng.normal(size=(384, 5)).astype(np.float32)
        y = rng.normal(size=384).astype(np.float32)
        config = dict(backend="native_cuda", device="cuda", dtype="float32", max_iter=40)
        host = RenewableHuberRegressor(**config).fit(X, y)
        with tf.device("/GPU:0"):
            X_device = tf.convert_to_tensor(X)
            y_device = tf.convert_to_tensor(y)
        device = RenewableHuberRegressor(**config).fit(X_device, y_device)

        np.testing.assert_allclose(device.coef_, host.coef_, rtol=4e-4, atol=4e-5)
        self.assertAlmostEqual(device.intercept_, host.intercept_, places=4)


if __name__ == "__main__":
    unittest.main()
