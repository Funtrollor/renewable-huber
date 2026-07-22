"""TensorFlow eager backend, imported lazily to keep TensorFlow optional."""

from __future__ import annotations

from typing import Any

from ..exceptions import BackendUnavailableError


class _TensorFlowNamespace:
    """Expose the small NumPy-shaped API needed by the portable core."""

    def __init__(self, tensorflow: Any, placement: str) -> None:
        self._tf = tensorflow
        self._placement = placement

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tf, name)

    def zeros(self, *shape: Any, dtype: Any = None) -> Any:
        with self._tf.device(self._placement):
            return self._tf.zeros(*shape, dtype=dtype)

    def ones(self, *shape: Any, dtype: Any = None) -> Any:
        with self._tf.device(self._placement):
            return self._tf.ones(*shape, dtype=dtype)

    def eye(self, size: int, *, dtype: Any = None) -> Any:
        with self._tf.device(self._placement):
            return self._tf.eye(size, dtype=dtype)

    def mean(self, value: Any) -> Any:
        return self._tf.reduce_mean(value)

    def sum(self, value: Any) -> Any:
        return self._tf.reduce_sum(value)

    def all(self, value: Any) -> Any:
        return self._tf.reduce_all(value)

    def isfinite(self, value: Any) -> Any:
        return self._tf.math.is_finite(value)

    def concatenate(self, values: tuple[Any, ...]) -> Any:
        return self._tf.concat(values, axis=0)

    def column_stack(self, values: tuple[Any, ...]) -> Any:
        columns = [
            value if value.shape.rank and value.shape.rank > 1 else self._tf.expand_dims(value, 1)
            for value in values
        ]
        return self._tf.concat(columns, axis=1)

    def transpose(self, value: Any) -> Any:
        return self._tf.transpose(value)

    def matmul(self, left: Any, right: Any) -> Any:
        """Match NumPy's matrix/vector ``@`` rules using TensorFlow primitives."""

        left_rank = left.shape.rank
        right_rank = right.shape.rank
        if left_rank == right_rank == 1:
            return self._tf.tensordot(left, right, axes=1)
        if right_rank == 1:
            return self._tf.linalg.matvec(left, right)
        if left_rank == 1:
            return self._tf.linalg.matvec(right, left, transpose_a=True)
        return self._tf.linalg.matmul(left, right)


class TensorFlowBackend:
    """TensorFlow backend for eager CPU and explicitly requested CUDA execution."""

    name = "tensorflow"

    def __init__(self, dtype: str = "float64", device: str = "auto") -> None:
        try:
            import tensorflow as tf
        except ImportError as error:
            raise BackendUnavailableError(
                "The TensorFlow backend requires TensorFlow. "
                "Install it with: pip install 'renewable-huber[gpu-tensorflow]'"
            ) from error

        if not tf.executing_eagerly():
            raise BackendUnavailableError(
                "The TensorFlow backend requires eager execution; do not call it from tf.function"
            )

        if device == "cuda":
            if not tf.config.list_physical_devices("GPU"):
                raise BackendUnavailableError(
                    "backend='tensorflow', device='cuda' requires a TensorFlow build with an "
                    "available GPU"
                )
            self._placement = "/GPU:0"
            self.device = "cuda:0"
        else:
            self._placement = "/CPU:0"
            self.device = "cpu"

        self._tf = tf
        self.dtype = getattr(tf, dtype)
        self.xp = _TensorFlowNamespace(tf, self._placement)

    def asarray(self, value: Any) -> Any:
        if hasattr(value, "to_numpy"):
            value = value.to_numpy()
        with self._tf.device(self._placement):
            return self._tf.identity(self._tf.cast(self._tf.convert_to_tensor(value), self.dtype))

    def copy(self, value: Any) -> Any:
        with self._tf.device(self._placement):
            return self._tf.identity(value)

    def reshape(self, value: Any, shape: tuple[int, ...]) -> Any:
        with self._tf.device(self._placement):
            return self._tf.reshape(value, shape)

    def to_numpy(self, value: Any) -> Any:
        return value.numpy()

    def scalar(self, value: Any) -> float:
        return float(value.numpy())

    def solve(self, matrix: Any, vector: Any) -> Any:
        right_hand_side = self._tf.expand_dims(vector, axis=-1)
        try:
            with self._tf.device(self._placement):
                solution = self._tf.linalg.solve(matrix, right_hand_side)
        except (self._tf.errors.InvalidArgumentError, self._tf.errors.UnimplementedError):
            with self._tf.device(self._placement):
                solution = self._tf.linalg.lstsq(matrix, right_hand_side, fast=False)
        return self._tf.squeeze(solution, axis=-1)

    def norm(self, value: Any) -> float:
        return self.scalar(self._tf.linalg.norm(value))

    def is_finite(self, value: Any) -> bool:
        return bool(self._tf.reduce_all(self._tf.math.is_finite(value)).numpy())

    def synchronize(self) -> None:
        """Eager TensorFlow finishes work before scalar conversion in this estimator."""
