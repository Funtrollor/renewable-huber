"""Framework-neutral CUDA DLPack producer adapters.

The native CUDA extension consumes the Python Array API DLPack protocol. CuPy
and PyTorch CUDA tensors implement that protocol directly. TensorFlow eager
tensors currently expose their zero-copy capsule through
``tf.experimental.dlpack.to_dlpack`` instead, so this module supplies the
missing protocol surface without ever materialising a NumPy array.
"""

from __future__ import annotations

import re
from typing import Any

from ..exceptions import BackendContractError, ValidationError

_DL_DEVICE_CUDA = 2
_TENSORFLOW_GPU_DEVICE = re.compile(r"(?:^|/)device:GPU:(\d+)$", re.IGNORECASE)
_TENSORFLOW_SHORT_GPU_DEVICE = re.compile(r"^/GPU:(\d+)$", re.IGNORECASE)


def _tensorflow_device_id(device: object) -> int | None:
    """Return TensorFlow's logical GPU ordinal without importing TensorFlow."""

    text = str(device or "")
    match = _TENSORFLOW_GPU_DEVICE.search(text) or _TENSORFLOW_SHORT_GPU_DEVICE.match(text)
    return None if match is None else int(match.group(1))


class TensorFlowCudaDlpackTensor:
    """Adapt one eager TensorFlow GPU tensor to the Python DLPack protocol.

    TensorFlow owns the storage and its capsule manager retains that storage
    for the consumer. The native extension claims and destroys each returned
    capsule exactly once after its private CUDA stream has finished copying the
    batch into reusable workspace.
    """

    __slots__ = ("_tensor", "_tf", "_device_id")

    def __init__(self, tensor: Any, tensorflow: Any) -> None:
        if not bool(tensorflow.executing_eagerly()):
            raise ValidationError(
                "native CUDA TensorFlow input requires eager execution; "
                "it cannot be consumed from tf.function"
            )
        if not bool(tensorflow.is_tensor(tensor)):
            raise BackendContractError("TensorFlow DLPack input must be an eager tf.Tensor")
        device_id = _tensorflow_device_id(getattr(tensor, "device", None))
        if device_id is None:
            raise ValidationError(
                "backend='native_cuda' requires TensorFlow input resident on a GPU"
            )
        self._tensor = tensor
        self._tf = tensorflow
        self._device_id = device_id

    @property
    def shape(self) -> tuple[int, ...]:
        shape = getattr(self._tensor, "shape", ())
        dimensions = shape.as_list() if hasattr(shape, "as_list") else tuple(shape)
        if any(dimension is None for dimension in dimensions):
            raise ValidationError("TensorFlow eager input must have a fully defined shape")
        return tuple(int(dimension) for dimension in dimensions)

    @property
    def dtype(self) -> Any:
        return self._tensor.dtype

    @property
    def tensor(self) -> Any:
        """Return the retained producer for internal device-only validation."""

        return self._tensor

    def reshape(self, shape: tuple[int, ...]) -> TensorFlowCudaDlpackTensor:
        return type(self)(self._tf.reshape(self._tensor, shape), self._tf)

    def __dlpack_device__(self) -> tuple[int, int]:
        return _DL_DEVICE_CUDA, self._device_id

    def __dlpack__(
        self,
        stream: int | None = None,
        *,
        max_version: tuple[int, int] | None = None,
        dl_device: tuple[int, int] | None = None,
        copy: bool | None = None,
    ) -> Any:
        del stream, max_version
        if copy is True:
            raise BufferError("native CUDA TensorFlow DLPack transport does not permit copies")
        if dl_device is not None and tuple(dl_device) != self.__dlpack_device__():
            raise BufferError("native CUDA TensorFlow DLPack transport cannot change devices")
        # TensorFlow's legacy exporter cannot negotiate the native engine's
        # consumer stream. Establish an explicit producer completion boundary
        # before exporting so the allocation is safe on any CUDA stream. This
        # synchronizes execution but does not copy tensor storage.
        async_wait = getattr(self._tf.experimental, "async_wait", None)
        if callable(async_wait):
            async_wait()
        else:
            config_experimental = getattr(getattr(self._tf, "config", None), "experimental", None)
            get_synchronous_execution = getattr(
                config_experimental, "get_synchronous_execution", None
            )
            synchronous = (
                get_synchronous_execution() if callable(get_synchronous_execution) else None
            )
            if synchronous is not True:
                raise RuntimeError(
                    "TensorFlow DLPack export cannot establish a safe CUDA stream boundary; "
                    "use a TensorFlow version with tf.experimental.async_wait or enable "
                    "synchronous eager execution"
                )
        return self._tf.experimental.dlpack.to_dlpack(self._tensor)

    def is_finite(self) -> bool:
        return bool(self._tf.reduce_all(self._tf.math.is_finite(self._tensor)).numpy())

    def minimum_scalar(self) -> float:
        return float(self._tf.reduce_min(self._tensor).numpy())

    def sum_scalar(self) -> float:
        return float(self._tf.reduce_sum(self._tensor).numpy())


def adapt_cuda_dlpack(value: Any) -> Any:
    """Return a zero-copy CUDA DLPack producer when the framework needs one."""

    module = type(value).__module__.split(".", 1)[0]
    if module == "torch" and bool(getattr(value, "requires_grad", False)):
        # ``detach`` is a storage-sharing view. PyTorch intentionally rejects
        # DLPack export of a tensor requiring gradients, while this estimator
        # is not an autograd operation.
        detach = getattr(value, "detach", None)
        if callable(detach):
            value = detach()
    if type(value).__module__.startswith("tensorflow"):
        # Always use the explicit synchronization adapter, including on a
        # future TensorFlow tensor that grows direct protocol methods. Its
        # current exporter cannot negotiate a consumer stream safely.
        import tensorflow as tf

        return TensorFlowCudaDlpackTensor(value, tf)
    if callable(getattr(value, "__dlpack__", None)) and callable(
        getattr(value, "__dlpack_device__", None)
    ):
        return value
    return value
