"""Opt-in bridge to the Rust/PyO3 and CUDA C++ native engine."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..exceptions import BackendContractError, BackendUnavailableError, ValidationError
from ._dlpack import adapt_cuda_dlpack
from .native_base import NativeEngineBackend

if TYPE_CHECKING:
    from ..config import EstimatorConfig
    from ..core import UpdateDiagnostics
    from ..state import RenewableHuberState

_EXPECTED_ABI_VERSION = 1
_EXPECTED_PYTHON_API_VERSION = 3


class NativeCudaBackend(NativeEngineBackend):
    """Host/DLPack adapter for the whole-batch native CUDA solver.

    Validation and checkpoint data intentionally remain NumPy-backed. The
    opaque native engine keeps its numerical state, CUDA handles, and reusable
    workspaces resident on one GPU across ``partial_fit`` calls.
    """

    name = "native_cuda"
    device = "cuda"
    # See NativeCpuBackend: the portable elementwise path is unreachable.
    supports_elementwise_workspace = False

    _EXPECTED_ABI_VERSION = _EXPECTED_ABI_VERSION
    _EXPECTED_PYTHON_API_VERSION = _EXPECTED_PYTHON_API_VERSION
    _EXTENSION_LABEL = "native CUDA"
    _ENGINE_INIT_ERROR = "The native CUDA engine could not initialize on the requested device"

    def __init__(
        self,
        dtype: str = "float64",
        *,
        device_id: int = 0,
        cuda_graphs: bool = False,
        cuda_fast_math: bool = False,
    ) -> None:
        super().__init__(dtype)
        try:
            from renewable_huber import _native_cuda
        except (ImportError, OSError) as error:
            raise BackendUnavailableError(
                "backend='native_cuda' requires the separately built Rust/CUDA extension. "
                "Build it with scripts/native/build_native_cuda.ps1 and ensure the matching "
                "CUDA Toolkit runtime is installed."
            ) from error

        try:
            version = _native_cuda.version()
        except Exception as error:
            raise BackendUnavailableError(
                "The native CUDA extension did not provide compatible version metadata"
            ) from error
        self._verify_version(version)
        if cuda_graphs and version.get("supports_cuda_graphs") is not True:
            raise BackendUnavailableError(
                "The native CUDA extension does not support requested CUDA Graph tuning"
            )
        if cuda_fast_math and version.get("supports_fast_math") is not True:
            raise BackendUnavailableError(
                "The native CUDA extension does not support requested fast-math tuning"
            )
        try:
            available = bool(_native_cuda.is_available())
            device_count = int(_native_cuda.device_count())
        except Exception as error:
            raise BackendUnavailableError(
                "The native CUDA extension could not query the CUDA runtime"
            ) from error
        if not available:
            raise BackendUnavailableError(
                "The renewable-huber native extension was built without CUDA support"
            )
        if device_id < 0 or device_id >= device_count:
            raise BackendUnavailableError(
                f"CUDA device {device_id} is unavailable; detected {device_count}"
            )
        self._native_module = _native_cuda
        self._initial_state_is_canonical_empty = version.get("initial_state") == "canonical_empty"
        self._supports_dlpack = version.get("device_input") == "dlpack"
        self._device_id = device_id
        self._cuda_graphs = cuda_graphs
        self._cuda_fast_math = cuda_fast_math
        self._engine: Any | None = None
        self._engine_state_token: int | None = None

    @property
    def cuda_features(self) -> dict[str, object]:
        if self._engine is not None:
            features = getattr(self._engine, "features", None)
            if callable(features):
                return dict(features())
        return {
            "cuda_graphs_requested": self._cuda_graphs,
            "cuda_graphs_enabled": False,
            "fast_math_requested": self._cuda_fast_math,
            "fast_math_enabled": False,
            "graph_captures": 0,
            "graph_replays": 0,
            "graph_fallbacks": 0,
        }

    @staticmethod
    def _cuda_dlpack_device(value: Any) -> tuple[int, int] | None:
        device_method = getattr(value, "__dlpack_device__", None)
        if not callable(device_method):
            return None
        try:
            device_type, device_id = device_method()
        except Exception:
            return None
        # DLPack device type 2 is kDLCUDA. Managed/host CUDA storage is not a
        # zero-copy device batch and is deliberately rejected by this path.
        if int(device_type) != 2:
            return None
        return int(device_type), int(device_id)

    def asarray(self, value: Any) -> Any:
        """Preserve CUDA DLPack producers; retain NumPy conversion for hosts."""

        value = adapt_cuda_dlpack(value)
        device = self._cuda_dlpack_device(value)
        if device is None:
            return super().asarray(value)
        if not self._supports_dlpack:
            raise BackendUnavailableError(
                "The loaded native CUDA extension does not support DLPack device input"
            )
        if device[1] != self._device_id:
            raise ValidationError(
                f"native CUDA input is on device {device[1]}, expected device {self._device_id}"
            )
        if not callable(getattr(value, "__dlpack__", None)):
            raise BackendContractError(
                "native CUDA device input must implement the DLPack protocol"
            )
        dtype_text = str(getattr(value, "dtype", "")).lower()
        dtype_names = tuple(name for name in ("float32", "float64") if name in dtype_text)
        if dtype_names and dtype_names[0] != self.dtype.name:
            # BackendContractError, not a bare TypeError: the estimator
            # translates an unrecognised TypeError from asarray into the
            # scikit-learn coercion message, which would hide exactly the
            # mismatch this line exists to report.
            raise BackendContractError(
                f"native CUDA DLPack dtype must exactly match {self.dtype.name}"
            )
        flags = getattr(value, "flags", None)
        is_contiguous = getattr(value, "is_contiguous", None)
        if flags is not None and not bool(getattr(flags, "c_contiguous", True)):
            raise ValidationError("native CUDA DLPack input must be C-contiguous")
        if callable(is_contiguous) and not bool(is_contiguous()):
            raise ValidationError("native CUDA DLPack input must be C-contiguous")
        return value

    def reshape(self, value: Any, shape: tuple[int, ...]) -> Any:
        if self._cuda_dlpack_device(value) is not None:
            return value.reshape(shape)
        return super().reshape(value, shape)

    @staticmethod
    def _device_scalar(value: Any) -> float:
        item = getattr(value, "item", None)
        return float(item() if callable(item) else value)

    def _device_namespace(self, value: Any) -> Any:
        module = type(value).__module__.split(".", 1)[0]
        if module == "cupy":
            import cupy

            return cupy
        if module == "torch":
            import torch

            return torch
        namespace = getattr(value, "__array_namespace__", None)
        if callable(namespace):
            return namespace()
        raise BackendContractError("CUDA DLPack input must expose finite/reduction operations")

    def is_finite(self, value: Any) -> bool:
        if self._cuda_dlpack_device(value) is None:
            return super().is_finite(value)
        validator = getattr(value, "is_finite", None)
        if callable(validator):
            return bool(validator())
        namespace = self._device_namespace(value)
        return bool(self._device_scalar(namespace.isfinite(value).all()))

    def minimum_scalar(self, value: Any) -> float:
        if self._cuda_dlpack_device(value) is None:
            return float(np.min(value))
        reducer = getattr(value, "minimum_scalar", None)
        if callable(reducer):
            return float(reducer())
        namespace = self._device_namespace(value)
        return self._device_scalar(namespace.min(value))

    def sum_scalar(self, value: Any) -> float:
        if self._cuda_dlpack_device(value) is None:
            return float(np.sum(value))
        reducer = getattr(value, "sum_scalar", None)
        if callable(reducer):
            return float(reducer())
        namespace = self._device_namespace(value)
        return self._device_scalar(namespace.sum(value))

    def renewable_update(
        self,
        X: np.ndarray,
        y: np.ndarray,
        state: RenewableHuberState,
        config: EstimatorConfig,
        *,
        sample_weight: np.ndarray | None,
        batch_weight: float,
    ) -> tuple[RenewableHuberState, UpdateDiagnostics]:
        """Run one complete unpenalized update without a Python solver loop."""

        if config.penalty != "none":
            raise ValidationError(
                "backend='native_cuda' currently supports penalty='none'; "
                "use backend='cupy' for the L1 solver"
            )

        n_parameters = int(state.coefficients.shape[0])
        self.restore_native_state(state, n_parameters=n_parameters)

        inputs = (X, y) if sample_weight is None else (X, y, sample_weight)
        device_flags = tuple(self._cuda_dlpack_device(value) is not None for value in inputs)
        if any(device_flags) and not all(device_flags):
            raise ValidationError(
                "native CUDA batches cannot mix host arrays and CUDA DLPack tensors"
            )
        device_input = all(device_flags)
        weights = sample_weight
        if not device_input and sample_weight is not None:
            weights = np.ascontiguousarray(sample_weight, dtype=self.dtype)
        with self._engine_call():
            update = self._engine.update_device if device_input else self._engine.update
            x_input = X if device_input else np.ascontiguousarray(X, dtype=self.dtype)
            y_input = y if device_input else np.ascontiguousarray(y, dtype=self.dtype)
            result = update(
                x_input,
                y_input,
                weights,
                batch_weight=float(batch_weight),
                n_features_in=state.n_features_in,
                fit_intercept=state.fit_intercept,
                tau=float(config.tau),
                bandwidth_scale=float(config.bandwidth_scale),
                max_iter=int(config.max_iter),
                tol=float(config.tol),
                ridge=float(config.ridge),
            )
        return self._adopt_result(result, state)

    def native_design_matrix(self, X: np.ndarray, *, fit_intercept: bool) -> np.ndarray:
        """Leave host features unexpanded; CUDA appends the intercept column.

        Materializing ``column_stack((X, ones))`` on the CPU copied the entire
        batch before every H2D transfer. The native ABI accepts the original
        feature matrix and expands it in reusable device workspace instead.
        ``fit_intercept`` remains explicit for protocol clarity; the engine
        derives the actual layout from state and configuration dimensions.
        """

        del fit_intercept
        return X

    def native_predict(self, X: np.ndarray, state: RenewableHuberState) -> np.ndarray:
        """Predict through the resident CUDA coefficient vector."""

        self.restore_native_state(state)
        with self._engine_call():
            if self._cuda_dlpack_device(X) is not None:
                raise ValidationError(
                    "native CUDA device-resident prediction is not supported yet; "
                    "pass a host array explicitly (no implicit device-to-host copy is performed)"
                )
            prediction = self._engine.predict(np.ascontiguousarray(X, dtype=self.dtype))
        return np.asarray(prediction, dtype=self.dtype)

    def _create_engine(self, n_parameters: int) -> Any:
        tuning: dict[str, bool] = {}
        if self._cuda_graphs:
            tuning["cuda_graphs"] = True
        if self._cuda_fast_math:
            tuning["fast_math"] = True
        return self._native_module.NativeCudaEngine(
            self.dtype.name, n_parameters, self._device_id, **tuning
        )

    def _new_engine_already_holds(self, state: RenewableHuberState) -> bool:
        """Skip the first restore when a new engine already equals ``state``.

        Avoids a redundant H2D copy, transpose, and stream sync on the first
        fit, while still restoring any non-canonical user state.
        """

        return (
            self._initial_state_is_canonical_empty
            and state.n_samples_seen == 0
            and state.batch_count == 0
            and state.previous_lambda == 0.0
            and state.effective_weight == 0.0
            and not np.any(state.coefficients)
            and not np.any(state.information)
        )
