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

_EXPECTED_ABI_VERSION = 2
_EXPECTED_PYTHON_API_VERSION = 4
#: Assumed when an extension does not advertise ``supported_penalties``.
_DEFAULT_PENALTIES = frozenset({"none"})


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
                "backend='native_cuda' requires the separately distributed Rust/CUDA extension. "
                "Install it with `pip install renewable-huber-native-cuda` (Windows or Linux "
                "x86-64), which also installs the matching NVIDIA CUDA 12 runtime wheels."
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
        self._supports_device_predict = version.get("device_predict") == "dlpack"
        advertised = version.get("supported_penalties")
        #: Penalties the loaded engine implements. Read by capabilities_of();
        #: an extension that does not say is assumed to implement only "none".
        self.native_update_penalties = (
            frozenset(str(penalty) for penalty in advertised)
            if isinstance(advertised, (list, tuple, set, frozenset))
            else _DEFAULT_PENALTIES
        )
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
        """Run one complete update without a Python solver loop."""

        if config.penalty not in self.native_update_penalties:
            # The core refuses this first; kept for direct backend callers so an
            # engine is never handed a penalty it would not implement.
            raise ValidationError(
                f"the loaded native CUDA engine does not support penalty={config.penalty!r}; "
                f"supported: {sorted(self.native_update_penalties)}"
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
                penalty=str(config.penalty),
                lambda_scale=float(config.lambda_scale),
            )
        return self._adopt_result(result, state)

    def native_design_matrix(self, X: np.ndarray, *, fit_intercept: bool) -> np.ndarray:
        """Leave features unexpanded; CUDA appends the intercept column.

        Materializing ``column_stack((X, ones))`` on the CPU copied the entire
        batch before every H2D transfer, and cannot be done at all for a CUDA
        DLPack tensor without an implicit device-to-host copy. The native ABI
        accepts the original feature matrix for both updates and predictions
        and expands it in reusable device workspace instead. ``fit_intercept``
        remains explicit for protocol clarity; the engine derives the actual
        layout from the state's dimensions.
        """

        del fit_intercept
        return X

    def native_predict(self, X: Any, state: RenewableHuberState) -> np.ndarray:
        """Predict through the resident CUDA coefficient vector.

        Host input is copied to the device once. A CUDA DLPack tensor is read in
        place on the engine stream; only the predictions cross to the host.
        Either way the result is a NumPy array, as for every host-fed update.
        """

        device_input = self._cuda_dlpack_device(X) is not None
        if device_input and not self._supports_device_predict:
            raise BackendUnavailableError(
                "The loaded native CUDA extension does not support DLPack prediction input"
            )
        self.restore_native_state(state)
        with self._engine_call():
            if device_input:
                prediction = self._engine.predict_device(
                    X, state.n_features_in, state.fit_intercept
                )
            else:
                prediction = self._engine.predict(
                    np.ascontiguousarray(X, dtype=self.dtype),
                    state.n_features_in,
                    state.fit_intercept,
                )
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
