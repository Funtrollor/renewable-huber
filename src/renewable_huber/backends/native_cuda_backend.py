"""Opt-in bridge to the Rust/PyO3 and CUDA C++ native engine."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..exceptions import BackendUnavailableError, ValidationError
from .numpy_backend import NumPyBackend

if TYPE_CHECKING:
    from ..config import EstimatorConfig
    from ..core import UpdateDiagnostics
    from ..state import RenewableHuberState

_EXPECTED_ABI_VERSION = 1
_EXPECTED_PYTHON_API_VERSION = 2


class NativeCudaBackend(NumPyBackend):
    """Host-input adapter for the whole-batch native CUDA solver.

    Validation and checkpoint data intentionally remain NumPy-backed. The
    opaque native engine keeps its numerical state, CUDA handles, and reusable
    workspaces resident on one GPU across ``partial_fit`` calls.
    """

    name = "native_cuda"
    device = "cuda"

    def __init__(self, dtype: str = "float64", *, device_id: int = 0) -> None:
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
            compatible = (
                isinstance(version, dict)
                and version.get("abi_version") == _EXPECTED_ABI_VERSION
                and version.get("python_api_version") == _EXPECTED_PYTHON_API_VERSION
            )
        except Exception as error:
            raise BackendUnavailableError(
                "The native CUDA extension did not provide compatible version metadata"
            ) from error
        if not compatible:
            raise BackendUnavailableError(
                "The native CUDA extension is incompatible with this renewable-huber build; "
                f"expected ABI {_EXPECTED_ABI_VERSION} and Python API "
                f"{_EXPECTED_PYTHON_API_VERSION}, received {version!r}"
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
        self._device_id = device_id
        self._engine: Any | None = None
        self._engine_batch_count: int | None = None

    @property
    def native_version(self) -> str:
        """Return the loaded native ABI version."""

        return str(self._native_module.version())

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

        n_parameters = int(X.shape[1])
        self.restore_native_state(state, n_parameters=n_parameters)

        weights = (
            None if sample_weight is None else np.ascontiguousarray(sample_weight, dtype=self.dtype)
        )
        try:
            result = self._engine.update(
                np.ascontiguousarray(X, dtype=self.dtype),
                np.ascontiguousarray(y, dtype=self.dtype),
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
        except Exception:
            # A hard native error may leave a CUDA stream or scratch buffer in
            # an unusable state. The authoritative host state was not advanced,
            # so discard the opaque engine and recreate it from that mirror on
            # the next call instead of attempting a silent continuation.
            self._engine = None
            self._engine_batch_count = None
            raise
        self._engine_batch_count = int(result["batch_count"])
        return self._decode_result(result, state)

    def native_predict(self, X: np.ndarray, state: RenewableHuberState) -> np.ndarray:
        """Predict through the resident CUDA coefficient vector."""

        if self._engine is None:
            self.restore_native_state(state)
        elif self._engine_batch_count != state.batch_count:
            self.restore_native_state(state)
        try:
            prediction = self._engine.predict(np.ascontiguousarray(X, dtype=self.dtype))
        except Exception:
            self._engine = None
            self._engine_batch_count = None
            raise
        return np.asarray(prediction, dtype=self.dtype)

    def restore_native_state(
        self,
        state: RenewableHuberState,
        *,
        n_parameters: int | None = None,
    ) -> None:
        """Restore portable state while retaining an existing engine workspace.

        This internal hook is also used by the benchmark harness to measure a
        repeatable steady-state transition without timing handle creation.
        """

        if self._engine is None:
            if n_parameters is None:
                n_parameters = int(state.coefficients.shape[0])
            try:
                self._engine = self._native_module.NativeCudaEngine(
                    self.dtype.name, n_parameters, self._device_id
                )
            except Exception as error:
                self._engine = None
                self._engine_batch_count = None
                raise BackendUnavailableError(
                    "The native CUDA engine could not initialize on the requested device"
                ) from error
        elif self._engine_batch_count == state.batch_count:
            return
        try:
            self._engine.restore(
                np.ascontiguousarray(state.coefficients, dtype=self.dtype),
                np.ascontiguousarray(state.information, dtype=self.dtype),
                int(state.n_samples_seen),
                int(state.batch_count),
                float(state.previous_lambda),
                float(state.effective_weight),
            )
        except Exception:
            self._engine = None
            self._engine_batch_count = None
            raise
        self._engine_batch_count = state.batch_count

    def _decode_result(
        self, result: dict[str, Any], previous_state: RenewableHuberState
    ) -> tuple[RenewableHuberState, UpdateDiagnostics]:
        from ..core import UpdateDiagnostics
        from ..state import RenewableHuberState

        state = RenewableHuberState(
            coefficients=np.asarray(result["coefficients"], dtype=self.dtype).copy(),
            information=np.asarray(result["information"], dtype=self.dtype).copy(),
            n_samples_seen=int(result["n_samples_seen"]),
            batch_count=int(result["batch_count"]),
            previous_lambda=float(result["previous_lambda"]),
            n_features_in=previous_state.n_features_in,
            fit_intercept=previous_state.fit_intercept,
            weight_sum=float(result["weight_sum"]),
        )
        diagnostics = UpdateDiagnostics(
            iterations=int(result["iterations"]),
            converged=bool(result["converged"]),
            objective=float(result["objective"]),
            lambda_value=float(result["lambda_value"]),
            bandwidth=float(result["bandwidth"]),
            used_regularized_fallback=bool(result["used_regularized_fallback"]),
        )
        return state, diagnostics
