"""Opt-in bridge to the Rust/PyO3 native CPU engine."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..exceptions import BackendUnavailableError
from .numpy_backend import NumPyBackend

if TYPE_CHECKING:
    from ..config import EstimatorConfig
    from ..core import UpdateDiagnostics
    from ..state import RenewableHuberState

_EXPECTED_ABI_VERSION = 1
_EXPECTED_PYTHON_API_VERSION = 1


class NativeCpuBackend(NumPyBackend):
    """Host-state adapter for the whole-batch native Rust CPU solver.

    The public estimator keeps validation, feature names, and portable state
    in NumPy. One opaque engine per estimator reuses Rust workspaces across
    batches while the NumPy state remains authoritative for checkpoints.
    """

    name = "native_cpu"
    device = "cpu"

    def __init__(self, dtype: str = "float64") -> None:
        super().__init__(dtype)
        try:
            from renewable_huber import _native_cpu
        except (ImportError, OSError) as error:
            raise BackendUnavailableError(
                "backend='native_cpu' requires the separately built Rust extension. "
                "Install renewable-huber-native-cpu or build it with Maturin."
            ) from error

        try:
            version = _native_cpu.version()
            compatible = (
                isinstance(version, dict)
                and version.get("abi_version") == _EXPECTED_ABI_VERSION
                and version.get("python_api_version") == _EXPECTED_PYTHON_API_VERSION
            )
        except Exception as error:
            raise BackendUnavailableError(
                "The native CPU extension did not provide compatible version metadata"
            ) from error
        if not compatible:
            raise BackendUnavailableError(
                "The native CPU extension is incompatible with this renewable-huber build; "
                f"expected ABI {_EXPECTED_ABI_VERSION} and Python API "
                f"{_EXPECTED_PYTHON_API_VERSION}, received {version!r}"
            )
        self._native_module = _native_cpu
        self._engine: Any | None = None
        self._engine_batch_count: int | None = None

    @property
    def native_version(self) -> str:
        """Return the loaded native version metadata."""

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
        """Run one complete update without returning to the Python solver loop."""

        self.restore_native_state(state, n_parameters=int(X.shape[1]))
        weights = (
            None if sample_weight is None else np.ascontiguousarray(sample_weight, dtype=self.dtype)
        )
        try:
            result = self._engine.update(
                np.ascontiguousarray(X, dtype=self.dtype),
                np.ascontiguousarray(y, dtype=self.dtype),
                weights,
                batch_weight=float(batch_weight),
                n_features_in=int(state.n_features_in),
                fit_intercept=bool(state.fit_intercept),
                tau=float(config.tau),
                penalty=str(config.penalty),
                lambda_scale=float(config.lambda_scale),
                bandwidth_scale=float(config.bandwidth_scale),
                max_iter=int(config.max_iter),
                tol=float(config.tol),
                ridge=float(config.ridge),
            )
        except Exception:
            self._discard_engine()
            raise
        self._engine_batch_count = int(result["batch_count"])
        return self._decode_result(result, state)

    def native_predict(self, X: np.ndarray, state: RenewableHuberState) -> np.ndarray:
        """Predict through the resident native coefficient vector."""

        self.restore_native_state(state)
        try:
            prediction = self._engine.predict(np.ascontiguousarray(X, dtype=self.dtype))
        except Exception:
            self._discard_engine()
            raise
        return np.asarray(prediction, dtype=self.dtype)

    def restore_native_state(
        self,
        state: RenewableHuberState,
        *,
        n_parameters: int | None = None,
    ) -> None:
        """Restore portable state if the engine mirror is absent or stale."""

        if self._engine is None:
            if n_parameters is None:
                n_parameters = int(state.coefficients.shape[0])
            try:
                self._engine = self._native_module.NativeCpuEngine(self.dtype.name, n_parameters)
            except Exception as error:
                self._discard_engine()
                raise BackendUnavailableError(
                    "The native CPU engine could not initialize"
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
            self._discard_engine()
            raise
        self._engine_batch_count = int(state.batch_count)

    def _discard_engine(self) -> None:
        self._engine = None
        self._engine_batch_count = None

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
