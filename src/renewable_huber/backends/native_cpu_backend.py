"""Opt-in bridge to the Rust/PyO3 native CPU engine."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..exceptions import BackendUnavailableError
from .native_base import NativeEngineBackend

if TYPE_CHECKING:
    from ..config import EstimatorConfig
    from ..core import UpdateDiagnostics
    from ..state import RenewableHuberState

_EXPECTED_ABI_VERSION = 1
_EXPECTED_PYTHON_API_VERSION = 2


class NativeCpuBackend(NativeEngineBackend):
    """Host-state adapter for the whole-batch native Rust CPU solver.

    The public estimator keeps validation, feature names, and portable state
    in NumPy. One opaque engine per estimator reuses Rust workspaces across
    batches while the NumPy state remains authoritative for checkpoints.
    """

    name = "native_cpu"
    device = "cpu"
    # Inherited from NumPyBackend but never applicable: this backend runs
    # the whole batch natively and never reaches the portable solver.
    supports_elementwise_workspace = False

    _EXPECTED_ABI_VERSION = _EXPECTED_ABI_VERSION
    _EXPECTED_PYTHON_API_VERSION = _EXPECTED_PYTHON_API_VERSION
    _EXTENSION_LABEL = "native CPU"
    _ENGINE_INIT_ERROR = "The native CPU engine could not initialize"

    def __init__(self, dtype: str = "float64", *, n_threads: int | None = None) -> None:
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
        except Exception as error:
            raise BackendUnavailableError(
                "The native CPU extension did not provide compatible version metadata"
            ) from error
        self._verify_version(version)

        self._native_module = _native_cpu
        self._requested_n_threads = n_threads
        self._engine: Any | None = None
        self._engine_state_token: int | None = None

    @property
    def effective_n_jobs(self) -> int | None:
        """Return the requested or engine-confirmed native worker count."""

        if self._engine is None:
            return self._requested_n_threads
        return int(self._engine.n_threads)

    def _create_engine(self, n_parameters: int) -> Any:
        return self._native_module.NativeCpuEngine(
            self.dtype.name,
            n_parameters,
            n_threads=self._requested_n_threads,
        )

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
        with self._engine_call():
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
        return self._adopt_result(result, state)

    def native_predict(self, X: np.ndarray, state: RenewableHuberState) -> np.ndarray:
        """Predict through the resident native coefficient vector."""

        self.restore_native_state(state)
        with self._engine_call():
            prediction = self._engine.predict(np.ascontiguousarray(X, dtype=self.dtype))
        return np.asarray(prediction, dtype=self.dtype)
