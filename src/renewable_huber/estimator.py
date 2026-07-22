"""Public scikit-learn-shaped API for renewable Huber regression."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .backends import ArrayBackend, resolve_backend
from .config import BackendName, DeviceName, DTypeName, EstimatorConfig, Penalty
from .core import UpdateDiagnostics, renewable_update
from .exceptions import NotFittedError, ValidationError
from .state import RenewableHuberState


class RenewableHuberRegressor:
    """Robust linear regression that can be updated one batch at a time.

    Parameters mirror :class:`~renewable_huber.config.EstimatorConfig`.  The
    estimator executes on NumPy or, when installed and selected, CuPy/CUDA.
    PyTorch and TensorFlow names remain reserved for subsequent releases.

    Notes
    -----
    ``fit`` resets the state and processes its input as one batch.  Use
    ``partial_fit`` repeatedly for a genuine streaming workflow.  The retained
    state contains coefficients and an accumulated information matrix, never
    historical raw observations.
    """

    def __init__(
        self,
        *,
        tau: float = 1.345,
        penalty: Penalty = "none",
        lambda_scale: float = 1.0,
        bandwidth_scale: float = 1.0,
        fit_intercept: bool = True,
        max_iter: int = 100,
        tol: float = 1e-6,
        ridge: float = 1e-8,
        backend: BackendName = "auto",
        device: DeviceName = "auto",
        dtype: DTypeName = "float64",
    ) -> None:
        self.tau = tau
        self.penalty = penalty
        self.lambda_scale = lambda_scale
        self.bandwidth_scale = bandwidth_scale
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.ridge = ridge
        self.backend = backend
        self.device = device
        self.dtype = dtype
        self._backend: ArrayBackend | None = None
        self._state: RenewableHuberState | None = None
        self._diagnostics: UpdateDiagnostics | None = None

    def get_params(self, deep: bool = True) -> dict[str, object]:
        """Return constructor parameters, compatible with scikit-learn cloning."""

        del deep
        return self._config().to_dict()

    def set_params(self, **params: object) -> RenewableHuberRegressor:
        """Set constructor parameters and invalidate any fitted state."""

        allowed = set(self.get_params())
        unknown = set(params) - allowed
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown parameter(s): {names}")
        for name, value in params.items():
            setattr(self, name, value)
        return self.reset()

    def reset(self) -> RenewableHuberRegressor:
        """Discard fitted state so a new stream can begin."""

        self._backend = None
        self._state = None
        self._diagnostics = None
        for attribute in (
            "coef_",
            "intercept_",
            "n_features_in_",
            "feature_names_in_",
            "backend_",
            "device_",
        ):
            if hasattr(self, attribute):
                delattr(self, attribute)
        return self

    def fit(self, X: ArrayLike, y: ArrayLike) -> RenewableHuberRegressor:
        """Reset the estimator and fit a single initial batch."""

        return self.reset().partial_fit(X, y)

    def partial_fit(self, X: ArrayLike, y: ArrayLike) -> RenewableHuberRegressor:
        """Consume one batch while retaining only renewable summary state."""

        config = self._config()
        config.validate()
        if self._backend is None:
            self._backend = resolve_backend(
                config.backend, device=config.device, dtype=config.dtype
            )
            self.backend_ = self._backend.name
            self.device_ = self._backend.device
        X_array, y_array = self._validate_batch(X, y, self._backend)

        if self._state is None:
            self.n_features_in_ = X_array.shape[1]
            self._capture_feature_names(X)
            self._state = RenewableHuberState.empty(
                self.n_features_in_,
                fit_intercept=config.fit_intercept,
                xp=self._backend.xp,
                dtype=self._backend.dtype,
            )
        elif X_array.shape[1] != self.n_features_in_:
            raise ValidationError(
                f"X has {X_array.shape[1]} features, but the stream was initialized with "
                f"{self.n_features_in_}"
            )

        design = self._design_matrix(X_array)
        self._state, self._diagnostics = renewable_update(
            design, y_array, self._state, config, self._backend
        )
        self._sync_public_coefficients()
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        """Predict with the current streamed estimator state."""

        state = self._require_state()
        backend = self._require_backend()
        X_array = self._validate_features(X, state.n_features_in, backend)
        return self._design_matrix(X_array) @ state.coefficients

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return the ordinary coefficient of determination (R²)."""

        backend = self._require_backend()
        prediction = self.predict(X)
        y_array = self._validate_target(y, prediction.shape[0], backend)
        residual_sum = backend.scalar(backend.xp.sum((y_array - prediction) ** 2))
        total_sum = backend.scalar(backend.xp.sum((y_array - backend.xp.mean(y_array)) ** 2))
        return 1.0 - residual_sum / total_sum if total_sum else 0.0

    @property
    def state_(self) -> RenewableHuberState:
        """Return a defensive copy of the renewable sufficient state."""

        return self._require_state().copy()

    @property
    def diagnostics_(self) -> UpdateDiagnostics:
        """Diagnostics generated by the most recently processed batch."""

        if self._diagnostics is None:
            raise NotFittedError("No batch has been processed")
        return self._diagnostics

    def state_dict(self) -> dict[str, object]:
        """Return portable model data for custom checkpointing integrations."""

        state = self._require_state()
        return {
            "config": self.get_params(),
            "coefficients": self._require_backend().to_numpy(state.coefficients).copy(),
            "information": self._require_backend().to_numpy(state.information).copy(),
            "n_samples_seen": state.n_samples_seen,
            "batch_count": state.batch_count,
            "previous_lambda": state.previous_lambda,
            "n_features_in": state.n_features_in,
            "fit_intercept": state.fit_intercept,
        }

    def save(self, path: str | Path) -> Path:
        """Save configuration and sufficient state as a safe ``.npz`` checkpoint."""

        from .serialization import save_model

        return save_model(self, path)

    @classmethod
    def load(cls, path: str | Path) -> RenewableHuberRegressor:
        """Restore a model previously created by :meth:`save`."""

        from .serialization import load_model

        return load_model(path)

    def _restore_state(self, state: RenewableHuberState) -> None:
        state.validate()
        config = self._config()
        if state.fit_intercept != config.fit_intercept:
            raise ValidationError("checkpoint fit_intercept does not match configuration")
        self._backend = resolve_backend(config.backend, device=config.device, dtype=config.dtype)
        self.backend_ = self._backend.name
        self.device_ = self._backend.device
        self._state = RenewableHuberState(
            coefficients=self._backend.asarray(state.coefficients),
            information=self._backend.asarray(state.information),
            n_samples_seen=state.n_samples_seen,
            batch_count=state.batch_count,
            previous_lambda=state.previous_lambda,
            n_features_in=state.n_features_in,
            fit_intercept=state.fit_intercept,
        )
        self.n_features_in_ = state.n_features_in
        self._sync_public_coefficients()

    def _config(self) -> EstimatorConfig:
        return EstimatorConfig(
            tau=self.tau,
            penalty=self.penalty,
            lambda_scale=self.lambda_scale,
            bandwidth_scale=self.bandwidth_scale,
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter,
            tol=self.tol,
            ridge=self.ridge,
            backend=self.backend,
            device=self.device,
            dtype=self.dtype,
        )

    def _validate_batch(self, X: ArrayLike, y: ArrayLike, backend: ArrayBackend) -> tuple[Any, Any]:
        X_array = self._validate_features(X, backend=backend)
        return X_array, self._validate_target(y, X_array.shape[0], backend)

    @staticmethod
    def _validate_features(
        X: ArrayLike, expected_features: int | None = None, backend: ArrayBackend | None = None
    ) -> Any:
        if backend is None:
            raise RuntimeError("an initialized backend is required for input validation")
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()  # type: ignore[union-attr]
        X_array = backend.asarray(X)
        if X_array.ndim != 2:
            raise ValidationError("X must be a two-dimensional numeric array")
        if X_array.shape[0] == 0 or X_array.shape[1] == 0:
            raise ValidationError("X must contain at least one sample and one feature")
        if expected_features is not None and X_array.shape[1] != expected_features:
            raise ValidationError(f"X must contain exactly {expected_features} features")
        if not backend.is_finite(X_array):
            raise ValidationError("X must not contain NaN or infinite values")
        return X_array

    @staticmethod
    def _validate_target(y: ArrayLike, expected_samples: int, backend: ArrayBackend) -> Any:
        if hasattr(y, "to_numpy"):
            y = y.to_numpy()  # type: ignore[union-attr]
        y_array = backend.asarray(y).reshape(-1)
        if y_array.shape[0] != expected_samples:
            raise ValidationError("X and y must contain the same number of samples")
        if not backend.is_finite(y_array):
            raise ValidationError("y must not contain NaN or infinite values")
        return y_array

    def _design_matrix(self, X: Any) -> Any:
        if self.fit_intercept:
            backend = self._require_backend()
            return backend.xp.column_stack((X, backend.xp.ones(X.shape[0], dtype=backend.dtype)))
        return X

    def _capture_feature_names(self, X: Any) -> None:
        columns = getattr(X, "columns", None)
        if columns is not None:
            self.feature_names_in_ = np.asarray([str(column) for column in columns], dtype=object)

    def _sync_public_coefficients(self) -> None:
        state = self._require_state()
        if state.fit_intercept:
            self.coef_ = state.coefficients[:-1].copy()
            self.intercept_ = self._require_backend().scalar(state.coefficients[-1])
        else:
            self.coef_ = state.coefficients.copy()
            self.intercept_ = 0.0

    def _require_state(self) -> RenewableHuberState:
        if self._state is None:
            raise NotFittedError("Call fit or partial_fit before using this estimator")
        return self._state

    def _require_backend(self) -> ArrayBackend:
        if self._backend is None:
            raise NotFittedError("Call fit or partial_fit before using this estimator")
        return self._backend
