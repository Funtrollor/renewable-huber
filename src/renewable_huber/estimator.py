"""Public scikit-learn-shaped API for renewable Huber regression."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .backends import ArrayBackend, resolve_backend
from .backends.capabilities import BackendCapabilities, capabilities_of
from .config import BackendName, DeviceName, DTypeName, EstimatorConfig, Penalty
from .core import UpdateDiagnostics, renewable_update
from .exceptions import BackendContractError, NotFittedError, ValidationError
from .serialization import (
    FORMAT_VERSION,
    CheckpointPayload,
    read_checkpoint,
    write_checkpoint,
)
from .state import RenewableHuberState


@dataclass(frozen=True, slots=True)
class _PreparedBatch:
    """Everything one validated batch contributes to the state transition."""

    design: Any
    target: Any
    weights: Any | None
    batch_weight: float
    state: RenewableHuberState
    first_batch: bool
    feature_names: np.ndarray | None


class RenewableHuberRegressor:
    """Robust linear regression that can be updated one batch at a time.

    Parameters mirror :class:`~renewable_huber.config.EstimatorConfig`. The
    estimator executes on NumPy, CuPy/CUDA, PyTorch, or TensorFlow tensors on
    CPU/CUDA. TensorFlow is supported in its default eager-execution mode.

    Notes
    -----
    ``fit`` resets the state and processes its input as one batch. Use
    ``partial_fit`` repeatedly for a genuine streaming workflow. The retained
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
        n_jobs: int | None = None,
        cuda_graphs: bool = False,
        cuda_fast_math: bool = False,
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
        self.n_jobs = n_jobs
        self.cuda_graphs = cuda_graphs
        self.cuda_fast_math = cuda_fast_math

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
            "n_samples_seen_",
            "n_iter_",
            "backend_",
            "device_",
            "n_jobs_",
            "cuda_features_",
        ):
            if hasattr(self, attribute):
                delattr(self, attribute)
        return self

    def fit(
        self, X: ArrayLike, y: ArrayLike, sample_weight: ArrayLike | None = None
    ) -> RenewableHuberRegressor:
        """Reset the estimator and fit a single initial batch."""

        return self.reset().partial_fit(X, y, sample_weight=sample_weight)

    def partial_fit(
        self, X: ArrayLike, y: ArrayLike, sample_weight: ArrayLike | None = None
    ) -> RenewableHuberRegressor:
        """Consume one batch while retaining only renewable summary state.

        ``sample_weight`` uses frequency-weight semantics: zero-weight rows are
        ignored numerically, and integer weights are equivalent to repeating
        rows. At least one row in every submitted batch must have positive
        weight.
        """

        config = self._config()
        config.validate()
        backend = getattr(self, "_backend", None)
        if backend is None:
            backend = self._resolve_backend(config)

        prepared = self._prepare_batch(X, y, sample_weight, config, backend)
        next_state, diagnostics = renewable_update(
            prepared.design,
            prepared.target,
            prepared.state,
            config,
            backend,
            sample_weight=prepared.weights,
            batch_weight=prepared.batch_weight,
        )
        self._commit(backend, next_state, diagnostics, prepared)
        return self

    def predict(self, X: ArrayLike) -> Any:
        """Predict with the current streamed estimator state."""

        state = self._require_state()
        backend = self._require_backend()
        self._validate_feature_names(X)
        X_array = self._validate_features(X, state.n_features_in, backend)
        design = self._design_matrix(X_array)
        capabilities = capabilities_of(backend)
        if capabilities.native_predict is not None:
            prediction = capabilities.native_predict(design, state)
            self._sync_backend_attributes(capabilities)
            return prediction
        return backend.xp.matmul(design, state.coefficients)

    def score(self, X: ArrayLike, y: ArrayLike, sample_weight: ArrayLike | None = None) -> float:
        """Return the ordinary coefficient of determination (R²)."""

        backend = self._require_backend()
        prediction = self.predict(X)
        y_array = self._validate_target(y, prediction.shape[0], backend)
        weights, weight_sum = self._validate_sample_weight(
            sample_weight, prediction.shape[0], backend
        )
        squared_residual = (y_array - prediction) ** 2
        if weights is None:
            residual_sum = backend.scalar(backend.xp.sum(squared_residual))
            target_mean = backend.xp.mean(y_array)
            total_sum = backend.scalar(backend.xp.sum((y_array - target_mean) ** 2))
        else:
            residual_sum = backend.scalar(backend.xp.sum(weights * squared_residual))
            target_mean = backend.xp.sum(weights * y_array) / weight_sum
            total_sum = backend.scalar(backend.xp.sum(weights * (y_array - target_mean) ** 2))
        if not total_sum:
            return 1.0 if not residual_sum else 0.0
        return 1.0 - residual_sum / total_sum

    @property
    def state_(self) -> RenewableHuberState:
        """Return a defensive copy of the renewable sufficient state."""

        return self._require_state().copy()

    @property
    def diagnostics_(self) -> UpdateDiagnostics:
        """Diagnostics generated by the most recently processed batch."""

        diagnostics = getattr(self, "_diagnostics", None)
        if diagnostics is None:
            raise NotFittedError("No batch has been processed")
        return diagnostics

    def state_dict(self) -> dict[str, object]:
        """Return portable model data for custom checkpointing integrations."""

        payload = self._checkpoint_payload()
        state = payload.state
        return {
            "config": payload.config,
            "coefficients": state.coefficients,
            "information": state.information,
            "n_samples_seen": state.n_samples_seen,
            "batch_count": state.batch_count,
            "previous_lambda": state.previous_lambda,
            "n_features_in": state.n_features_in,
            "fit_intercept": state.fit_intercept,
            "weight_sum": state.effective_weight,
            "feature_names_in": payload.feature_names,
        }

    def save(self, path: str | Path) -> Path:
        """Save configuration and sufficient state as a safe ``.npz`` checkpoint."""

        return write_checkpoint(self._checkpoint_payload(), path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        backend: BackendName | None = None,
        device: DeviceName | None = None,
        dtype: DTypeName | None = None,
    ) -> RenewableHuberRegressor:
        """Restore a model, optionally migrating its state to another backend."""

        return cls._from_checkpoint_payload(
            read_checkpoint(path),
            backend=backend,
            device=device,
            dtype=dtype,
        )

    def _checkpoint_payload(self) -> CheckpointPayload:
        """Project this fitted model onto the persistence boundary type.

        The state is copied to host arrays here, so the serialization layer
        never has to know which backend produced them. Diagnostics are left
        unset: no released checkpoint format records them, and inventing a
        last-batch summary the file cannot contain would be worse than absence.
        """

        state = self._require_state()
        backend = self._require_backend()
        host_state = RenewableHuberState(
            coefficients=backend.to_numpy(state.coefficients).copy(),
            information=backend.to_numpy(state.information).copy(),
            n_samples_seen=state.n_samples_seen,
            batch_count=state.batch_count,
            previous_lambda=state.previous_lambda,
            n_features_in=state.n_features_in,
            fit_intercept=state.fit_intercept,
            weight_sum=state.effective_weight,
        )
        return CheckpointPayload(
            format_version=FORMAT_VERSION,
            config=self.get_params(),
            state=host_state,
            feature_names=(
                self.feature_names_in_.tolist() if hasattr(self, "feature_names_in_") else None
            ),
        )

    @classmethod
    def _from_checkpoint_payload(
        cls,
        payload: CheckpointPayload,
        *,
        backend: BackendName | None = None,
        device: DeviceName | None = None,
        dtype: DTypeName | None = None,
    ) -> RenewableHuberRegressor:
        """Build an estimator of this class from a decoded checkpoint payload.

        ``cls`` is what gets constructed, so a subclass keeps its own type and
        any extra behaviour when it loads its own checkpoint.
        """

        config = dict(payload.config)
        if backend is not None:
            config["backend"] = backend
            # A backend change without an explicit device must not carry the
            # saved device forward: "auto" re-resolves it for the new backend.
            if device is None:
                config["device"] = "auto"
        if device is not None:
            config["device"] = device
        if dtype is not None:
            config["dtype"] = dtype
        try:
            model = cls(**config)
        except (KeyError, TypeError) as error:
            raise ValidationError("Invalid renewable-huber checkpoint configuration") from error
        model._restore_state(payload.state, feature_names=payload.feature_names)
        if payload.diagnostics is not None:
            model._diagnostics = payload.diagnostics
        return model

    def _restore_state(
        self, state: RenewableHuberState, *, feature_names: list[str] | None = None
    ) -> None:
        state.validate()
        config = self._config()
        config.validate()
        if state.fit_intercept != config.fit_intercept:
            raise ValidationError("checkpoint fit_intercept does not match configuration")
        self._backend = self._resolve_backend(config)
        self.backend_ = self._backend.name
        self.device_ = self._backend.device
        self._sync_backend_attributes(capabilities_of(self._backend))
        self._state = RenewableHuberState(
            coefficients=self._backend.asarray(state.coefficients),
            information=self._backend.asarray(state.information),
            n_samples_seen=state.n_samples_seen,
            batch_count=state.batch_count,
            previous_lambda=state.previous_lambda,
            n_features_in=state.n_features_in,
            fit_intercept=state.fit_intercept,
            weight_sum=state.effective_weight,
        )
        self.n_features_in_ = state.n_features_in
        self.n_samples_seen_ = state.n_samples_seen
        if feature_names is not None:
            if (
                not isinstance(feature_names, list)
                or len(feature_names) != state.n_features_in
                or not all(isinstance(name, str) for name in feature_names)
            ):
                raise ValidationError("checkpoint feature names do not match feature metadata")
            self.feature_names_in_ = np.asarray(feature_names, dtype=object)
        self._sync_public_coefficients()

    def _prepare_batch(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: ArrayLike | None,
        config: EstimatorConfig,
        backend: ArrayBackend,
    ) -> _PreparedBatch:
        """Validate one batch and assemble everything the update needs.

        Nothing here touches ``self``'s fitted state: a validation failure must
        leave the estimator exactly as it was.
        """

        state = getattr(self, "_state", None)
        first_batch = state is None
        if state is not None:
            self._validate_feature_names(X)
        X_array, y_array = self._validate_batch(X, y, backend)
        weights, batch_weight = self._validate_sample_weight(
            sample_weight, X_array.shape[0], backend
        )

        feature_names = None
        if first_batch:
            feature_names = self._feature_names(X)
            state = RenewableHuberState.empty(
                X_array.shape[1],
                fit_intercept=config.fit_intercept,
                xp=backend.xp,
                dtype=backend.dtype,
            )
        elif X_array.shape[1] != self.n_features_in_:
            raise ValidationError(
                f"X has {X_array.shape[1]} features, but RenewableHuberRegressor is expecting "
                f"{self.n_features_in_} features as input"
            )

        return _PreparedBatch(
            design=self._design_matrix(X_array, backend=backend),
            target=y_array,
            weights=weights,
            batch_weight=batch_weight,
            state=state,
            first_batch=first_batch,
            feature_names=feature_names,
        )

    def _commit(
        self,
        backend: ArrayBackend,
        next_state: RenewableHuberState,
        diagnostics: UpdateDiagnostics,
        prepared: _PreparedBatch,
    ) -> None:
        """Adopt a completed transition and project it onto the public attributes.

        Reached only after ``renewable_update`` returns, so a failed update
        leaves every fitted attribute untouched.
        """

        self._backend = backend
        self.backend_ = backend.name
        self.device_ = backend.device
        self._sync_backend_attributes(capabilities_of(backend))
        self._state = next_state
        self._diagnostics = diagnostics
        if prepared.first_batch:
            self.n_features_in_ = next_state.n_features_in
            if prepared.feature_names is not None:
                self.feature_names_in_ = prepared.feature_names
        self.n_samples_seen_ = next_state.n_samples_seen
        self.n_iter_ = diagnostics.iterations
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
            n_jobs=self.n_jobs,
            cuda_graphs=self.cuda_graphs,
            cuda_fast_math=self.cuda_fast_math,
        )

    @staticmethod
    def _resolve_backend(config: EstimatorConfig) -> ArrayBackend:
        native_threads = config.resolved_n_jobs() if config.backend == "native_cpu" else None
        return resolve_backend(
            config.backend,
            device=config.device,
            dtype=config.dtype,
            n_jobs=native_threads,
            cuda_graphs=config.cuda_graphs,
            cuda_fast_math=config.cuda_fast_math,
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
        RenewableHuberRegressor._reject_sparse(X)
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()  # type: ignore[union-attr]
        RenewableHuberRegressor._reject_complex(X, "X")
        try:
            X_array = backend.asarray(X)
        except BackendContractError:
            # A backend refusing input for a stated reason has already said
            # something more precise than the coercion message below. Only
            # an unrecognised TypeError means "this is not numeric data".
            raise
        except TypeError as error:
            raise TypeError("float() argument must be a string or a number") from error
        except ValidationError:
            # Same reasoning as BackendContractError above: ValidationError is
            # ours and already names the violated rule, while a raw ValueError
            # from array coercion carries no usable detail.
            raise
        except ValueError as error:
            raise ValidationError(
                "X must be a two-dimensional numeric array. Reshape your data."
            ) from error
        if len(X_array.shape) != 2:
            raise ValidationError("X must be a two-dimensional numeric array. Reshape your data.")
        if X_array.shape[0] == 0:
            raise ValidationError(
                f"Found array with 0 sample(s) (shape={X_array.shape}) while a minimum of 1 "
                "is required."
            )
        if X_array.shape[1] == 0:
            raise ValidationError(
                f"0 feature(s) (shape={X_array.shape}) while a minimum of 1 is required."
            )
        if expected_features is not None and X_array.shape[1] != expected_features:
            raise ValidationError(
                f"X has {X_array.shape[1]} features, but RenewableHuberRegressor is expecting "
                f"{expected_features} features as input"
            )
        if not backend.is_finite(X_array):
            raise ValidationError("X must not contain NaN or infinite values")
        return X_array

    @staticmethod
    def _validate_target(y: ArrayLike, expected_samples: int, backend: ArrayBackend) -> Any:
        RenewableHuberRegressor._reject_sparse(y)
        if hasattr(y, "to_numpy"):
            y = y.to_numpy()  # type: ignore[union-attr]
        RenewableHuberRegressor._reject_complex(y, "y")
        try:
            y_array = backend.reshape(backend.asarray(y), (-1,))
        except BackendContractError:
            # A backend refusing input for a stated reason has already said
            # something more precise than the coercion message below. Only
            # an unrecognised TypeError means "this is not numeric data".
            raise
        except TypeError as error:
            raise TypeError("float() argument must be a string or a number") from error
        except ValidationError:
            # Same reasoning as BackendContractError above: ValidationError is
            # ours and already names the violated rule, while a raw ValueError
            # from array coercion carries no usable detail.
            raise
        except ValueError as error:
            raise ValidationError("y must be a one-dimensional numeric array") from error
        if y_array.shape[0] != expected_samples:
            raise ValidationError("X and y must contain the same number of samples")
        if not backend.is_finite(y_array):
            raise ValidationError("y must not contain NaN or infinite values")
        return y_array

    @staticmethod
    def _validate_sample_weight(
        sample_weight: ArrayLike | None, expected_samples: int, backend: ArrayBackend
    ) -> tuple[Any | None, float]:
        if sample_weight is None:
            return None, float(expected_samples)
        RenewableHuberRegressor._reject_sparse(sample_weight)
        if hasattr(sample_weight, "to_numpy"):
            sample_weight = sample_weight.to_numpy()  # type: ignore[union-attr]
        RenewableHuberRegressor._reject_complex(sample_weight, "sample_weight")
        try:
            weights = backend.reshape(backend.asarray(sample_weight), (-1,))
        except BackendContractError:
            # A backend refusing input for a stated reason has already said
            # something more precise than the coercion message below. Only
            # an unrecognised TypeError means "this is not numeric data".
            raise
        except TypeError as error:
            raise TypeError("float() argument must be a string or a number") from error
        except ValidationError:
            # Same reasoning as BackendContractError above: ValidationError is
            # ours and already names the violated rule, while a raw ValueError
            # from array coercion carries no usable detail.
            raise
        except ValueError as error:
            raise ValidationError(
                "sample_weight must be a one-dimensional numeric array"
            ) from error
        if weights.shape[0] != expected_samples:
            raise ValidationError("sample_weight and X must contain the same number of samples")
        if not backend.is_finite(weights):
            raise ValidationError("sample_weight must not contain NaN or infinite values")
        capabilities = capabilities_of(backend)
        minimum = (
            capabilities.minimum_scalar(weights)
            if capabilities.minimum_scalar is not None
            else backend.scalar(backend.xp.min(weights))
        )
        if minimum < 0:
            raise ValidationError("sample_weight must be non-negative")
        weight_sum = (
            capabilities.sum_scalar(weights)
            if capabilities.sum_scalar is not None
            else backend.scalar(backend.xp.sum(weights))
        )
        if weight_sum <= 0:
            raise ValidationError("sample_weight cannot be all zero")
        return weights, weight_sum

    def _design_matrix(self, X: Any, *, backend: ArrayBackend | None = None) -> Any:
        # Only the update path passes ``backend``, so only it delegates the
        # design matrix. predict() deliberately builds the expanded matrix on
        # the host: the native CUDA engine's predict entry point requires the
        # full n_parameters width, while its update accepts unexpanded features
        # and appends the intercept on device.
        if backend is not None:
            capabilities = capabilities_of(backend)
            if capabilities.native_design_matrix is not None:
                return capabilities.native_design_matrix(X, fit_intercept=self.fit_intercept)
        if self.fit_intercept:
            if backend is None:
                backend = self._require_backend()
            return backend.xp.column_stack((X, backend.xp.ones(X.shape[0], dtype=backend.dtype)))
        return X

    def _validate_feature_names(self, X: Any) -> None:
        incoming = self._feature_names(X)
        fitted = getattr(self, "feature_names_in_", None)
        if incoming is None or fitted is None:
            return
        if not np.array_equal(incoming, fitted):
            raise ValidationError(
                "X feature names must match the names and order used during the first batch"
            )

    @staticmethod
    def _feature_names(X: Any) -> np.ndarray | None:
        columns = getattr(X, "columns", None)
        if columns is None:
            return None
        names = list(columns)
        string_flags = [isinstance(name, str) for name in names]
        if any(string_flags) and not all(string_flags):
            raise ValidationError(
                "X feature names must either all be strings or all be non-strings"
            )
        if not all(string_flags):
            return None
        return np.asarray(names, dtype=object)

    @staticmethod
    def _reject_sparse(value: Any) -> None:
        module = type(value).__module__
        if module == "sparse" or module.startswith(("scipy.sparse", "sparse.")):
            raise TypeError(
                "Sparse data was passed, but renewable-huber requires dense input; "
                "convert explicitly with X.toarray()"
            )

    @staticmethod
    def _reject_complex(value: Any, name: str) -> None:
        dtype = getattr(value, "dtype", None)
        if dtype is not None and "complex" in str(dtype).lower():
            raise ValidationError(f"Complex data not supported for {name}")

    def _sync_backend_attributes(self, capabilities: BackendCapabilities) -> None:
        """Project the backend's live reports onto the public fitted attributes.

        ``n_jobs_`` is always set, to ``None`` when the backend does not report
        one. ``cuda_features_`` is only created by a backend that has them, so
        its absence stays meaningful on non-CUDA backends.
        """

        self.n_jobs_ = capabilities.read_n_jobs() if capabilities.read_n_jobs else None
        if capabilities.read_cuda_features is not None:
            features = capabilities.read_cuda_features()
            if features is not None:
                self.cuda_features_ = dict(features)

    def _sync_public_coefficients(self) -> None:
        state = self._require_state()
        backend = self._require_backend()
        if state.fit_intercept:
            self.coef_ = backend.copy(state.coefficients[:-1])
            self.intercept_ = backend.scalar(state.coefficients[-1])
        else:
            self.coef_ = backend.copy(state.coefficients)
            self.intercept_ = 0.0

    def _require_state(self) -> RenewableHuberState:
        state = getattr(self, "_state", None)
        if state is None:
            raise NotFittedError("Call fit or partial_fit before using this estimator")
        return state

    def _require_backend(self) -> ArrayBackend:
        backend = getattr(self, "_backend", None)
        if backend is None:
            raise NotFittedError("Call fit or partial_fit before using this estimator")
        return backend

    def __sklearn_is_fitted__(self) -> bool:
        """Allow scikit-learn's ``check_is_fitted`` to inspect the adapter."""

        return getattr(self, "_state", None) is not None
