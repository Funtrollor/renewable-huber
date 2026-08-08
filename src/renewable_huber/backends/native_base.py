"""Shared machinery for the two native-engine backends.

The CPU and CUDA adapters converged on the same lifecycle: negotiate an ABI
version at construction, keep one opaque engine alive across calls, restore the
portable host state into it only when the mirror is stale, and discard the
engine outright if any native call fails.  Both had written that out
independently, including a `_decode_result` that was identical apart from a
comment, so a new state or diagnostics field could easily be added to one and
forgotten in the other.

Everything genuinely platform-specific stays in the subclasses: the extension
import and its error text, thread-count reporting, DLPack handling, CUDA tuning
flags and device validation.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import numpy as np

from ..exceptions import BackendUnavailableError
from .numpy_backend import NumPyBackend

if TYPE_CHECKING:
    from ..core import UpdateDiagnostics
    from ..state import RenewableHuberState


class NativeEngineBackend(NumPyBackend):
    """Host-state adapter for a resident native engine.

    The portable NumPy state stays authoritative.  The engine is a cache of it:
    it may be discarded at any point and rebuilt from the host mirror, which is
    what makes discarding the correct response to a native failure.
    """

    #: Version contract this build was written against.
    _EXPECTED_ABI_VERSION = 1
    _EXPECTED_PYTHON_API_VERSION: int
    #: Human-readable name used in version-negotiation errors.
    _EXTENSION_LABEL: str
    #: Raised when the engine constructor itself fails.
    _ENGINE_INIT_ERROR: str

    _native_module: Any
    _engine: Any | None
    _engine_state_token: int | None

    # -- version negotiation ------------------------------------------------

    @classmethod
    def _verify_version(cls, version: Any) -> None:
        """Reject an extension that does not speak this build's contract.

        Checked before any engine is created, so an incompatible extension can
        never allocate device resources or advance state.
        """

        compatible = (
            isinstance(version, dict)
            and version.get("abi_version") == cls._EXPECTED_ABI_VERSION
            and version.get("python_api_version") == cls._EXPECTED_PYTHON_API_VERSION
        )
        if not compatible:
            raise BackendUnavailableError(
                f"The {cls._EXTENSION_LABEL} extension is incompatible with this "
                f"renewable-huber build; expected ABI {cls._EXPECTED_ABI_VERSION} "
                f"and Python API {cls._EXPECTED_PYTHON_API_VERSION}, received {version!r}"
            )

    @property
    def native_version(self) -> str:
        """Return the loaded native version metadata."""

        return str(self._native_module.version())

    # -- engine lifecycle ---------------------------------------------------

    def _create_engine(self, n_parameters: int) -> Any:
        """Construct the platform's engine.  Implemented by each subclass."""

        raise NotImplementedError

    def _new_engine_already_holds(self, state: RenewableHuberState) -> bool:
        """Whether a freshly created engine already equals ``state``.

        Lets a backend skip the first restore when its engines start out in the
        canonical empty state.  Only ever consulted for an engine created in
        this call, so a stale engine can never take this path.
        """

        del state
        return False

    def _discard_engine(self) -> None:
        self._engine = None
        self._engine_state_token = None

    @contextmanager
    def _engine_call(self) -> Iterator[None]:
        """Discard the engine if the wrapped native call fails.

        A hard native failure can leave a stream, workspace, or device
        allocation unusable. The authoritative host state was not advanced, so
        rebuilding from that mirror on the next call is always safe -- and is
        the only alternative to silently continuing on an engine of unknown
        condition.
        """

        try:
            yield
        except Exception:
            self._discard_engine()
            raise

    def restore_native_state(
        self,
        state: RenewableHuberState,
        *,
        n_parameters: int | None = None,
    ) -> None:
        """Restore portable state if the engine mirror is absent or stale.

        This internal hook is also used by the benchmark harness to measure a
        repeatable steady-state transition without timing handle creation.
        """

        created_engine = self._engine is None
        if created_engine:
            if n_parameters is None:
                n_parameters = int(state.coefficients.shape[0])
            try:
                self._engine = self._create_engine(n_parameters)
            except Exception as error:
                self._discard_engine()
                raise BackendUnavailableError(self._ENGINE_INIT_ERROR) from error
        elif self._engine_state_token == state.mirror_token:
            return

        if created_engine and self._new_engine_already_holds(state):
            self._engine_state_token = state.mirror_token
            return

        with self._engine_call():
            self._engine.restore(
                np.ascontiguousarray(state.coefficients, dtype=self.dtype),
                np.ascontiguousarray(state.information, dtype=self.dtype),
                int(state.n_samples_seen),
                int(state.batch_count),
                float(state.previous_lambda),
                float(state.effective_weight),
            )
        self._engine_state_token = state.mirror_token

    # -- result decoding ----------------------------------------------------

    def _adopt_result(
        self, result: dict[str, Any], previous_state: RenewableHuberState
    ) -> tuple[RenewableHuberState, UpdateDiagnostics]:
        """Decode a completed update and re-point the mirror token at it."""

        # The engine has advanced, so it no longer mirrors ``previous_state``.
        # Leave the token invalid if result decoding itself fails.
        self._engine_state_token = None
        next_state, diagnostics = self._decode_result(result, previous_state)
        self._engine_state_token = next_state.mirror_token
        return next_state, diagnostics

    def _decode_result(
        self, result: dict[str, Any], previous_state: RenewableHuberState
    ) -> tuple[RenewableHuberState, UpdateDiagnostics]:
        """Build portable state and diagnostics from one native result dict.

        The accepted keys are pinned by native/contracts/rh_cuda_contract.json
        and checked against the PyO3 bindings by
        tests/test_native_cuda_contract.py.
        """

        from ..core import UpdateDiagnostics
        from ..state import RenewableHuberState

        coefficients = np.asarray(result["coefficients"], dtype=self.dtype)
        information = np.asarray(result["information"], dtype=self.dtype)
        # Current extension builds explicitly advertise that state arrays are
        # fresh Python-owned snapshots. Older builds and test doubles retain the
        # defensive copy the backend protocol requires.
        if not bool(result.get("state_is_detached", False)):
            coefficients = coefficients.copy()
            information = information.copy()
        state = RenewableHuberState(
            coefficients=coefficients,
            information=information,
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
