"""Typed description of the optional behaviour a backend may provide.

:class:`~renewable_huber.backends.protocol.ArrayBackend` describes what every
backend must do.  Beyond that, the estimator and the portable core used to probe
for a dozen further abilities with ``getattr(backend, "...", None)`` scattered
across two modules, which meant the set of optional names had no type, no single
definition, and no test that could notice one going missing.

:func:`capabilities_of` is now the only place that probing happens.  Everything
else asks a :class:`BackendCapabilities` for what it needs.

Two of these abilities are *not* fixed values.  ``effective_n_jobs`` reports the
requested thread count before a native engine exists and the engine's confirmed
count afterwards, and ``cuda_features`` carries CUDA Graph counters that keep
incrementing.  Capabilities therefore hold zero-argument accessors for those two
rather than snapshots, so a caller always sees the current value.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

__all__ = ["BackendCapabilities", "capabilities_of"]

_CACHE_ATTRIBUTE = "_renewable_huber_capabilities"


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    """Optional backend behaviour, resolved once instead of probed per call.

    Every field is either a bound callable the backend supplied or ``None``.
    ``None`` always means "this backend cannot do it, use the portable path" --
    never "it failed".
    """

    #: Run one whole batch inside a native engine, bypassing the Python solver.
    native_update: Callable[..., Any] | None = None
    #: Predict through resident native coefficients.
    native_predict: Callable[..., Any] | None = None
    #: Build the design matrix, possibly leaving the intercept to the engine.
    native_design_matrix: Callable[..., Any] | None = None

    #: Device-side reductions that avoid a scalar round trip through the host.
    minimum_scalar: Callable[[Any], float] | None = None
    sum_scalar: Callable[[Any], float] | None = None

    #: Fused elementwise kernels.  Each may still return ``None`` at call time
    #: to decline a particular input; callers must keep the portable fallback.
    huber_loss: Callable[[Any, float], Any] | None = None
    smoothed_curvature: Callable[[Any, float, float], Any] | None = None
    huber_score_and_smoothed_curvature: Callable[[Any, float, float], Any] | None = None

    #: Live accessors.  See the module docstring for why these are not values.
    read_n_jobs: Callable[[], int | None] | None = None
    read_cuda_features: Callable[[], Mapping[str, Any]] | None = None

    #: Whether ``xp.empty_like`` yields a workspace the portable solver can
    #: reuse across its elementwise passes.
    elementwise_workspace: bool = False


def _callable_or_none(backend: Any, name: str) -> Callable[..., Any] | None:
    value = getattr(backend, name, None)
    return value if callable(value) else None


def _reader(backend: Any, name: str) -> Callable[[], Any] | None:
    """Return an accessor that re-reads ``name`` on every call, or ``None``.

    ``hasattr`` evaluates the property once here to decide whether the backend
    offers it at all; the closure then re-reads it so callers observe changes.
    """

    if not hasattr(backend, name):
        return None
    return lambda: getattr(backend, name)


def _probe(backend: Any) -> BackendCapabilities:
    return BackendCapabilities(
        native_update=_callable_or_none(backend, "renewable_update"),
        native_predict=_callable_or_none(backend, "native_predict"),
        native_design_matrix=_callable_or_none(backend, "native_design_matrix"),
        minimum_scalar=_callable_or_none(backend, "minimum_scalar"),
        sum_scalar=_callable_or_none(backend, "sum_scalar"),
        huber_loss=_callable_or_none(backend, "cuda_huber_loss"),
        smoothed_curvature=_callable_or_none(backend, "cuda_smoothed_curvature"),
        huber_score_and_smoothed_curvature=_callable_or_none(
            backend, "cuda_huber_score_and_smoothed_curvature"
        ),
        read_n_jobs=_reader(backend, "effective_n_jobs"),
        read_cuda_features=_reader(backend, "cuda_features"),
        elementwise_workspace=bool(getattr(backend, "supports_elementwise_workspace", False)),
    )


def capabilities_of(backend: Any) -> BackendCapabilities:
    """Return what ``backend`` can do beyond the required protocol.

    A backend may declare a :class:`BackendCapabilities` as its ``capabilities``
    attribute.  Otherwise the set is derived structurally, which keeps
    third-party backends and the suite's test doubles working without having to
    inherit from anything.

    The result is cached on the backend instance.  Backends are built by
    :func:`~renewable_huber.backends.resolve_backend` and do not gain or lose
    methods afterwards, so a per-instance cache is safe -- and it matters,
    because the fused-kernel fields are read once per solver iteration.
    """

    cached = getattr(backend, _CACHE_ATTRIBUTE, None)
    if isinstance(cached, BackendCapabilities):
        return cached

    declared = getattr(backend, "capabilities", None)
    capabilities = declared if isinstance(declared, BackendCapabilities) else _probe(backend)

    try:
        setattr(backend, _CACHE_ATTRIBUTE, capabilities)
    except (AttributeError, TypeError):
        # A backend using __slots__ cannot hold the cache; deriving per call
        # stays correct, only slower.
        pass
    return capabilities
