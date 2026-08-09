"""Host-local runtime policy that lets ``backend="auto"`` use the Rust CPU engine.

``backend="auto"`` used to mean "NumPy on CPU, CuPy on CUDA" with no other
input.  That is safe but leaves the native CPU engine unused unless a caller
already knows it wins, and whether it wins is a property of *this* host: the
same shape is 1.17x-15.65x faster than NumPy on the fixed Ryzen runner and
slower than NumPy on the WSL2 development host, because the two have different
core counts and different linked BLAS builds.

The design constraints this module answers are therefore:

* **No CPU identity.**  Brand strings, model numbers and micro-architecture
  names are never read.  They do not predict BLAS quality, thread count or
  memory bandwidth, and a lookup table keyed on them silently mispredicts on
  every processor nobody measured.
* **Nothing persisted.**  A calibration written to disk becomes wrong the
  moment the file is copied to another machine, or the moment NumPy is
  relinked.  The cache lives in the process and dies with it -- and does not
  even survive a ``fork``, because a worker child is often pinned to a
  different set of cores than the parent that measured.
* **Valid only for the context it was taken in.**  The cache key carries a
  :class:`RuntimeSignature`: the affinity mask itself where the platform
  reports one, the usable CPU count, and the thread-pool environment.  Re-pin
  the process -- even to a different set of the *same size* -- or change
  ``OMP_NUM_THREADS``, and the stale measurement is discarded rather than
  reused.  None of those inputs names a processor.
* **Generalisation, not a shape map.**  Probes measure a handful of shapes;
  the decision is made for shapes nobody measured.  What is fitted is the
  *ratio* ``native / NumPy`` as a smooth function of normalised log work
  features, together with the uncertainty of that prediction, so extrapolating
  further from the probes automatically demands a larger measured win.
* **Bounded cost.**  A one-shot ``fit`` must never pay for a second full solve
  of the caller's data.  The hard bound is the ladder: a fixed set of small
  probe shapes run a fixed number of times, so the work is known before
  anything is measured and no clock can change it.  A soft probe-start
  deadline stops the ladder early on a slow host, and the whole thing only
  runs once the caller's own batch is large enough to amortise it.  See
  ``docs/cpu-auto-dispatch-rfc.md`` for the arithmetic.
* **Fail to NumPy.**  Every failure mode -- extension missing, engine refusing
  to build, probe raising, clock too coarse, degenerate fit, budget exhausted
  -- resolves to the portable NumPy backend and is recorded, never raised.

Explicit ``backend="numpy"`` and ``backend="native_cpu"`` do not consult any of
this, and ``device="cuda"`` never reaches it.
"""

from __future__ import annotations

import math
import os
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol

import numpy as np

try:  # Optional: installed transitively by scikit-learn, absent in NumPy-only installs.
    from threadpoolctl import threadpool_info as _threadpool_info
except ImportError:  # pragma: no cover - availability differs by installation extra
    _threadpool_info = None

__all__ = [
    "CalibrationKey",
    "CalibrationOutcome",
    "CpuDispatchCache",
    "CpuDispatchDecision",
    "DispatchPolicy",
    "Probe",
    "ProbeMeasurement",
    "ProbeRunner",
    "RatioModel",
    "RuntimeSignature",
    "auto_cpu_dispatch_applies",
    "current_runtime_signature",
    "reset_cpu_dispatch_cache",
    "select_cpu_backend",
]

#: The two engines compared.  Both consume and produce host NumPy arrays, which
#: is what makes swapping between them mid-decision free of any conversion.
NUMPY = "numpy"
NATIVE_CPU = "native_cpu"
ENGINES: tuple[str, str] = (NUMPY, NATIVE_CPU)


@dataclass(frozen=True, slots=True)
class Probe:
    """One calibration shape, expressed the way the estimator sees a batch."""

    samples: int
    features: int

    @property
    def parameters(self) -> int:
        """Design width.  Probes always carry an intercept column."""

        return self.features + 1

    @property
    def work_units(self) -> float:
        """Normalised work of one solver iteration.

        ``samples * parameters ** 2`` is the weighted Gram accumulation, which
        dominates every other term in the update for any shape this policy is
        allowed to consider.  It is a pure count, so it is comparable between
        hosts, dtypes and engines -- which is exactly what a budget argument
        needs.
        """

        return float(self.samples) * float(self.parameters) ** 2


#: Two levels per axis plus a geometric centre point.  A 2x2 factorial design
#: with a centre run is the smallest layout that identifies both slopes, keeps
#: the normal matrix well conditioned, and leaves residual degrees of freedom
#: for an honest noise estimate.  Ordered by work so that exhausting the budget
#: still leaves a design that spans both axes.
PROBE_LADDER: tuple[Probe, ...] = (
    Probe(samples=1_024, features=6),
    Probe(samples=8_192, features=6),
    Probe(samples=2_896, features=14),
    Probe(samples=1_024, features=32),
    Probe(samples=8_192, features=32),
)


@dataclass(frozen=True, slots=True)
class DispatchPolicy:
    """Every tunable in one reviewable place.

    Tests build their own instance rather than monkeypatching constants, so a
    threshold can be exercised without depending on the shipped value.
    """

    #: Floor for *using* a host model, in single-iteration work units.  Set at
    #: the bottom of the calibrated range -- the smallest probe -- so the model
    #: is never asked about a batch smaller than anything it measured, and so a
    #: batch too small to notice an engine swap never churns one.
    minimum_work_units: float = 5.0e4
    #: Floor for *paying* for a host model.  Acquiring evidence costs a ladder
    #: (:meth:`ladder_work_units`); using evidence already in the cache costs
    #: three dot products.  Only the first needs a batch large enough to
    #: amortise it, which is why the two thresholds are separate and why a
    #: process that has already calibrated dispatches far smaller batches.
    calibration_work_units: float = 1.5e8
    #: Fraction by which native must be predicted to beat NumPy.  *Every*
    #: native selection clears this independently: there is no sticky state, so
    #: two estimators asking about the same shape get the same answer whatever
    #: order they ran in, and one estimator's choice never lowers the bar for
    #: the next.
    enter_margin: float = 0.15
    #: Multiplier applied to the prediction standard error before comparing
    #: against the margin.  This is a deliberately blunt conservative
    #: allowance, **not** a calibrated confidence level: three coefficients are
    #: fitted from five probes, so two residual degrees of freedom remain, and
    #: no honest 95% claim can be read off a normal quantile at that size.  The
    #: residual-scale floor below dominates the arithmetic in practice.
    uncertainty_multiplier: float = 2.0
    #: Floor on the fitted residual scale, in log units.  Five probes can agree
    #: by luck; without a floor that would be read as certainty.  0.05 ~ 5%.
    minimum_log_residual_scale: float = 0.05
    #: Soft deadline for *starting* another probe, checked between probes.  A
    #: probe already running is never interrupted, so a calibration may finish
    #: after this instant; the hard bound on cost is the fixed ladder itself
    #: (:meth:`ladder_work_units`), which is what makes the budget argument in
    #: the RFC a statement about work rather than about a stopwatch.
    probe_start_deadline_seconds: float = 0.25
    probe_warmups: int = 1
    #: Timed rounds per probe.  Each round runs both engines, alternating which
    #: goes first, so neither engine is systematically measured on colder pages
    #: than the other.  Two is the smallest count that gives each engine one
    #: first position and one second position.
    probe_repeats: int = 2
    #: Probes pin the iteration count so both engines do identical work and so
    #: one probe cannot run away.  A real batch usually runs more iterations,
    #: over which native amortises its call overhead better than NumPy does, so
    #: a short probe understates native -- an error in the safe direction.
    probe_max_iter: int = 3
    #: The native engine rejects a non-positive tolerance; this is small enough
    #: that both engines run the full ``probe_max_iter``.
    probe_tol: float = 1e-12
    probe_ladder: tuple[Probe, ...] = PROBE_LADDER
    #: Three coefficients are fitted, so four probes is the smallest design
    #: with any residual degrees of freedom left.
    minimum_probes: int = 4
    #: Sanity rail on absurd inputs.  The leverage term already inflates the
    #: uncertainty smoothly with distance; this only refuses the ridiculous.
    maximum_log_extrapolation: float = math.log(1024.0)
    #: Probe datasets are generated from this seed, so a calibration depends on
    #: the host and nothing else.
    seed: int = 20_260_809

    def ordered_probes(self) -> tuple[Probe, ...]:
        """Return the ladder cheapest-first, so an early stop keeps the span."""

        return tuple(sorted(self.probe_ladder, key=lambda probe: probe.work_units))

    def ladder_work_units(self) -> float:
        """Total normalised work one full calibration spends.

        Counts every engine, every repetition, every warmup and the pinned
        iteration count, so it can be compared directly against a batch's own
        work.  **This is the hard bound on calibration cost**: it is fixed by
        the policy, independent of any clock, and cannot be exceeded however
        slow the host is.  The soft deadline only decides how much of it gets
        spent.  Used by the RFC and the benchmark tooling; nothing in the
        decision path depends on it.
        """

        per_call = sum(probe.work_units for probe in self.probe_ladder)
        calls = self.probe_warmups + self.probe_repeats
        return per_call * calls * len(ENGINES) * self.probe_max_iter


DEFAULT_POLICY = DispatchPolicy()


#: Variables every mainstream BLAS, OpenMP and Rayon build reads to size its
#: pool.  Their *values* are part of the execution context a measurement was
#: taken in; none of them names a processor.
THREAD_ENVIRONMENT_KEYS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "RAYON_NUM_THREADS",
)


@dataclass(frozen=True, slots=True)
class RuntimeSignature:
    """How much machine this process can currently use.

    A measurement is only valid while the execution context that produced it
    holds.  Pin the process to two cores, or set ``OMP_NUM_THREADS=1`` between
    fits, and the ratio between a threaded native engine and a threaded BLAS
    moves -- without anything about the *host* changing.  Folding this into the
    cache key means such a change misses the cache and re-measures instead of
    answering from a calibration that no longer describes the machine.

    Everything here is a CPU number, a count, or a string the caller set.
    Nothing identifies the processor, and nothing is written anywhere.

    Deliberately *not* here: the native extension's own pool size. It is not an
    independent input -- the requested pool is already ``CalibrationKey``'s
    ``n_threads`` and the pool it actually gets is governed by the affinity
    mask and ``RAYON_NUM_THREADS`` recorded below -- and reading it makes the
    signature change for a reason that has nothing to do with the machine: it
    is unknown until something imports the extension, and calibrating is what
    imports it, so the very first calibration would invalidate itself and be
    paid twice.
    """

    #: Which CPUs, not how many. Two four-core pinnings on different cores are
    #: different machines from a memory-locality and cache-sharing point of
    #: view, so a length alone would let one be answered from the other's
    #: measurement.
    affinity: tuple[int, ...] | None
    usable_cpus: int | None
    thread_environment: tuple[tuple[str, str], ...]
    #: Effective BLAS/OpenMP pools when optional ``threadpoolctl`` is present.
    #: Paths and architecture strings are deliberately excluded.
    threadpools: tuple[tuple[str, str, str, int | None], ...]
    threadpool_source: str
    #: How the CPU set was obtained, so an unavailable probe is visible in the
    #: record rather than indistinguishable from a one-core machine.
    source: str

    def describe(self) -> str:
        return (
            f"{self.usable_cpus if self.usable_cpus is not None else 'unknown'} usable CPUs "
            f"via {self.source}"
        )


def _cpu_context() -> tuple[tuple[int, ...] | None, int | None, str]:
    """Return ``(affinity mask, count, source)`` for this process, portably.

    ``sched_getaffinity`` is the only one of these that notices a ``taskset``
    or a cgroup, and it is the only one that reports *which* CPUs rather than
    how many, so it is preferred where it exists.  It is absent on macOS and
    Windows and can raise under a restrictive sandbox; ``cpu_count`` may return
    ``None`` anywhere.  Each degradation is named rather than guessed at, so a
    caller reading the record can tell "one core" from "could not look".
    """

    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        try:
            mask = tuple(sorted(affinity(0)))
        except OSError:
            pass
        else:
            return mask, len(mask), "sched_getaffinity"
    count = os.cpu_count()
    if count is not None:
        return None, count, "cpu_count"
    return None, None, "unavailable"


def _threadpool_context() -> tuple[tuple[tuple[str, str, str, int | None], ...], str]:
    """Return effective loaded BLAS/OpenMP pool sizes without requiring the helper.

    scikit-learn and joblib can change these pools through ``threadpoolctl``
    without changing any environment variable.  When the optional helper is
    unavailable, affinity and environment inputs still provide the portable
    baseline signature; an introspection failure must never break ``fit``.
    """

    if _threadpool_info is None:
        return (), "unavailable"
    try:
        pools = _threadpool_info()
    except Exception:
        return (), "error"

    snapshot: list[tuple[str, str, str, int | None]] = []
    for pool in pools:
        if not isinstance(pool, dict):
            continue
        threads = pool.get("num_threads")
        if isinstance(threads, bool) or not isinstance(threads, int):
            threads = None
        snapshot.append(
            (
                pool.get("user_api") if isinstance(pool.get("user_api"), str) else "",
                pool.get("internal_api") if isinstance(pool.get("internal_api"), str) else "",
                pool.get("prefix") if isinstance(pool.get("prefix"), str) else "",
                threads,
            )
        )
    return tuple(sorted(snapshot)), "threadpoolctl"


def current_runtime_signature() -> RuntimeSignature:
    """Sample the execution context this process is running in right now."""

    mask, usable, source = _cpu_context()
    threadpools, threadpool_source = _threadpool_context()
    return RuntimeSignature(
        affinity=mask,
        usable_cpus=usable,
        thread_environment=tuple(
            (name, os.environ[name]) for name in THREAD_ENVIRONMENT_KEYS if name in os.environ
        ),
        threadpools=threadpools,
        threadpool_source=threadpool_source,
        source=source,
    )


@dataclass(frozen=True, slots=True)
class CalibrationKey:
    """What a calibration is valid for.

    Not in the key: sample count and feature count, because generalising over
    those is the entire point; and anything naming the processor, because this
    process is already running on it and a brand string predicts nothing.
    """

    dtype: str
    penalty: str
    n_threads: int | None
    signature: RuntimeSignature


@dataclass(frozen=True, slots=True)
class ProbeMeasurement:
    """One probe shape timed on both engines."""

    samples: int
    parameters: int
    numpy_seconds: float
    native_seconds: float

    @property
    def ratio(self) -> float:
        return self.native_seconds / self.numpy_seconds


@dataclass(frozen=True, slots=True)
class RatioModel:
    """``log(native / NumPy)`` as a linear function of normalised log shape.

    Features are ``log(samples)`` and ``log(parameters)``, both centred on the
    probe design.  Centring is what keeps the normal matrix well conditioned
    and makes the leverage term read as "distance from what was measured".
    """

    coefficients: tuple[float, float, float]
    centre: tuple[float, float]
    inverse_gram: tuple[tuple[float, float, float], ...]
    residual_scale: float
    probe_count: int
    log_samples_span: tuple[float, float]
    log_parameters_span: tuple[float, float]

    def _row(self, samples: int, parameters: int) -> np.ndarray:
        return np.array(
            [
                1.0,
                math.log(samples) - self.centre[0],
                math.log(parameters) - self.centre[1],
            ],
            dtype=float,
        )

    def predict(self, samples: int, parameters: int) -> tuple[float, float]:
        """Return the predicted ``log`` ratio and its prediction standard error.

        The ``1 +`` inside the square root is what makes this a prediction
        interval rather than a confidence interval on the mean: a single future
        measurement carries its own noise on top of the fitted line.
        """

        row = self._row(samples, parameters)
        mean = float(row @ np.asarray(self.coefficients, dtype=float))
        leverage = float(row @ np.asarray(self.inverse_gram, dtype=float) @ row)
        error = self.residual_scale * math.sqrt(1.0 + max(leverage, 0.0))
        return mean, error

    def log_extrapolation(self, samples: int, parameters: int) -> float:
        """Return how far outside the measured box the shape sits, in log units."""

        distances = (
            self.log_samples_span[0] - math.log(samples),
            math.log(samples) - self.log_samples_span[1],
            self.log_parameters_span[0] - math.log(parameters),
            math.log(parameters) - self.log_parameters_span[1],
        )
        return max(0.0, *distances)


@dataclass(frozen=True, slots=True)
class CalibrationOutcome:
    """The result of trying to calibrate one key, successful or not.

    Immutable on purpose.  An earlier revision carried a mutable
    ``sticky_native`` flag so a key that had already chosen native could keep
    it under a looser threshold.  That made the answer depend on which
    estimator in the process asked first, which is not a property any caller
    can see or control; every native choice now clears the entry margin on its
    own evidence.
    """

    status: str
    reason: str
    model: RatioModel | None = None
    measurements: tuple[ProbeMeasurement, ...] = ()
    elapsed_seconds: float = 0.0

    @property
    def calibrated(self) -> bool:
        return self.model is not None


@dataclass(frozen=True, slots=True)
class CpuDispatchDecision:
    """A backend choice plus the evidence that produced it."""

    backend: str
    reason: str
    work_units: float
    calibrated: bool
    predicted_ratio: float | None = None
    ratio_upper_bound: float | None = None
    calibration_seconds: float = 0.0
    probe_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible record for ``auto_dispatch_``."""

        return {
            "backend": self.backend,
            "reason": self.reason,
            "work_units": self.work_units,
            "calibrated": self.calibrated,
            "predicted_ratio": self.predicted_ratio,
            "ratio_upper_bound": self.ratio_upper_bound,
            "calibration_seconds": self.calibration_seconds,
            "probe_count": self.probe_count,
        }


class ProbeRunner(Protocol):
    """What the calibrator needs from whatever actually executes a probe.

    Splitting *doing* from *timing* is what makes the policy testable: the
    suite supplies a runner that performs no work and a clock that advances by
    a scripted amount, and gets a fully deterministic simulated host.
    """

    def prepare(self, key: CalibrationKey) -> str | None:
        """Build both engines.  Return why native is unusable, or ``None``."""

    def prepare_probe(self, probe: Probe) -> None:
        """Materialise the probe's data outside any timed region."""

    def run(self, engine: str, probe: Probe) -> None:
        """Execute one update of ``probe`` on ``engine``."""


class NativeCpuProbeRunner:
    """Times real ``renewable_update`` calls for the two host CPU engines.

    Backends and datasets are built in :meth:`prepare` and :meth:`prepare_probe`
    so the region the calibrator times contains one update call and nothing
    else.  Each probe gets its own native backend: a resident engine is sized
    for one design width and cannot be handed a batch of another, which is also
    exactly how one estimator uses one engine for one stream.  The engine
    allocates its workspaces on its first call, which the untimed warmup
    absorbs.
    """

    def __init__(self, policy: DispatchPolicy = DEFAULT_POLICY) -> None:
        self._policy = policy
        self._key: CalibrationKey | None = None
        self._numpy_backend: Any = None
        self._native_backends: dict[Probe, Any] = {}
        self._config: Any = None
        self._data: dict[Probe, tuple[np.ndarray, np.ndarray]] = {}

    def prepare(self, key: CalibrationKey) -> str | None:
        from ..config import EstimatorConfig

        self._key = key
        self._config = EstimatorConfig(
            penalty=key.penalty,  # type: ignore[arg-type]
            dtype=key.dtype,  # type: ignore[arg-type]
            max_iter=self._policy.probe_max_iter,
            tol=self._policy.probe_tol,
            backend=NATIVE_CPU,
            n_jobs=key.n_threads,
        )
        self._numpy_backend = self._resolve(NUMPY)
        try:
            self._resolve(NATIVE_CPU)
        except Exception as error:  # BackendUnavailableError and anything under it
            return f"the native CPU engine is unusable on this host: {error}"
        return None

    def _resolve(self, engine: str) -> Any:
        from . import resolve_backend

        assert self._key is not None
        if engine == NUMPY:
            return resolve_backend(NUMPY, dtype=self._key.dtype)
        return resolve_backend(NATIVE_CPU, dtype=self._key.dtype, n_jobs=self._key.n_threads)

    def prepare_probe(self, probe: Probe) -> None:
        if probe in self._data:
            return
        self._native_backends[probe] = self._resolve(NATIVE_CPU)
        dtype = np.dtype(self._numpy_backend.dtype)
        rng = np.random.default_rng(self._policy.seed + probe.samples * 31 + probe.features)
        features = rng.standard_normal((probe.samples, probe.features), dtype=np.float64)
        coefficients = rng.standard_normal(probe.features)
        target = features @ coefficients + rng.standard_normal(probe.samples) * 0.25
        design = np.ascontiguousarray(
            np.column_stack((features, np.ones(probe.samples))), dtype=dtype
        )
        self._data[probe] = (design, np.ascontiguousarray(target, dtype=dtype))

    def run(self, engine: str, probe: Probe) -> None:
        from ..core import renewable_update
        from ..state import RenewableHuberState

        backend = self._numpy_backend if engine == NUMPY else self._native_backends[probe]
        design, target = self._data[probe]
        state = RenewableHuberState.empty(
            probe.features,
            fit_intercept=True,
            xp=backend.xp,
            dtype=backend.dtype,
        )
        renewable_update(
            design,
            target,
            state,
            self._config,
            backend,
            batch_weight=float(probe.samples),
        )


class CpuDispatchCache:
    """Process-local, thread-safe store of one calibration per key.

    Two callers racing on the same key must not both pay for a ladder, so a
    per-key lock serialises them and the loser sees the winner's result.
    Different keys still calibrate concurrently.  Nothing here is written to
    disk, so nothing here can outlive the process that measured it.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[CalibrationKey, CalibrationOutcome] = {}
        self._key_locks: dict[CalibrationKey, threading.Lock] = {}

    def outcome(
        self,
        key: CalibrationKey,
        calibrate: Callable[[], CalibrationOutcome],
    ) -> CalibrationOutcome:
        """Return the cached outcome, calibrating exactly once if absent."""

        with self._lock:
            cached = self._entries.get(key)
            if cached is not None:
                return cached
            key_lock = self._key_locks.setdefault(key, threading.Lock())

        with key_lock:
            with self._lock:
                cached = self._entries.get(key)
            if cached is not None:
                return cached
            outcome = calibrate()
            with self._lock:
                # A concurrent fork reset may have emptied the map; storing the
                # freshly measured outcome is correct either way.
                self._entries[key] = outcome
            return outcome

    def peek(self, key: CalibrationKey) -> CalibrationOutcome | None:
        """Return the cached outcome without ever starting a calibration."""

        with self._lock:
            return self._entries.get(key)

    def discard_other_signatures(self, signature: RuntimeSignature) -> int:
        """Drop every entry measured under a different execution context.

        Called before each lookup, so a change of CPU affinity or thread
        environment does not merely miss the cache -- it removes the stale
        measurement, which is what keeps a long-lived process that is
        repeatedly re-pinned from accumulating calibrations that can never be
        valid again.  Returns how many were dropped, for the tests.
        """

        with self._lock:
            stale = [key for key in self._entries if key.signature != signature]
            for key in stale:
                del self._entries[key]
                self._key_locks.pop(key, None)
            return len(stale)

    def snapshot(self) -> dict[CalibrationKey, CalibrationOutcome]:
        with self._lock:
            return dict(self._entries)

    def is_empty(self) -> bool:
        """Whether no execution context has paid for a calibration yet."""

        with self._lock:
            return not self._entries

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._key_locks.clear()

    def invalidate_after_fork(self) -> None:
        """Rebuild the locks and discard every measurement, in the child.

        Two separate problems, one fix.  The child inherits the memory image,
        including a mutex another thread held at the instant of the fork; that
        thread does not exist in the child, so an inherited lock is a deadlock
        waiting to happen.  And the child is frequently *not* running under the
        parent's execution context -- a joblib or multiprocessing worker may be
        pinned to a subset of cores, or have had its thread environment
        rewritten before the fork returned -- so the parent's measurements do
        not necessarily describe it.

        Discarding is cheap: the child re-measures only if it meets the
        calibration gate, and only once.
        """

        self._lock = threading.Lock()
        self._key_locks = {}
        self._entries = {}


_CACHE = CpuDispatchCache()

if hasattr(os, "register_at_fork"):  # pragma: no branch - POSIX only
    os.register_at_fork(after_in_child=_CACHE.invalidate_after_fork)


def reset_cpu_dispatch_cache() -> None:
    """Discard every cached calibration.  Intended for tests and benchmarks."""

    _CACHE.clear()


def auto_cpu_dispatch_applies(backend: str, device: str) -> bool:
    """Whether the workload-aware CPU policy governs this configuration.

    Only ``backend="auto"`` off the CUDA device.  An explicit backend name is
    an instruction, not a hint, and ``device="cuda"`` is resolved by
    :func:`~renewable_huber.backends.resolve_backend` exactly as before.
    """

    return backend == "auto" and device != "cuda"


def select_cpu_backend(
    *,
    samples: int,
    parameters: int,
    dtype: str,
    penalty: str,
    n_threads: int | None = None,
    policy: DispatchPolicy = DEFAULT_POLICY,
    cache: CpuDispatchCache | None = None,
    runner: ProbeRunner | None = None,
    clock: Callable[[], float] = perf_counter,
) -> CpuDispatchDecision:
    """Choose between NumPy and the native CPU engine for one batch shape.

    ``parameters`` is the design width the solver will actually see, so the
    intercept column is already included by the caller.

    A pure function of the batch shape, the policy, and whatever measurement
    the cache holds for the current execution context.  Calling it twice with
    the same arguments gives the same answer, and no caller's decision changes
    another caller's.
    """

    store = _CACHE if cache is None else cache
    work_units = float(samples) * float(parameters) ** 2
    if work_units < policy.minimum_work_units:
        return CpuDispatchDecision(
            backend=NUMPY,
            reason=(
                f"batch work {work_units:.3g} is below the {policy.minimum_work_units:.3g} "
                "dispatch threshold; NumPy is used without calibrating"
            ),
            work_units=work_units,
            calibrated=False,
        )

    # The common first small fit cannot reuse evidence that does not exist.
    # Return before optional thread-pool introspection so sklearn's richer
    # execution-context signature does not add latency to an uncalibrated
    # workload that is too small to measure anyway. A concurrent calibration
    # completing just after this check only makes this one call conservative.
    if work_units < policy.calibration_work_units and store.is_empty():
        return CpuDispatchDecision(
            backend=NUMPY,
            reason=(
                f"batch work {work_units:.3g} is below the "
                f"{policy.calibration_work_units:.3g} calibration threshold and this "
                "process has no host model yet; NumPy is used without measuring"
            ),
            work_units=work_units,
            calibrated=False,
        )

    signature = current_runtime_signature()
    key = CalibrationKey(dtype=dtype, penalty=penalty, n_threads=n_threads, signature=signature)
    # Anything measured under a different affinity or thread environment is not
    # merely a cache miss, it is wrong; drop it rather than leave it to rot.
    store.discard_other_signatures(signature)
    outcome = store.peek(key)
    if outcome is None:
        if work_units < policy.calibration_work_units:
            return CpuDispatchDecision(
                backend=NUMPY,
                reason=(
                    f"batch work {work_units:.3g} is below the "
                    f"{policy.calibration_work_units:.3g} calibration threshold and this "
                    "process has no host model yet; NumPy is used without measuring"
                ),
                work_units=work_units,
                calibrated=False,
            )
        outcome = store.outcome(
            key,
            lambda: _calibrate(
                key,
                policy=policy,
                runner=NativeCpuProbeRunner(policy) if runner is None else runner,
                clock=clock,
            ),
        )
    if outcome.model is None:
        return CpuDispatchDecision(
            backend=NUMPY,
            reason=outcome.reason,
            work_units=work_units,
            calibrated=False,
            calibration_seconds=outcome.elapsed_seconds,
            probe_count=len(outcome.measurements),
        )

    model = outcome.model
    distance = model.log_extrapolation(samples, parameters)
    if distance > policy.maximum_log_extrapolation:
        return CpuDispatchDecision(
            backend=NUMPY,
            reason=(
                f"shape {samples}x{parameters} is {math.exp(distance):.3g}x outside the "
                "calibrated range; the host model is not extrapolated that far"
            ),
            work_units=work_units,
            calibrated=True,
            calibration_seconds=outcome.elapsed_seconds,
            probe_count=model.probe_count,
        )

    mean, error = model.predict(samples, parameters)
    upper = mean + policy.uncertainty_multiplier * error
    threshold = math.log1p(-policy.enter_margin)
    native = upper < threshold

    outcome_phrase = (
        "clearing the entry margin"
        if native
        else f"short of the {math.exp(threshold):.3f} threshold; NumPy is retained"
    )
    reason = (
        f"calibrated host model predicts native/NumPy {math.exp(mean):.3f} "
        f"(conservative upper bound {math.exp(upper):.3f}) at {samples}x{parameters} "
        f"on {signature.describe()}, {outcome_phrase}"
    )
    return CpuDispatchDecision(
        backend=NATIVE_CPU if native else NUMPY,
        reason=reason,
        work_units=work_units,
        calibrated=True,
        predicted_ratio=math.exp(mean),
        ratio_upper_bound=math.exp(upper),
        calibration_seconds=outcome.elapsed_seconds,
        probe_count=model.probe_count,
    )


def _calibrate(
    key: CalibrationKey,
    *,
    policy: DispatchPolicy,
    runner: ProbeRunner,
    clock: Callable[[], float],
) -> CalibrationOutcome:
    """Run the probe ladder once and fit the host ratio model.

    Never raises.  Every path out of here is a :class:`CalibrationOutcome` the
    caller can act on, because a failure to measure must degrade to NumPy
    rather than break a ``fit`` the user did not ask to be optimised.
    """

    started = clock()

    def finished(status: str, reason: str, **extra: Any) -> CalibrationOutcome:
        return CalibrationOutcome(
            status=status,
            reason=reason,
            elapsed_seconds=max(0.0, clock() - started),
            **extra,
        )

    try:
        unavailable = runner.prepare(key)
    except Exception as error:
        return finished("failed", f"probe setup raised {type(error).__name__}: {error}")
    if unavailable is not None:
        return finished("unavailable", unavailable)

    # Soft: checked only between probes, so a probe that has started always
    # runs to completion. The hard bound on what this costs is the ladder.
    deadline = started + policy.probe_start_deadline_seconds
    measurements: list[ProbeMeasurement] = []
    for probe in policy.ordered_probes():
        if clock() >= deadline:
            break
        try:
            runner.prepare_probe(probe)
            timings = _time_probe_pair(runner, probe, policy=policy, clock=clock)
        except Exception as error:
            return finished(
                "failed",
                f"probe {probe.samples}x{probe.parameters} raised {type(error).__name__}: {error}",
                measurements=tuple(measurements),
            )
        if any(seconds <= 0.0 for seconds in timings.values()):
            # A clock too coarse to resolve this probe cannot support a ratio.
            # Larger probes may still resolve, so keep going.
            continue
        measurements.append(
            ProbeMeasurement(
                samples=probe.samples,
                parameters=probe.parameters,
                numpy_seconds=timings[NUMPY],
                native_seconds=timings[NATIVE_CPU],
            )
        )

    if len(measurements) < policy.minimum_probes:
        return finished(
            "insufficient",
            f"only {len(measurements)} of {len(policy.probe_ladder)} probes produced usable "
            f"timings before the {policy.probe_start_deadline_seconds:.3g}s probe-start "
            "deadline; NumPy is retained",
            measurements=tuple(measurements),
        )

    model = _fit_ratio_model(measurements, policy=policy)
    if model is None:
        return finished(
            "insufficient",
            "the probe timings do not identify a host cost model; NumPy is retained",
            measurements=tuple(measurements),
        )
    return finished(
        "calibrated",
        f"calibrated {len(measurements)} probes on this host",
        model=model,
        measurements=tuple(measurements),
    )


def engine_order(round_index: int) -> tuple[str, ...]:
    """Return the engine order for one round, alternating who goes first.

    Exported because the property it encodes is the point: with a fixed order,
    whichever engine runs first on a probe pays for the first touch of that
    probe's arrays and the first spin-up of its thread pool, and the other one
    is measured on a warm machine.  Always running NumPy first therefore biases
    every ratio in native's favour -- systematically, on every probe, in the
    one direction the policy is supposed to be conservative about.
    """

    return ENGINES if round_index % 2 == 0 else tuple(reversed(ENGINES))


def _time_probe_pair(
    runner: ProbeRunner,
    probe: Probe,
    *,
    policy: DispatchPolicy,
    clock: Callable[[], float],
) -> dict[str, float]:
    """Time both engines on one probe as an interleaved paired comparison.

    Each round runs both engines back to back and the rounds alternate which
    goes first, so over ``probe_repeats`` rounds each engine holds each
    position the same number of times and the first-touch cost cancels instead
    of landing on one side.  The per-engine minimum is then the right summary
    for a timing whose noise is one-sided: interference can only make a run
    slower than the machine is capable of.
    """

    best = {engine: math.inf for engine in ENGINES}
    round_index = 0
    for _ in range(policy.probe_warmups):
        for engine in engine_order(round_index):
            runner.run(engine, probe)
        round_index += 1
    for _ in range(policy.probe_repeats):
        for engine in engine_order(round_index):
            started = clock()
            runner.run(engine, probe)
            best[engine] = min(best[engine], clock() - started)
        round_index += 1
    return {
        engine: (0.0 if not math.isfinite(seconds) else seconds) for engine, seconds in best.items()
    }


def _fit_ratio_model(
    measurements: Sequence[ProbeMeasurement],
    *,
    policy: DispatchPolicy,
) -> RatioModel | None:
    """Fit ``log(native/NumPy) ~ 1 + log(samples) + log(parameters)``.

    Returns ``None`` when the design is rank deficient or the numbers are not
    finite, which is a refusal to dispatch rather than an error.
    """

    log_samples = np.array([math.log(item.samples) for item in measurements], dtype=float)
    log_parameters = np.array([math.log(item.parameters) for item in measurements], dtype=float)
    log_ratio = np.array([math.log(item.ratio) for item in measurements], dtype=float)
    if not np.isfinite(log_ratio).all():
        return None

    centre = (float(log_samples.mean()), float(log_parameters.mean()))
    design = np.column_stack(
        (
            np.ones(len(measurements)),
            log_samples - centre[0],
            log_parameters - centre[1],
        )
    )
    if np.linalg.matrix_rank(design) < design.shape[1]:
        return None
    try:
        coefficients, _, _, _ = np.linalg.lstsq(design, log_ratio, rcond=None)
        inverse_gram = np.linalg.inv(design.T @ design)
    except np.linalg.LinAlgError:
        return None
    if not (np.isfinite(coefficients).all() and np.isfinite(inverse_gram).all()):
        return None

    residuals = design @ coefficients - log_ratio
    degrees_of_freedom = len(measurements) - design.shape[1]
    scale = (
        math.sqrt(float(residuals @ residuals) / degrees_of_freedom)
        if degrees_of_freedom > 0
        else 0.0
    )
    return RatioModel(
        coefficients=tuple(float(value) for value in coefficients),  # type: ignore[arg-type]
        centre=centre,
        inverse_gram=tuple(tuple(float(value) for value in row) for row in inverse_gram),
        residual_scale=max(scale, policy.minimum_log_residual_scale),
        probe_count=len(measurements),
        log_samples_span=(float(log_samples.min()), float(log_samples.max())),
        log_parameters_span=(float(log_parameters.min()), float(log_parameters.max())),
    )
