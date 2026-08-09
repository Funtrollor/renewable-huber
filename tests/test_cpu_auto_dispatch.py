"""Runtime CPU auto-dispatch policy, verified against simulated hosts.

Every test here injects a clock and a probe runner, so the whole suite runs on
any CPU, needs no Rust extension, and produces the same numbers on a fast
workstation and a loaded CI container. That portability is the point: the
policy's job is to behave sensibly on hosts this repository will never see, and
the only way to test that is to fabricate them.

The simulated hosts are exact power laws, ``seconds = c * n**a * p**b``, so the
ratio the policy fits is log-linear by construction and the fitted surface can
be checked against ground truth at shapes no probe visited. A separate host is
deliberately *not* log-linear, to show what the uncertainty band is for.
"""

from __future__ import annotations

import ast
import json
import math
import sys
import threading
import types
import unittest
from dataclasses import dataclass, replace
from itertools import permutations
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np

import renewable_huber.backends.cpu_dispatch as cpu_dispatch
from renewable_huber import RenewableHuberRegressor
from renewable_huber.backends.cpu_dispatch import (
    PROBE_LADDER,
    CalibrationKey,
    CpuDispatchCache,
    DispatchPolicy,
    Probe,
    auto_cpu_dispatch_applies,
    select_cpu_backend,
)
from renewable_huber.backends.native_cpu_backend import NativeCpuBackend
from renewable_huber.backends.numpy_backend import NumPyBackend
from renewable_huber.exceptions import BackendUnavailableError

MODULE_PATH = Path(cpu_dispatch.__file__)


class FakeClock:
    """A monotonic clock advanced only by simulated work."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@dataclass(frozen=True, slots=True)
class PowerLaw:
    """``seconds = coefficient * samples**samples_power * parameters**parameters_power``."""

    coefficient: float
    samples_power: float = 1.0
    parameters_power: float = 2.0

    def seconds(self, samples: int, parameters: int) -> float:
        return (
            self.coefficient
            * float(samples) ** self.samples_power
            * float(parameters) ** self.parameters_power
        )


class SimulatedHost:
    """A :class:`ProbeRunner` that charges scripted costs to a fake clock."""

    def __init__(
        self,
        clock: FakeClock,
        numpy_engine: PowerLaw,
        native_engine: PowerLaw,
        *,
        unavailable: str | None = None,
        prepare_error: Exception | None = None,
        failing_engine: str | None = None,
    ) -> None:
        self.clock = clock
        self.engines = {"numpy": numpy_engine, "native_cpu": native_engine}
        self.unavailable = unavailable
        self.prepare_error = prepare_error
        self.failing_engine = failing_engine
        self.prepare_calls = 0
        self.prepared_probes: list[Probe] = []
        self.runs: list[tuple[str, Probe]] = []

    def prepare(self, key: CalibrationKey) -> str | None:
        self.prepare_calls += 1
        if self.prepare_error is not None:
            raise self.prepare_error
        return self.unavailable

    def prepare_probe(self, probe: Probe) -> None:
        self.prepared_probes.append(probe)

    def run(self, engine: str, probe: Probe) -> None:
        self.runs.append((engine, probe))
        if engine == self.failing_engine:
            raise RuntimeError("simulated probe failure")
        self.clock.advance(self.engines[engine].seconds(probe.samples, probe.parameters))

    def true_ratio(self, samples: int, parameters: int) -> float:
        return self.engines["native_cpu"].seconds(samples, parameters) / self.engines[
            "numpy"
        ].seconds(samples, parameters)


def unconstrained_policy(**overrides: Any) -> DispatchPolicy:
    """A policy whose size and budget gates are open, for model-focused tests."""

    settings: dict[str, Any] = {
        "minimum_work_units": 0.0,
        "calibration_work_units": 0.0,
        "probe_start_deadline_seconds": 1_000.0,
    }
    settings.update(overrides)
    return DispatchPolicy(**settings)


def decide(
    host: SimulatedHost,
    clock: FakeClock,
    samples: int,
    parameters: int,
    *,
    policy: DispatchPolicy | None = None,
    cache: CpuDispatchCache | None = None,
) -> cpu_dispatch.CpuDispatchDecision:
    return select_cpu_backend(
        samples=samples,
        parameters=parameters,
        dtype="float64",
        penalty="none",
        policy=policy or unconstrained_policy(),
        cache=cache or CpuDispatchCache(),
        runner=host,
        clock=clock,
    )


def sole_key(cache: CpuDispatchCache) -> CalibrationKey:
    """Return the cache's only key.

    Tests do not construct a :class:`CalibrationKey` by hand: it carries a
    :class:`RuntimeSignature` sampled from the machine the test is running on,
    and hard-coding one would either pin this suite to a particular affinity
    mask or quietly stop matching what the policy stores.
    """

    keys = list(cache.snapshot())
    assert len(keys) == 1, f"expected exactly one calibration, found {len(keys)}"
    return keys[0]


def uniform_host(ratio: float, *, numpy_scale: float = 1e-9) -> tuple[SimulatedHost, FakeClock]:
    """A host on which native is exactly ``ratio`` times NumPy at every shape."""

    clock = FakeClock()
    return (
        SimulatedHost(
            clock,
            PowerLaw(coefficient=numpy_scale),
            PowerLaw(coefficient=numpy_scale * ratio),
        ),
        clock,
    )


class ProbeLadderTests(unittest.TestCase):
    def test_the_ladder_spans_two_levels_on_each_axis(self) -> None:
        samples = {probe.samples for probe in PROBE_LADDER}
        parameters = {probe.parameters for probe in PROBE_LADDER}
        self.assertGreaterEqual(len(samples), 3)
        self.assertGreaterEqual(len(parameters), 3)

    def test_the_ladder_identifies_both_slopes(self) -> None:
        design = np.column_stack(
            (
                np.ones(len(PROBE_LADDER)),
                [math.log(probe.samples) for probe in PROBE_LADDER],
                [math.log(probe.parameters) for probe in PROBE_LADDER],
            )
        )
        self.assertEqual(np.linalg.matrix_rank(design), 3)
        self.assertGreater(len(PROBE_LADDER), 3, "a rank-3 fit needs residual freedom")

    def test_probes_are_ordered_cheapest_first(self) -> None:
        ordered = DispatchPolicy().ordered_probes()
        work = [probe.work_units for probe in ordered]
        self.assertEqual(work, sorted(work))
        self.assertEqual(set(ordered), set(PROBE_LADDER))

    def test_ladder_work_counts_every_call_it_will_make(self) -> None:
        policy = DispatchPolicy()
        expected = (
            sum(probe.work_units for probe in policy.probe_ladder)
            * (policy.probe_warmups + policy.probe_repeats)
            * 2
            * policy.probe_max_iter
        )
        self.assertAlmostEqual(policy.ladder_work_units(), expected)

    def test_the_shipped_gate_amortises_the_shipped_ladder(self) -> None:
        # The calibration gate exists so a batch is never asked to pay for more
        # exploration than a couple of its own iterations. Guarding the ratio
        # keeps a future ladder change from quietly breaking that promise.
        policy = DispatchPolicy()
        self.assertLessEqual(policy.ladder_work_units(), 2.0 * policy.calibration_work_units)
        smallest = min(probe.work_units for probe in policy.probe_ladder)
        largest = max(probe.work_units for probe in policy.probe_ladder)
        self.assertGreaterEqual(
            policy.minimum_work_units,
            smallest / 2.0,
            "dispatch must not run below the range the probes measured",
        )
        self.assertLessEqual(
            policy.minimum_work_units,
            largest,
            "a floor above the probes would waste the interpolation range",
        )
        self.assertLess(policy.minimum_work_units, policy.calibration_work_units)


class WorkThresholdTests(unittest.TestCase):
    def test_a_small_batch_never_touches_the_host(self) -> None:
        host, clock = uniform_host(0.1)
        decision = decide(host, clock, 512, 8, policy=DispatchPolicy())
        self.assertEqual(decision.backend, "numpy")
        self.assertFalse(decision.calibrated)
        self.assertEqual(host.prepare_calls, 0)
        self.assertIn("below the", decision.reason)

    def test_a_medium_batch_will_not_pay_for_a_first_calibration(self) -> None:
        host, clock = uniform_host(0.1)
        policy = DispatchPolicy()
        work_units = 0.5 * (policy.minimum_work_units + policy.calibration_work_units)
        parameters = 64
        samples = int(work_units / parameters**2)
        decision = decide(host, clock, samples, parameters, policy=policy)
        self.assertEqual(decision.backend, "numpy")
        self.assertEqual(host.prepare_calls, 0)
        self.assertIn("calibration threshold", decision.reason)

    def test_a_first_medium_batch_skips_runtime_introspection_too(self) -> None:
        host, clock = uniform_host(0.1)
        policy = DispatchPolicy()
        work_units = 0.5 * (policy.minimum_work_units + policy.calibration_work_units)
        parameters = 64
        samples = int(work_units / parameters**2)
        with mock.patch.object(
            cpu_dispatch,
            "current_runtime_signature",
            side_effect=AssertionError("small first fit must not inspect thread pools"),
        ):
            decision = decide(host, clock, samples, parameters, policy=policy)
        self.assertEqual(decision.backend, "numpy")

    def test_a_medium_batch_reuses_a_calibration_another_batch_paid_for(self) -> None:
        host, clock = uniform_host(0.1)
        policy = DispatchPolicy()
        cache = CpuDispatchCache()
        large = decide(host, clock, 200_000, 91, policy=policy, cache=cache)
        self.assertEqual(large.backend, "native_cpu")
        self.assertEqual(host.prepare_calls, 1)

        parameters = 64
        samples = int(0.5 * (policy.minimum_work_units + policy.calibration_work_units) / 64**2)
        medium = decide(host, clock, samples, parameters, policy=policy, cache=cache)
        self.assertEqual(medium.backend, "native_cpu")
        self.assertEqual(host.prepare_calls, 1, "the second decision must not re-measure")

    def test_dispatch_is_refused_below_the_smallest_useful_batch(self) -> None:
        host, clock = uniform_host(0.1)
        policy = DispatchPolicy()
        cache = CpuDispatchCache()
        decide(host, clock, 200_000, 91, policy=policy, cache=cache)
        tiny = decide(host, clock, 64, 8, policy=policy, cache=cache)
        self.assertEqual(tiny.backend, "numpy")
        self.assertIn("dispatch threshold", tiny.reason)


class SimulatedHostTests(unittest.TestCase):
    def test_a_host_where_native_wins_selects_native(self) -> None:
        host, clock = uniform_host(0.25)
        decision = decide(host, clock, 100_000, 91)
        self.assertEqual(decision.backend, "native_cpu")
        self.assertAlmostEqual(decision.predicted_ratio or 0.0, 0.25, places=6)
        self.assertLess(decision.ratio_upper_bound or 9.0, 0.85)

    def test_a_host_where_native_loses_keeps_numpy(self) -> None:
        host, clock = uniform_host(2.5)
        decision = decide(host, clock, 100_000, 91)
        self.assertEqual(decision.backend, "numpy")
        self.assertAlmostEqual(decision.predicted_ratio or 0.0, 2.5, places=6)

    def test_a_marginal_win_does_not_clear_the_entry_margin(self) -> None:
        host, clock = uniform_host(0.95)
        decision = decide(host, clock, 100_000, 91)
        self.assertEqual(decision.backend, "numpy")
        self.assertIn("short of the", decision.reason)

    def test_only_the_ratio_matters_not_the_absolute_speed(self) -> None:
        # A slow-NumPy host and a fast-NumPy host with the same relative
        # engines must decide identically: dispatch is a comparison, and
        # anchoring it to absolute seconds would make it machine-specific.
        slow_host, slow_clock = uniform_host(0.25, numpy_scale=1e-7)
        fast_host, fast_clock = uniform_host(0.25, numpy_scale=1e-11)
        slow = decide(slow_host, slow_clock, 100_000, 91)
        fast = decide(fast_host, fast_clock, 100_000, 91)
        self.assertEqual(slow.backend, fast.backend)
        self.assertAlmostEqual(slow.predicted_ratio or 0.0, fast.predicted_ratio or 1.0, places=9)

    def test_a_host_whose_crossover_sits_inside_the_probe_range(self) -> None:
        # native scales better in samples but worse in parameters, so it wins
        # tall-and-narrow and loses short-and-wide. One host, both answers.
        clock = FakeClock()
        host = SimulatedHost(
            clock,
            PowerLaw(coefficient=1e-9, samples_power=1.0, parameters_power=2.0),
            PowerLaw(coefficient=1.6e-10, samples_power=0.85, parameters_power=2.8),
        )
        cache = CpuDispatchCache()
        tall = decide(host, clock, 4_000_000, 12, cache=cache)
        wide = decide(host, clock, 4_000_000, 400, cache=cache)
        self.assertLess(host.true_ratio(4_000_000, 12), 0.5)
        self.assertGreater(host.true_ratio(4_000_000, 400), 1.5)
        self.assertEqual(tall.backend, "native_cpu")
        self.assertEqual(wide.backend, "numpy")
        self.assertEqual(host.prepare_calls, 1, "one calibration answers both shapes")


class GeneralisationTests(unittest.TestCase):
    def test_the_fitted_surface_reproduces_unmeasured_shapes(self) -> None:
        clock = FakeClock()
        host = SimulatedHost(
            clock,
            PowerLaw(coefficient=3e-9, samples_power=1.0, parameters_power=2.0),
            PowerLaw(coefficient=7e-8, samples_power=0.8, parameters_power=2.3),
        )
        cache = CpuDispatchCache()
        unseen = ((250_000, 33), (60_000, 129), (1_500_000, 9), (12_000, 257))
        for samples, parameters in unseen:
            with self.subTest(shape=(samples, parameters)):
                decision = decide(host, clock, samples, parameters, cache=cache)
                self.assertIsNotNone(decision.predicted_ratio)
                self.assertAlmostEqual(
                    math.log(decision.predicted_ratio or 0.0),
                    math.log(host.true_ratio(samples, parameters)),
                    places=6,
                    msg="a log-linear host must be reproduced exactly off the probe grid",
                )

    def test_no_measured_shape_is_stored_anywhere(self) -> None:
        # Generalisation, not a lookup: the model is three coefficients, a
        # centre and a covariance, and never the batch shapes it will be asked
        # about.
        host, clock = uniform_host(0.25)
        cache = CpuDispatchCache()
        decide(host, clock, 123_457, 91, cache=cache)
        outcome = cache.peek(sole_key(cache))
        assert outcome is not None and outcome.model is not None
        self.assertEqual(len(outcome.model.coefficients), 3)
        stored = {(item.samples, item.parameters) for item in outcome.measurements}
        self.assertNotIn((123_457, 91), stored)
        self.assertEqual(stored, {(probe.samples, probe.parameters) for probe in PROBE_LADDER})

    def test_uncertainty_grows_with_distance_from_the_probes(self) -> None:
        host, clock = uniform_host(0.25)
        cache = CpuDispatchCache()
        decide(host, clock, 4_096, 18, cache=cache)
        outcome = cache.peek(sole_key(cache))
        assert outcome is not None and outcome.model is not None
        model = outcome.model
        near = model.predict(4_096, 18)[1]
        far = model.predict(4_000_000, 18)[1]
        further = model.predict(400_000_000, 18)[1]
        self.assertLess(near, far)
        self.assertLess(far, further)

    def test_far_extrapolation_must_beat_a_larger_margin(self) -> None:
        # A uniform 0.70 clears the 15% entry margin next to the probes. The
        # host has not changed 1,000x further out; the confidence in the
        # prediction has, and that alone must be enough to decline.
        # Separate caches, so what is compared is distance alone and not the
        # hysteresis a previous native selection would have introduced.
        host, clock = uniform_host(0.70)
        near = decide(host, clock, 8_192, 49, cache=CpuDispatchCache())
        far = decide(host, clock, 8_000_000, 49, cache=CpuDispatchCache())
        self.assertEqual(near.backend, "native_cpu")
        self.assertEqual(far.backend, "numpy")
        self.assertAlmostEqual(near.predicted_ratio or 0.0, far.predicted_ratio or 1.0, places=6)
        self.assertGreater(far.ratio_upper_bound or 0.0, near.ratio_upper_bound or 9.0)

    def test_an_absurd_shape_is_refused_outright(self) -> None:
        host, clock = uniform_host(0.05)
        decision = decide(host, clock, 10**18, 49)
        self.assertEqual(decision.backend, "numpy")
        self.assertIn("outside the calibrated range", decision.reason)
        self.assertTrue(decision.calibrated)

    def test_a_misspecified_host_is_answered_with_doubt_not_confidence(self) -> None:
        # A ratio that is not log-linear cannot be fitted; what the policy owes
        # the caller is a wide band, not a wrong point estimate.
        clock = FakeClock()

        class Bumpy(SimulatedHost):
            def run(self, engine: str, probe: Probe) -> None:
                self.runs.append((engine, probe))
                seconds = self.engines[engine].seconds(probe.samples, probe.parameters)
                if engine == "native_cpu" and probe.parameters > 20:
                    seconds *= 40.0
                self.clock.advance(seconds)

        host = Bumpy(clock, PowerLaw(coefficient=1e-9), PowerLaw(coefficient=2e-10))
        cache = CpuDispatchCache()
        decision = decide(host, clock, 100_000, 91, cache=cache)
        outcome = cache.peek(sole_key(cache))
        assert outcome is not None and outcome.model is not None
        self.assertGreater(
            outcome.model.residual_scale, DispatchPolicy().minimum_log_residual_scale
        )
        self.assertGreater(
            (decision.ratio_upper_bound or 0.0) / (decision.predicted_ratio or 1.0),
            1.5,
            "a poor fit must widen the band it is judged on",
        )


class DecisionIndependenceTests(unittest.TestCase):
    """No estimator's choice may change another estimator's.

    An earlier revision kept a sticky ``native`` flag per calibration key and
    judged a subsequent decision against a looser threshold once native had
    been chosen. Two independent estimators asking about the same shape then
    got different answers depending on what a third had asked earlier in the
    process -- state no caller can see, control, or reproduce.
    """

    def _host(self) -> tuple[SimulatedHost, FakeClock]:
        clock = FakeClock()
        # native/NumPy = 0.0375 * samples**0.3: better than NumPy on small
        # batches, worse on very large ones, so shapes exist on both sides of
        # the margin and in the region a hold band used to cover.
        return (
            SimulatedHost(
                clock,
                PowerLaw(coefficient=1e-9, samples_power=1.0),
                PowerLaw(coefficient=3.75e-11, samples_power=1.3),
            ),
            clock,
        )

    #: 8,192 clears the entry margin; 21,800 sits in what used to be the hold
    #: band; 56,400 is a clear loss.
    ENTER, BETWEEN, LOSS = 8_192, 21_800, 56_400

    def test_a_clear_win_selects_native(self) -> None:
        host, clock = self._host()
        self.assertEqual(decide(host, clock, self.ENTER, 18).backend, "native_cpu")

    def test_the_middle_shape_resolves_to_numpy_on_a_fresh_cache(self) -> None:
        host, clock = self._host()
        self.assertEqual(decide(host, clock, self.BETWEEN, 18).backend, "numpy")

    def test_a_previous_native_choice_does_not_carry_the_middle_shape(self) -> None:
        host, clock = self._host()
        cache = CpuDispatchCache()
        self.assertEqual(decide(host, clock, self.ENTER, 18, cache=cache).backend, "native_cpu")
        followed = decide(host, clock, self.BETWEEN, 18, cache=cache)
        self.assertEqual(followed.backend, "numpy")
        self.assertIn("short of the", followed.reason)

    def test_the_answer_does_not_depend_on_the_order_shapes_are_asked_in(self) -> None:
        shapes = (self.ENTER, self.BETWEEN, self.LOSS)
        answers: list[tuple[tuple[int, str], ...]] = []
        for order in permutations(shapes):
            host, clock = self._host()
            cache = CpuDispatchCache()
            answers.append(
                tuple(
                    sorted(
                        (samples, decide(host, clock, samples, 18, cache=cache).backend)
                        for samples in order
                    )
                )
            )
        self.assertEqual(len(set(answers)), 1, f"order changed the outcome: {set(answers)}")

    def test_repeating_one_question_repeats_one_answer(self) -> None:
        host, clock = self._host()
        cache = CpuDispatchCache()
        first = decide(host, clock, self.ENTER, 18, cache=cache)
        for _ in range(5):
            repeat = decide(host, clock, self.ENTER, 18, cache=cache)
            self.assertEqual(repeat.backend, first.backend)
            self.assertEqual(repeat.reason, first.reason)

    def test_no_selection_state_is_kept_anywhere(self) -> None:
        host, clock = self._host()
        cache = CpuDispatchCache()
        decide(host, clock, self.ENTER, 18, cache=cache)
        outcome = next(iter(cache.snapshot().values()))
        self.assertFalse(hasattr(outcome, "sticky_native"))
        self.assertFalse(hasattr(cache, "record"))
        self.assertFalse(hasattr(cache, "sticky_native"))
        self.assertFalse(hasattr(DispatchPolicy(), "hold_slack"))

    def test_every_native_choice_clears_the_entry_margin(self) -> None:
        # Mutation guard: the only threshold a native answer may be judged
        # against is 1 - enter_margin, whatever happened before.
        host, clock = self._host()
        policy = unconstrained_policy()
        cache = CpuDispatchCache()
        limit = math.log1p(-policy.enter_margin)
        for samples in (self.ENTER, self.BETWEEN, self.LOSS, self.ENTER):
            decision = decide(host, clock, samples, 18, policy=policy, cache=cache)
            if decision.backend == "native_cpu":
                self.assertLess(math.log(decision.ratio_upper_bound or 9.0), limit)


class CalibrationBudgetTests(unittest.TestCase):
    """The deadline is soft; the ladder is the hard bound.

    Stating it the other way round -- "a 0.25 second hard cap" -- would be a
    promise the implementation does not keep: the clock is consulted between
    probes, never during one, so a probe that starts inside the deadline runs
    to completion however long it takes. What genuinely cannot be exceeded is
    the ladder's fixed work, which is decided before anything is measured.
    """

    def _flat_host(self, seconds: float) -> tuple[SimulatedHost, FakeClock]:
        clock = FakeClock()
        engine = PowerLaw(coefficient=seconds, samples_power=0.0, parameters_power=0.0)
        return SimulatedHost(clock, engine, engine), clock

    def test_the_deadline_stops_the_ladder_and_refuses_to_dispatch(self) -> None:
        host, clock = self._flat_host(0.02)
        policy = unconstrained_policy(probe_start_deadline_seconds=0.25)
        decision = decide(host, clock, 100_000, 91, policy=policy)
        self.assertEqual(decision.backend, "numpy")
        self.assertIn("probe-start deadline", decision.reason)

    def test_a_probe_that_has_started_is_never_interrupted(self) -> None:
        # One probe costs far more than the whole deadline. The honest outcome
        # is that it still finishes and the calibration overruns; a test that
        # asserted the opposite would be asserting a hard cap that does not
        # exist.
        host, clock = self._flat_host(1.0)
        policy = unconstrained_policy(probe_start_deadline_seconds=0.25)
        decide(host, clock, 100_000, 91, policy=policy)
        expected = (policy.probe_warmups + policy.probe_repeats) * len(("numpy", "native_cpu"))
        self.assertEqual(clock.now, float(expected))
        self.assertGreater(clock.now, policy.probe_start_deadline_seconds)

    def test_the_hard_bound_is_the_ladder_not_the_clock(self) -> None:
        # However slow the host, the number of probe executions is fixed by the
        # policy alone.
        for seconds in (1e-9, 1.0, 1_000.0):
            with self.subTest(seconds=seconds):
                host, clock = self._flat_host(seconds)
                policy = unconstrained_policy(probe_start_deadline_seconds=1e9)
                decide(host, clock, 100_000, 91, policy=policy)
                expected = (
                    len(PROBE_LADDER)
                    * (policy.probe_warmups + policy.probe_repeats)
                    * len(("numpy", "native_cpu"))
                )
                self.assertEqual(len(host.runs), expected)

    def test_a_sufficient_deadline_completes_the_ladder(self) -> None:
        host, clock = self._flat_host(0.02)
        policy = unconstrained_policy(probe_start_deadline_seconds=5.0)
        decision = decide(host, clock, 100_000, 91, policy=policy)
        self.assertTrue(decision.calibrated)
        self.assertEqual(decision.probe_count, len(PROBE_LADDER))

    def test_a_partial_ladder_still_dispatches_when_it_identifies_the_model(self) -> None:
        host, clock = self._flat_host(0.02)
        # Each probe costs (1 warmup + 2 repeats) x 2 engines x 0.02s = 0.12s,
        # so probes start at 0.00, 0.12, 0.24 and 0.36. A 0.40s deadline admits
        # four of the five, which is the declared minimum for a fit.
        policy = unconstrained_policy(probe_start_deadline_seconds=0.40)
        decision = decide(host, clock, 100_000, 91, policy=policy)
        self.assertTrue(decision.calibrated)
        self.assertEqual(decision.probe_count, 4)

    def test_calibration_cost_is_reported_to_the_caller(self) -> None:
        host, clock = self._flat_host(0.01)
        decision = decide(host, clock, 100_000, 91)
        self.assertGreater(decision.calibration_seconds, 0.0)
        self.assertAlmostEqual(decision.calibration_seconds, clock.now)

    def test_the_shipped_policy_names_the_deadline_as_a_deadline(self) -> None:
        policy = DispatchPolicy()
        self.assertFalse(hasattr(policy, "calibration_budget_seconds"))
        self.assertEqual(policy.probe_start_deadline_seconds, 0.25)


class PairedTimingTests(unittest.TestCase):
    """Neither engine may be systematically measured first."""

    def test_rounds_alternate_which_engine_leads(self) -> None:
        self.assertEqual(cpu_dispatch.engine_order(0), ("numpy", "native_cpu"))
        self.assertEqual(cpu_dispatch.engine_order(1), ("native_cpu", "numpy"))
        self.assertEqual(cpu_dispatch.engine_order(2), ("numpy", "native_cpu"))

    def test_each_engine_leads_the_same_number_of_timed_rounds(self) -> None:
        host, clock = uniform_host(1.0)
        policy = unconstrained_policy()
        decide(host, clock, 100_000, 91, policy=policy)
        per_probe = (policy.probe_warmups + policy.probe_repeats) * 2
        leaders: list[str] = []
        for probe_index in range(len(PROBE_LADDER)):
            block = host.runs[probe_index * per_probe : (probe_index + 1) * per_probe]
            timed = block[policy.probe_warmups * 2 :]
            leaders.extend(engine for engine, _ in timed[::2])
        self.assertEqual(leaders.count("numpy"), leaders.count("native_cpu"))

    def test_a_first_touch_penalty_does_not_land_on_one_engine(self) -> None:
        # Mutation guard. This host is exactly fair -- both engines cost the
        # same -- except that whichever runs first on a probe pays a fixed
        # first-touch penalty. Timing NumPy's rounds before native's, as an
        # earlier revision did, hands that penalty to NumPy every time and
        # manufactures a native win out of nothing.
        clock = FakeClock()

        class FirstTouchHost(SimulatedHost):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                self.touched: set[Probe] = set()

            def run(self, engine: str, probe: Probe) -> None:
                self.runs.append((engine, probe))
                seconds = self.engines[engine].seconds(probe.samples, probe.parameters)
                if probe not in self.touched:
                    self.touched.add(probe)
                    seconds *= 4.0
                self.clock.advance(seconds)

        fair = PowerLaw(coefficient=1e-9)
        host = FirstTouchHost(clock, fair, fair)
        decision = decide(host, clock, 100_000, 91)
        self.assertEqual(
            decision.backend,
            "numpy",
            "a first-touch penalty must not be readable as a native speedup",
        )
        self.assertGreaterEqual(decision.predicted_ratio or 0.0, 0.999)


class CalibrationFailureTests(unittest.TestCase):
    def test_an_unavailable_extension_is_recorded_not_raised(self) -> None:
        clock = FakeClock()
        host = SimulatedHost(
            clock,
            PowerLaw(coefficient=1e-9),
            PowerLaw(coefficient=1e-12),
            unavailable="the native CPU engine is unusable on this host: no wheel",
        )
        decision = decide(host, clock, 100_000, 91)
        self.assertEqual(decision.backend, "numpy")
        self.assertIn("no wheel", decision.reason)
        self.assertFalse(decision.calibrated)

    def test_a_raising_setup_is_recorded_not_raised(self) -> None:
        clock = FakeClock()
        host = SimulatedHost(
            clock,
            PowerLaw(coefficient=1e-9),
            PowerLaw(coefficient=1e-12),
            prepare_error=OSError("libgomp.so.1: cannot open shared object file"),
        )
        decision = decide(host, clock, 100_000, 91)
        self.assertEqual(decision.backend, "numpy")
        self.assertIn("probe setup raised OSError", decision.reason)

    def test_a_raising_probe_is_recorded_not_raised(self) -> None:
        clock = FakeClock()
        host = SimulatedHost(
            clock,
            PowerLaw(coefficient=1e-9),
            PowerLaw(coefficient=1e-12),
            failing_engine="native_cpu",
        )
        decision = decide(host, clock, 100_000, 91)
        self.assertEqual(decision.backend, "numpy")
        self.assertIn("raised RuntimeError", decision.reason)

    def test_a_clock_too_coarse_to_resolve_a_probe_refuses_to_dispatch(self) -> None:
        clock = FakeClock()
        free = PowerLaw(coefficient=0.0)
        host = SimulatedHost(clock, free, free)
        decision = decide(host, clock, 100_000, 91)
        self.assertEqual(decision.backend, "numpy")
        self.assertIn("produced usable timings", decision.reason)

    def test_a_degenerate_ladder_identifies_nothing(self) -> None:
        host, clock = uniform_host(0.1)
        policy = unconstrained_policy(
            probe_ladder=(Probe(2048, 16),) * 4,
            minimum_probes=4,
        )
        decision = decide(host, clock, 100_000, 91, policy=policy)
        self.assertEqual(decision.backend, "numpy")
        self.assertIn("do not identify a host cost model", decision.reason)

    def test_a_failure_is_cached_so_it_is_paid_for_once(self) -> None:
        clock = FakeClock()
        host = SimulatedHost(
            clock,
            PowerLaw(coefficient=1e-9),
            PowerLaw(coefficient=1e-12),
            unavailable="no extension",
        )
        cache = CpuDispatchCache()
        for _ in range(4):
            self.assertEqual(decide(host, clock, 100_000, 91, cache=cache).backend, "numpy")
        self.assertEqual(host.prepare_calls, 1)


class CacheTests(unittest.TestCase):
    def test_concurrent_callers_calibrate_exactly_once(self) -> None:
        host, clock = uniform_host(0.25)
        cache = CpuDispatchCache()
        threads_count = 8
        barrier = threading.Barrier(threads_count)
        results: list[str] = []
        results_lock = threading.Lock()

        def worker() -> None:
            barrier.wait()
            decision = decide(host, clock, 100_000, 91, cache=cache)
            with results_lock:
                results.append(decision.backend)

        threads = [threading.Thread(target=worker) for _ in range(threads_count)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(len(results), threads_count)
        self.assertEqual(set(results), {"native_cpu"})
        self.assertEqual(host.prepare_calls, 1)

    def test_different_keys_are_calibrated_separately(self) -> None:
        host, clock = uniform_host(0.25)
        cache = CpuDispatchCache()
        for penalty in ("none", "l1"):
            select_cpu_backend(
                samples=100_000,
                parameters=91,
                dtype="float64",
                penalty=penalty,
                policy=unconstrained_policy(),
                cache=cache,
                runner=host,
                clock=clock,
            )
        self.assertEqual(host.prepare_calls, 2)
        self.assertEqual(len(cache.snapshot()), 2)

    def test_thread_count_is_part_of_the_key(self) -> None:
        host, clock = uniform_host(0.25)
        cache = CpuDispatchCache()
        for threads in (None, 1, 8):
            select_cpu_backend(
                samples=100_000,
                parameters=91,
                dtype="float64",
                penalty="none",
                n_threads=threads,
                policy=unconstrained_policy(),
                cache=cache,
                runner=host,
                clock=clock,
            )
        self.assertEqual(host.prepare_calls, 3)

    def test_clearing_the_cache_forces_a_fresh_measurement(self) -> None:
        host, clock = uniform_host(0.25)
        cache = CpuDispatchCache()
        decide(host, clock, 100_000, 91, cache=cache)
        cache.clear()
        self.assertEqual(cache.snapshot(), {})
        decide(host, clock, 100_000, 91, cache=cache)
        self.assertEqual(host.prepare_calls, 2)

    def test_the_module_cache_can_be_reset(self) -> None:
        cpu_dispatch.reset_cpu_dispatch_cache()
        self.assertEqual(cpu_dispatch._CACHE.snapshot(), {})

    def test_a_fork_discards_measurements_and_rebuilds_locks(self) -> None:
        # A child is frequently not running under the parent's execution
        # context -- a joblib worker may be pinned to a subset of cores -- so
        # inheriting the parent's timings is inheriting a wrong answer. The
        # inherited lock is the second, independent hazard: its holder does not
        # exist in the child.
        host, clock = uniform_host(0.25)
        cache = CpuDispatchCache()
        decide(host, clock, 100_000, 91, cache=cache)
        self.assertEqual(len(cache.snapshot()), 1)

        cache._lock.acquire()
        cache.invalidate_after_fork()
        self.assertEqual(cache.snapshot(), {}, "a child must not reuse the parent's measurements")
        self.assertEqual(decide(host, clock, 100_000, 91, cache=cache).backend, "native_cpu")
        self.assertEqual(host.prepare_calls, 2, "the child must re-measure, not deadlock")

    def test_the_process_cache_is_registered_for_fork_invalidation(self) -> None:
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertIn("register_at_fork(after_in_child=_CACHE.invalidate_after_fork)", source)
        self.assertTrue(hasattr(cpu_dispatch._CACHE, "invalidate_after_fork"))
        self.assertFalse(hasattr(cpu_dispatch._CACHE, "reinitialise_locks"))


class RuntimeSignatureTests(unittest.TestCase):
    """A measurement is only valid for the execution context it was taken in."""

    def setUp(self) -> None:
        cpu_dispatch.reset_cpu_dispatch_cache()
        self.addCleanup(cpu_dispatch.reset_cpu_dispatch_cache)

    def test_the_signature_names_no_processor(self) -> None:
        signature = cpu_dispatch.current_runtime_signature()
        rendered = repr(signature).lower()
        for forbidden in ("intel", "amd", "ryzen", "apple", "x86", "aarch", "genuine"):
            self.assertNotIn(forbidden, rendered)
        self.assertIn(signature.source, {"sched_getaffinity", "cpu_count", "unavailable"})

    def test_the_signature_reports_usable_cpus_not_installed_ones(self) -> None:
        signature = cpu_dispatch.current_runtime_signature()
        if signature.source == "unavailable":
            self.assertIsNone(signature.usable_cpus)
        else:
            self.assertIsInstance(signature.usable_cpus, int)
            self.assertGreaterEqual(signature.usable_cpus or 0, 1)

    def test_the_affinity_mask_itself_is_recorded_where_it_exists(self) -> None:
        with mock.patch.object(
            cpu_dispatch.os, "sched_getaffinity", return_value={5, 1, 3}, create=True
        ):
            signature = cpu_dispatch.current_runtime_signature()
        self.assertEqual(signature.affinity, (1, 3, 5), "the mask must be sorted and complete")
        self.assertEqual(signature.usable_cpus, 3)
        self.assertEqual(signature.source, "sched_getaffinity")

    def test_two_equal_sized_pinnings_are_different_contexts(self) -> None:
        # Mutation guard for the defect this replaced: recording only the
        # length made {0,1,2,3} and {8,9,10,11} the same context, so a process
        # moved to another socket answered from the old socket's measurement.
        # Cache sharing, memory locality and turbo behaviour all differ there.
        with mock.patch.object(
            cpu_dispatch.os, "sched_getaffinity", return_value={0, 1, 2, 3}, create=True
        ):
            first = cpu_dispatch.current_runtime_signature()
        with mock.patch.object(
            cpu_dispatch.os, "sched_getaffinity", return_value={8, 9, 10, 11}, create=True
        ):
            second = cpu_dispatch.current_runtime_signature()
        self.assertEqual(first.usable_cpus, second.usable_cpus)
        self.assertNotEqual(first.affinity, second.affinity)
        self.assertNotEqual(first, second)

    def test_a_repinning_of_the_same_size_forces_a_re_measurement(self) -> None:
        host, clock = uniform_host(0.25)
        cache = CpuDispatchCache()
        with self._pinned_to((0, 1, 2, 3)):
            decide(host, clock, 100_000, 91, cache=cache)
            decide(host, clock, 100_000, 91, cache=cache)
            self.assertEqual(host.prepare_calls, 1)
        with self._pinned_to((8, 9, 10, 11)):
            decide(host, clock, 100_000, 91, cache=cache)
        self.assertEqual(
            host.prepare_calls,
            2,
            "a process moved to a different set of four CPUs must re-measure",
        )
        self.assertEqual(len(cache.snapshot()), 1, "and the old entry must be dropped")

    def test_missing_affinity_introspection_degrades_to_a_named_fallback(self) -> None:
        with mock.patch.object(cpu_dispatch.os, "sched_getaffinity", None, create=True):
            signature = cpu_dispatch.current_runtime_signature()
        self.assertEqual(signature.source, "cpu_count")
        self.assertEqual(signature.usable_cpus, cpu_dispatch.os.cpu_count())
        self.assertIsNone(signature.affinity, "a count is not a mask and must not pretend to be")

    def test_no_introspection_at_all_is_recorded_rather_than_guessed(self) -> None:
        with (
            mock.patch.object(cpu_dispatch.os, "sched_getaffinity", None, create=True),
            mock.patch.object(cpu_dispatch.os, "cpu_count", return_value=None),
        ):
            signature = cpu_dispatch.current_runtime_signature()
        self.assertEqual(signature.source, "unavailable")
        self.assertIsNone(signature.usable_cpus)
        self.assertIsNone(signature.affinity)
        self.assertIn("unknown", signature.describe())

    def test_an_unreadable_affinity_mask_falls_back_instead_of_raising(self) -> None:
        with mock.patch.object(
            cpu_dispatch.os, "sched_getaffinity", side_effect=OSError("not permitted"), create=True
        ):
            signature = cpu_dispatch.current_runtime_signature()
        self.assertEqual(signature.source, "cpu_count")
        self.assertIsNone(signature.affinity)

    def test_the_thread_environment_is_part_of_the_signature(self) -> None:
        with mock.patch.dict(cpu_dispatch.os.environ, {"OMP_NUM_THREADS": "1"}, clear=False):
            one = cpu_dispatch.current_runtime_signature()
        with mock.patch.dict(cpu_dispatch.os.environ, {"OMP_NUM_THREADS": "8"}, clear=False):
            eight = cpu_dispatch.current_runtime_signature()
        self.assertNotEqual(one, eight)

    def test_every_thread_variable_is_observed(self) -> None:
        base = dict.fromkeys(cpu_dispatch.THREAD_ENVIRONMENT_KEYS, "2")
        with mock.patch.dict(cpu_dispatch.os.environ, base, clear=False):
            reference = cpu_dispatch.current_runtime_signature()
            for name in cpu_dispatch.THREAD_ENVIRONMENT_KEYS:
                with self.subTest(variable=name):
                    with mock.patch.dict(cpu_dispatch.os.environ, {name: "3"}, clear=False):
                        self.assertNotEqual(cpu_dispatch.current_runtime_signature(), reference)

    def test_effective_threadpool_limits_are_part_of_the_signature(self) -> None:
        def pool(threads: int) -> list[dict[str, object]]:
            return [
                {
                    "user_api": "blas",
                    "internal_api": "openblas",
                    "prefix": "libopenblas",
                    "num_threads": threads,
                    "filepath": f"/machine-specific/{threads}",
                    "architecture": "forbidden-identity",
                }
            ]

        with mock.patch.object(cpu_dispatch, "_threadpool_info", return_value=pool(1)):
            one = cpu_dispatch.current_runtime_signature()
        with mock.patch.object(cpu_dispatch, "_threadpool_info", return_value=pool(8)):
            eight = cpu_dispatch.current_runtime_signature()
        self.assertNotEqual(one, eight)
        self.assertEqual(one.threadpools, (("blas", "openblas", "libopenblas", 1),))
        self.assertNotIn("machine-specific", repr(one))
        self.assertNotIn("forbidden-identity", repr(one))

    def test_threadpool_introspection_is_optional_and_failure_safe(self) -> None:
        with mock.patch.object(cpu_dispatch, "_threadpool_info", None):
            absent = cpu_dispatch.current_runtime_signature()
        with mock.patch.object(
            cpu_dispatch, "_threadpool_info", side_effect=RuntimeError("broken helper")
        ):
            failed = cpu_dispatch.current_runtime_signature()
        self.assertEqual(absent.threadpools, ())
        self.assertEqual(absent.threadpool_source, "unavailable")
        self.assertEqual(failed.threadpools, ())
        self.assertEqual(failed.threadpool_source, "error")

    def test_a_changed_effective_blas_pool_invalidates_the_cache(self) -> None:
        host, clock = uniform_host(0.25)
        cache = CpuDispatchCache()

        def info(threads: int) -> list[dict[str, object]]:
            return [{"user_api": "blas", "internal_api": "mkl", "num_threads": threads}]

        with mock.patch.object(cpu_dispatch, "_threadpool_info", return_value=info(1)):
            decide(host, clock, 100_000, 91, cache=cache)
            decide(host, clock, 100_000, 91, cache=cache)
        with mock.patch.object(cpu_dispatch, "_threadpool_info", return_value=info(8)):
            decide(host, clock, 100_000, 91, cache=cache)
        self.assertEqual(host.prepare_calls, 2)
        self.assertEqual(len(cache.snapshot()), 1)

    @staticmethod
    def _pinned_to(mask: tuple[int, ...]) -> Any:
        """Pretend the process is pinned to exactly these CPUs."""

        return mock.patch.object(
            cpu_dispatch, "_cpu_context", return_value=(mask, len(mask), "sched_getaffinity")
        )

    def test_a_changed_affinity_invalidates_the_cached_measurement(self) -> None:
        host, clock = uniform_host(0.25)
        cache = CpuDispatchCache()
        with self._pinned_to(tuple(range(16))):
            decide(host, clock, 100_000, 91, cache=cache)
            decide(host, clock, 100_000, 91, cache=cache)
            self.assertEqual(host.prepare_calls, 1)
        with self._pinned_to((0, 1)):
            decide(host, clock, 100_000, 91, cache=cache)
        self.assertEqual(host.prepare_calls, 2, "a re-pinned process must re-measure")

    def test_a_stale_measurement_is_dropped_not_merely_missed(self) -> None:
        host, clock = uniform_host(0.25)
        cache = CpuDispatchCache()
        with self._pinned_to(tuple(range(16))):
            decide(host, clock, 100_000, 91, cache=cache)
        for index, count in enumerate((8, 4, 2), start=1):
            with self._pinned_to(tuple(range(count))):
                decide(host, clock, 100_000, 91, cache=cache)
            self.assertEqual(
                len(cache.snapshot()),
                1,
                f"context change {index} left an entry that can never be valid again",
            )


class NativeThreadIntrospectionTests(unittest.TestCase):
    """The extension's own pool size must stay out of the signature.

    It looked like a useful context input and is in fact a trap. It is unknown
    until something imports the extension, and the thing that imports it is
    calibration -- so a first calibration would run under signature A, change
    the signature to B on its way out, and be re-run in full the next time
    anything asked. The requested pool is already ``CalibrationKey.n_threads``
    and the pool it actually gets is governed by the affinity mask and
    ``RAYON_NUM_THREADS``, both of which *are* recorded.
    """

    def setUp(self) -> None:
        cpu_dispatch.reset_cpu_dispatch_cache()
        self.addCleanup(cpu_dispatch.reset_cpu_dispatch_cache)

    def test_the_signature_has_no_native_thread_field(self) -> None:
        signature = cpu_dispatch.current_runtime_signature()
        self.assertFalse(hasattr(signature, "native_threads"))
        self.assertFalse(hasattr(cpu_dispatch, "_native_thread_count"))

    def test_loading_the_extension_does_not_change_the_signature(self) -> None:
        with mock.patch.dict(sys.modules, {}, clear=False):
            sys.modules.pop("renewable_huber._native_cpu", None)
            without = cpu_dispatch.current_runtime_signature()
        fake = types.SimpleNamespace(version=lambda: {"parallel_threads": 24})
        with mock.patch.dict(sys.modules, {"renewable_huber._native_cpu": fake}):
            with_extension = cpu_dispatch.current_runtime_signature()
        self.assertEqual(without, with_extension)

    def test_a_different_reported_pool_size_does_not_change_the_signature(self) -> None:
        signatures = set()
        for threads in (1, 4, 24, 128):
            fake = types.SimpleNamespace(
                version=lambda threads=threads: {"parallel_threads": threads}
            )
            with mock.patch.dict(sys.modules, {"renewable_huber._native_cpu": fake}):
                signatures.add(cpu_dispatch.current_runtime_signature())
        self.assertEqual(len(signatures), 1)

    def test_calibrating_does_not_invalidate_its_own_measurement(self) -> None:
        # Mutation guard, and the defect itself: with the extension's pool size
        # in the signature, the act of calibrating changes the key, so the very
        # next question re-measures from scratch.
        host, clock = uniform_host(0.25)
        cache = CpuDispatchCache()

        class LoadsTheExtension(SimulatedHost):
            """A runner that imports the extension, the way the real one does."""

            def prepare(self, key: cpu_dispatch.CalibrationKey) -> str | None:
                sys.modules["renewable_huber._native_cpu"] = types.SimpleNamespace(
                    version=lambda: {"parallel_threads": 24}
                )
                return super().prepare(key)

        loading = LoadsTheExtension(clock, host.engines["numpy"], host.engines["native_cpu"])
        with mock.patch.dict(sys.modules, {}, clear=False):
            sys.modules.pop("renewable_huber._native_cpu", None)
            decide(loading, clock, 100_000, 91, cache=cache)
            decide(loading, clock, 100_000, 91, cache=cache)
        self.assertEqual(
            loading.prepare_calls,
            1,
            "calibration must not invalidate the entry it just created",
        )

    def test_the_module_does_not_inspect_loaded_modules_at_all(self) -> None:
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertNotIn("sys.modules", source)
        self.assertNotIn("parallel_threads", source)


class PolicyInputTests(unittest.TestCase):
    """The inputs the policy is forbidden to use, checked against the source."""

    def _tree(self) -> ast.Module:
        return ast.parse(MODULE_PATH.read_text(encoding="utf-8"))

    def test_no_processor_identity_is_consulted(self) -> None:
        source = MODULE_PATH.read_text(encoding="utf-8").lower()
        for forbidden in (
            "platform.",
            "cpuinfo",
            "/proc/cpuinfo",
            "uname",
            "processor()",
            "brand_raw",
            "machine()",
        ):
            self.assertNotIn(forbidden, source, f"{forbidden!r} would key the policy on the CPU")

    def _imported_modules(self) -> set[str]:
        """Top-level names of every absolute import the module makes."""

        tree = self._tree()
        imported = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        return imported | {
            (node.module or "").split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.level == 0
        }

    def test_nothing_is_persisted(self) -> None:
        imported = self._imported_modules()
        for forbidden in ("json", "pathlib", "pickle", "shelve", "sqlite3", "tempfile", "shutil"):
            self.assertNotIn(forbidden, imported, "a persisted calibration is invalid elsewhere")
        self.assertNotIn("open(", MODULE_PATH.read_text(encoding="utf-8"))

    def test_os_is_used_only_for_fork_recovery_and_context_introspection(self) -> None:
        # Each of these is a count or a caller-set string. None of them can
        # return a brand, a model number or a micro-architecture name, which is
        # what makes this list a meaningful boundary rather than a formality.
        self.assertEqual(
            self._module_attributes("os"),
            {"register_at_fork", "sched_getaffinity", "cpu_count", "environ"},
        )

    def test_sys_is_not_consulted_at_all(self) -> None:
        # Reading `sys.modules` was how the extension's pool size reached the
        # runtime signature. That input is gone; the import should be too, so
        # reintroducing either is visible here.
        self.assertEqual(self._module_attributes("sys"), set())
        self.assertNotIn("sys", self._imported_modules())

    def test_the_attribute_scan_sees_getattr_as_well_as_dots(self) -> None:
        # Anti-vacuity: the module reaches `sched_getaffinity` through
        # `getattr(os, ...)` because it does not exist everywhere, so a scan
        # that only understood `os.x` would silently allow anything spelled
        # that way.
        module = ast.parse("import os\ngetattr(os, 'sched_setaffinity', None)\n")
        self.assertIn("sched_setaffinity", self._attributes_of(module, "os"))

    def _module_attributes(self, module_name: str) -> set[str]:
        return self._attributes_of(self._tree(), module_name)

    @staticmethod
    def _attributes_of(tree: ast.Module, module_name: str) -> set[str]:
        """Every attribute of ``module_name`` reached by a dot or by ``getattr``."""

        used: set[str] = set()
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == module_name
            ):
                used.add(node.attr)
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and node.args
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == module_name
                and len(node.args) > 1
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
            ):
                used.add(node.args[1].value)
        return used

    def test_the_parse_is_not_vacuous(self) -> None:
        names = {
            node.name
            for node in ast.walk(self._tree())
            if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        }
        self.assertIn("select_cpu_backend", names)
        self.assertIn("CpuDispatchCache", names)


class DispatchAppliesTests(unittest.TestCase):
    def test_only_auto_off_cuda_is_governed(self) -> None:
        self.assertTrue(auto_cpu_dispatch_applies("auto", "auto"))
        self.assertTrue(auto_cpu_dispatch_applies("auto", "cpu"))
        self.assertFalse(auto_cpu_dispatch_applies("auto", "cuda"))
        for name in ("numpy", "native_cpu", "cupy", "native_cuda", "torch", "tensorflow"):
            for device in ("auto", "cpu", "cuda"):
                self.assertFalse(auto_cpu_dispatch_applies(name, device))


class BackendSwapSafetyTests(unittest.TestCase):
    """Why replacing NumPy with the native CPU engine mid-``partial_fit`` is free.

    The estimator validates a batch on NumPy and only then learns its shape, so
    the dispatch decision necessarily happens after ``_prepare_batch``. That is
    sound exactly as long as both CPU engines see identical arrays. If the
    native adapter ever stops inheriting NumPy's conversions, or starts
    building its own design matrix, the swap silently changes what the solver
    receives -- with no test failing anywhere else.
    """

    def test_the_native_cpu_adapter_shares_numpy_array_handling(self) -> None:
        for name in ("asarray", "copy", "reshape", "to_numpy", "scalar"):
            self.assertIs(
                getattr(NativeCpuBackend, name),
                getattr(NumPyBackend, name),
                f"{name} must stay NumPy's, or a mid-stream swap changes the data",
            )
        self.assertIs(NativeCpuBackend.xp, NumPyBackend.xp)
        self.assertEqual(NativeCpuBackend.device, NumPyBackend.device)

    def test_the_native_cpu_adapter_does_not_build_its_own_design_matrix(self) -> None:
        self.assertFalse(hasattr(NativeCpuBackend, "native_design_matrix"))


class _StubBackend(NumPyBackend):
    """A NumPy backend wearing another backend's name."""

    def __init__(self, name: str, dtype: str = "float64") -> None:
        super().__init__(dtype)
        self.name = name


class _FailingNativeBackend(_StubBackend):
    """Fails on its first update, the way a wedged native engine does."""

    def __init__(self, dtype: str = "float64", error: Exception | None = None) -> None:
        super().__init__("native_cpu", dtype)
        self.calls = 0
        self.error = error or BackendUnavailableError("The native CPU engine could not initialize")

    def renewable_update(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        raise self.error


def _native_decision() -> cpu_dispatch.CpuDispatchDecision:
    return cpu_dispatch.CpuDispatchDecision(
        backend="native_cpu",
        reason="stubbed",
        work_units=1.0e9,
        calibrated=True,
        predicted_ratio=0.2,
        ratio_upper_bound=0.3,
    )


class EstimatorDispatchTests(unittest.TestCase):
    def setUp(self) -> None:
        cpu_dispatch.reset_cpu_dispatch_cache()
        self.addCleanup(cpu_dispatch.reset_cpu_dispatch_cache)
        rng = np.random.default_rng(11)
        self.X = rng.normal(size=(256, 4))
        self.y = self.X @ rng.normal(size=4) + rng.normal(scale=0.1, size=256)

    def test_a_small_auto_fit_stays_on_numpy_and_says_why(self) -> None:
        model = RenewableHuberRegressor().fit(self.X, self.y)
        self.assertEqual(model.backend_, "numpy")
        self.assertEqual(model.auto_dispatch_["backend"], "numpy")
        self.assertFalse(model.auto_dispatch_["calibrated"])
        self.assertEqual(model.auto_dispatch_["work_units"], 256 * 5**2)
        json.dumps(model.auto_dispatch_)

    def test_an_explicit_backend_never_consults_the_policy(self) -> None:
        for name in ("numpy", "native_cpu"):
            with self.subTest(backend=name):
                with (
                    mock.patch.object(
                        cpu_dispatch, "select_cpu_backend", side_effect=AssertionError
                    ),
                    mock.patch.object(
                        RenewableHuberRegressor, "_resolve_backend", return_value=NumPyBackend()
                    ),
                    mock.patch(
                        "renewable_huber.estimator.select_cpu_backend",
                        side_effect=AssertionError("policy consulted for an explicit backend"),
                    ),
                ):
                    model = RenewableHuberRegressor(backend=name).fit(self.X, self.y)
                self.assertFalse(hasattr(model, "auto_dispatch_"))

    def test_a_cuda_request_never_consults_the_policy(self) -> None:
        with (
            mock.patch.object(
                RenewableHuberRegressor, "_resolve_backend", return_value=_StubBackend("cupy")
            ),
            mock.patch(
                "renewable_huber.estimator.select_cpu_backend",
                side_effect=AssertionError("policy consulted for device='cuda'"),
            ),
        ):
            model = RenewableHuberRegressor(device="cuda").fit(self.X, self.y)
        self.assertEqual(model.backend_, "cupy")
        self.assertFalse(hasattr(model, "auto_dispatch_"))

    def test_the_policy_is_consulted_once_per_stream(self) -> None:
        selector = mock.Mock(
            return_value=cpu_dispatch.CpuDispatchDecision(
                backend="numpy", reason="stubbed", work_units=1.0, calibrated=False
            )
        )
        with mock.patch("renewable_huber.estimator.select_cpu_backend", selector):
            model = RenewableHuberRegressor()
            model.partial_fit(self.X, self.y)
            model.partial_fit(self.X, self.y)
            model.partial_fit(self.X, self.y)
        self.assertEqual(selector.call_count, 1)

    def test_the_policy_sees_the_design_width_not_the_feature_count(self) -> None:
        selector = mock.Mock(
            return_value=cpu_dispatch.CpuDispatchDecision(
                backend="numpy", reason="stubbed", work_units=1.0, calibrated=False
            )
        )
        with mock.patch("renewable_huber.estimator.select_cpu_backend", selector):
            RenewableHuberRegressor(fit_intercept=True).fit(self.X, self.y)
            RenewableHuberRegressor(fit_intercept=False).fit(self.X, self.y)
        widths = [call.kwargs["parameters"] for call in selector.call_args_list]
        self.assertEqual(widths, [5, 4])

    def test_a_native_selection_is_applied(self) -> None:
        resolved: list[str | None] = []

        def record(config: Any, *, name: str | None = None) -> Any:
            resolved.append(name)
            return _StubBackend(name or config.backend)

        with (
            mock.patch(
                "renewable_huber.estimator.select_cpu_backend", return_value=_native_decision()
            ),
            mock.patch.object(RenewableHuberRegressor, "_resolve_backend", staticmethod(record)),
        ):
            model = RenewableHuberRegressor().fit(self.X, self.y)
        self.assertEqual(resolved, [None, "native_cpu"])
        self.assertEqual(model.backend_, "native_cpu")
        self.assertEqual(model.auto_dispatch_["backend"], "native_cpu")

    def test_a_native_engine_that_cannot_be_built_falls_back_silently(self) -> None:
        def resolve(config: Any, *, name: str | None = None) -> Any:
            if name == "native_cpu":
                raise BackendUnavailableError("no wheel installed")
            return NumPyBackend(config.dtype)

        with (
            mock.patch(
                "renewable_huber.estimator.select_cpu_backend", return_value=_native_decision()
            ),
            mock.patch.object(RenewableHuberRegressor, "_resolve_backend", staticmethod(resolve)),
        ):
            model = RenewableHuberRegressor().fit(self.X, self.y)
        self.assertEqual(model.backend_, "numpy")
        self.assertEqual(model.auto_dispatch_["backend"], "numpy")
        self.assertIn("could not be constructed", model.auto_dispatch_["reason"])

    def test_a_native_engine_that_fails_its_first_update_falls_back_silently(self) -> None:
        failing = _FailingNativeBackend()
        reference = RenewableHuberRegressor(backend="numpy").fit(self.X, self.y)
        with (
            mock.patch(
                "renewable_huber.estimator.select_cpu_backend", return_value=_native_decision()
            ),
            mock.patch.object(RenewableHuberRegressor, "_resolve_backend", self._resolver(failing)),
        ):
            model = RenewableHuberRegressor().fit(self.X, self.y)
        self.assertEqual(failing.calls, 1)
        self.assertEqual(model.backend_, "numpy")
        self.assertIn("failed on its first update", model.auto_dispatch_["reason"])
        np.testing.assert_allclose(model.coef_, reference.coef_)
        self.assertEqual(model.n_iter_, reference.n_iter_)

    #: An auto-selected engine may fail in any of these ways. None of them is
    #: the caller's fault and none of them may reach the caller: they asked for
    #: ``auto``, not for this engine.
    ORDINARY_FAILURES = (
        BackendUnavailableError("The native CPU engine could not initialize"),
        RuntimeError("PanicException: called `Option::unwrap()` on a `None` value"),
        ValueError("invalid native configuration: tolerance must be finite and positive"),
        MemoryError("failed to allocate native workspace"),
        OSError("libgomp.so.1: cannot open shared object file"),
        TypeError("argument 'weights': 'NoneType' object cannot be converted"),
        OverflowError("Python int too large to convert to C long"),
        AttributeError("'builtins.NativeCpuEngine' object has no attribute 'update'"),
    )

    def _resolver(self, failing: Any) -> Any:
        def resolve(config: Any, *, name: str | None = None) -> Any:
            return failing if name == "native_cpu" else NumPyBackend(config.dtype)

        return staticmethod(resolve)

    def test_any_ordinary_failure_of_an_auto_selected_engine_falls_back(self) -> None:
        reference = RenewableHuberRegressor(backend="numpy").fit(self.X, self.y)
        for error in self.ORDINARY_FAILURES:
            with self.subTest(error=type(error).__name__):
                failing = _FailingNativeBackend(error=error)
                with (
                    mock.patch(
                        "renewable_huber.estimator.select_cpu_backend",
                        return_value=_native_decision(),
                    ),
                    mock.patch.object(
                        RenewableHuberRegressor, "_resolve_backend", self._resolver(failing)
                    ),
                ):
                    model = RenewableHuberRegressor().fit(self.X, self.y)
                self.assertEqual(failing.calls, 1)
                self.assertEqual(model.backend_, "numpy")
                self.assertIn(type(error).__name__, model.auto_dispatch_["reason"])
                np.testing.assert_allclose(model.coef_, reference.coef_)

    def test_a_construction_failure_of_any_kind_falls_back(self) -> None:
        for error in self.ORDINARY_FAILURES:
            with self.subTest(error=type(error).__name__):

                def resolve(
                    config: Any, *, name: str | None = None, error: Exception = error
                ) -> Any:
                    if name == "native_cpu":
                        raise error
                    return NumPyBackend(config.dtype)

                with (
                    mock.patch(
                        "renewable_huber.estimator.select_cpu_backend",
                        return_value=_native_decision(),
                    ),
                    mock.patch.object(
                        RenewableHuberRegressor, "_resolve_backend", staticmethod(resolve)
                    ),
                ):
                    model = RenewableHuberRegressor().fit(self.X, self.y)
                self.assertEqual(model.backend_, "numpy")
                self.assertIn("could not be constructed", model.auto_dispatch_["reason"])

    def test_an_explicit_native_failure_of_any_kind_is_still_raised(self) -> None:
        # The symmetric half. Collapsing these two would make an explicitly
        # requested engine silently run something else.
        for error in self.ORDINARY_FAILURES:
            with self.subTest(error=type(error).__name__):
                failing = _FailingNativeBackend(error=error)
                with mock.patch.object(
                    RenewableHuberRegressor,
                    "_resolve_backend",
                    staticmethod(lambda *a, **k: failing),
                ):
                    with self.assertRaises(type(error)):
                        RenewableHuberRegressor(backend="native_cpu").fit(self.X, self.y)

    def test_a_keyboard_interrupt_is_never_swallowed(self) -> None:
        failing = _FailingNativeBackend()
        failing.error = KeyboardInterrupt()  # type: ignore[assignment]
        with (
            mock.patch(
                "renewable_huber.estimator.select_cpu_backend", return_value=_native_decision()
            ),
            mock.patch.object(RenewableHuberRegressor, "_resolve_backend", self._resolver(failing)),
        ):
            with self.assertRaises(KeyboardInterrupt):
                RenewableHuberRegressor().fit(self.X, self.y)

    def test_a_fallback_that_also_fails_reports_the_numpy_error(self) -> None:
        # If the input is genuinely bad, retrying on NumPy must surface NumPy's
        # complaint rather than hide it behind the native one.
        failing = _FailingNativeBackend(error=RuntimeError("native gave up"))
        with (
            mock.patch(
                "renewable_huber.estimator.select_cpu_backend", return_value=_native_decision()
            ),
            mock.patch.object(RenewableHuberRegressor, "_resolve_backend", self._resolver(failing)),
            mock.patch.object(
                RenewableHuberRegressor, "_run_update", side_effect=ValueError("numpy said no")
            ),
        ):
            with self.assertRaises(ValueError) as caught:
                RenewableHuberRegressor().fit(self.X, self.y)
        self.assertEqual(str(caught.exception), "numpy said no")

    def test_reset_forgets_the_dispatch_record(self) -> None:
        model = RenewableHuberRegressor().fit(self.X, self.y)
        self.assertTrue(hasattr(model, "auto_dispatch_"))
        model.reset()
        self.assertFalse(hasattr(model, "auto_dispatch_"))

    def test_set_params_forgets_the_dispatch_record(self) -> None:
        model = RenewableHuberRegressor().fit(self.X, self.y)
        model.set_params(tau=2.0)
        self.assertFalse(hasattr(model, "auto_dispatch_"))

    def test_get_params_still_reports_auto(self) -> None:
        with mock.patch(
            "renewable_huber.estimator.select_cpu_backend", return_value=_native_decision()
        ):
            with mock.patch.object(
                RenewableHuberRegressor,
                "_resolve_backend",
                staticmethod(lambda config, *, name=None: _StubBackend(name or config.backend)),
            ):
                model = RenewableHuberRegressor().fit(self.X, self.y)
        self.assertEqual(model.get_params()["backend"], "auto")
        self.assertEqual(model.backend_, "native_cpu")


class CheckpointRestoreDispatchTests(unittest.TestCase):
    def setUp(self) -> None:
        cpu_dispatch.reset_cpu_dispatch_cache()
        self.addCleanup(cpu_dispatch.reset_cpu_dispatch_cache)
        rng = np.random.default_rng(5)
        self.X = rng.normal(size=(128, 3))
        self.y = self.X @ rng.normal(size=3) + rng.normal(scale=0.1, size=128)

    def test_restoring_a_checkpoint_never_calibrates(self) -> None:
        import tempfile

        model = RenewableHuberRegressor().fit(self.X, self.y)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            model.save(path)
            selector = mock.Mock(side_effect=AssertionError("load must not measure"))
            with mock.patch("renewable_huber.estimator.select_cpu_backend", selector):
                restored = RenewableHuberRegressor.load(path)
            self.assertEqual(restored.backend_, "numpy")
            self.assertFalse(hasattr(restored, "auto_dispatch_"))
            np.testing.assert_allclose(restored.coef_, model.coef_)

    def test_a_restored_stream_dispatches_on_its_next_batch(self) -> None:
        import tempfile

        model = RenewableHuberRegressor().fit(self.X, self.y)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            model.save(path)
            restored = RenewableHuberRegressor.load(path)
            selector = mock.Mock(
                return_value=cpu_dispatch.CpuDispatchDecision(
                    backend="numpy", reason="stubbed", work_units=1.0, calibrated=False
                )
            )
            with mock.patch("renewable_huber.estimator.select_cpu_backend", selector):
                restored.partial_fit(self.X, self.y)
                restored.partial_fit(self.X, self.y)
        self.assertEqual(selector.call_count, 1)

    def test_an_explicit_backend_override_on_load_is_not_governed(self) -> None:
        import tempfile

        model = RenewableHuberRegressor().fit(self.X, self.y)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            model.save(path)
            restored = RenewableHuberRegressor.load(path, backend="numpy")
            selector = mock.Mock(side_effect=AssertionError("explicit backend was dispatched"))
            with mock.patch("renewable_huber.estimator.select_cpu_backend", selector):
                restored.partial_fit(self.X, self.y)
        self.assertEqual(restored.backend_, "numpy")

    def test_the_checkpoint_still_records_auto(self) -> None:
        import tempfile

        model = RenewableHuberRegressor().fit(self.X, self.y)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            model.save(path)
            with np.load(path, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata"]))
        self.assertEqual(metadata["config"]["backend"], "auto")
        self.assertEqual(metadata["config"]["device"], "auto")
        self.assertNotIn("auto_dispatch", metadata, "the checkpoint format is unchanged")


class PolicyReplacementTests(unittest.TestCase):
    """``DispatchPolicy`` is data, so a caller can reason about a variant."""

    def test_a_replaced_threshold_changes_only_that_threshold(self) -> None:
        policy = replace(DispatchPolicy(), enter_margin=0.5)
        self.assertEqual(policy.enter_margin, 0.5)
        self.assertEqual(policy.uncertainty_multiplier, DispatchPolicy().uncertainty_multiplier)

    def test_a_stricter_margin_refuses_a_win_the_default_accepts(self) -> None:
        host, clock = uniform_host(0.6)
        lenient = decide(host, clock, 100_000, 91, policy=unconstrained_policy())
        strict = decide(host, clock, 100_000, 91, policy=unconstrained_policy(enter_margin=0.8))
        self.assertEqual(lenient.backend, "native_cpu")
        self.assertEqual(strict.backend, "numpy")


if __name__ == "__main__":
    unittest.main()
