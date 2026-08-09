# RFC: runtime CPU backend dispatch for `backend="auto"`

Status: accepted and implemented in PR #29 (`b29b394`).
Scope: `src/renewable_huber/backends/cpu_dispatch.py` and the estimator hook
that calls it. No numerical algorithm, native kernel, GPU path, checkpoint
format or public constructor parameter changes.

## 1. The problem

`backend="auto"` resolved to NumPy on CPU and CuPy on CUDA, with no other
input. That is safe, and it means the Rust CPU engine is only ever used by a
caller who already knows it wins.

Whether it wins is a property of the host, not of the shape:

| shape | fixed Ryzen 9900X runner, 24 threads | this WSL2 dev host |
|---|---:|---:|
| `wide` 16,384 x 256 float64 | native **19.7x faster** | native **2.8x slower** |
| `reference` 100,000 x 90 float64 | native 1.17x-1.68x faster (v2 baseline) | native **2.2x slower** |
| `latency-smoke` 2,048 x 8 float64 | -- | native **1.7x faster** |

(Left column: `docs/native-core-p1.md`. Right column: measured while writing
this, see §8.) Two hosts, opposite answers, same shapes. Any policy that
decides from the shape alone is wrong on one of them.

## 2. What was rejected, and why

**A CPU model lookup table.** Brand and model strings do not predict what
matters: the linked BLAS, its thread count, memory bandwidth, whether the
container is CPU-limited, whether another tenant is on the socket. A table
keyed on them is a guess dressed as data, and it mispredicts silently on every
processor nobody measured. The policy therefore reads no processor identity at
all; `tests/test_cpu_auto_dispatch.py::PolicyInputTests` fails if `platform.`,
`uname`, `cpuinfo` or a brand string appears in the module.

**A persisted exact-shape map.** A calibration file is valid only on the
machine that wrote it. Copy the image, move the container, relink NumPy, change
`OMP_NUM_THREADS`, and the file is silently wrong -- with no version to
invalidate. Nothing is written to disk; the same test asserts the module
imports no `json`, `pathlib`, `pickle`, `shelve`, `sqlite3` or `tempfile`, and
calls no `open()`.

The in-memory cache has the same problem in miniature, and §3.5 is how it is
answered.

**A shape-keyed cache.** It answers only shapes it has seen, and streaming
callers rarely repeat a shape exactly. The offline advisor,
`scripts/benchmarks/dispatch_policy.py`, is exactly this by design, which is
why it stayed offline. It is unchanged and remains the evidence tool for
promotion decisions: it audits a *recorded* measurement against one exact
workload, which is a different job from deciding at runtime for a shape nobody
measured. Sharing an abstraction between them would have meant giving the
runtime policy a schema-validated record format it has no use for.

**Running the caller's batch on both engines and keeping the winner.** This is
the obvious approach and it is the one the brief forbids: on a one-shot `fit`
of a million rows it doubles the cost of the only solve the caller asked for,
and the amount doubled is unbounded.

## 3. The policy

### 3.1 Features

Two, both normalised:

- `log(samples)` -- rows in the batch;
- `log(parameters)` -- design width, so the intercept column is included.

and one derived scalar used only for gating:

- `work_units = samples * parameters**2`, the weighted Gram accumulation. It
  dominates every other term in the update at any shape this policy will
  consider, and it is a pure count, so it is comparable across hosts, dtypes
  and engines. That comparability is what makes the budget argument in §4 a
  statement about work rather than about one machine's seconds.

### 3.2 What is fitted

Not a cost model per engine -- the *ratio*:

```
log(native_seconds / numpy_seconds) = b0 + b1 * (log n - n0) + b2 * (log p - p0)
```

centred on the probe design. Fitting the ratio rather than two absolute cost
curves is deliberate:

- the decision is a comparison, so the ratio is the only quantity that matters;
- everything common to both engines -- clock speed, thermal state, whether the
  benchmark ran while a test suite was compiling -- cancels;
- it is dimensionless, so the same coefficients mean the same thing on a
  workstation and in a throttled CI container.

`tests/...::SimulatedHostTests::test_only_the_ratio_matters_not_the_absolute_speed`
pins that: two simulated hosts 10,000x apart in absolute speed with identical
relative engines produce identical decisions.

### 3.3 Uncertainty allowance

Three coefficients are fitted from five probes, leaving two residual degrees of
freedom. For a target shape `x0`:

```
allowance = uncertainty_multiplier * s * sqrt(1 + x0' (X'X)^-1 x0)
```

**This is a conservative allowance, not a confidence interval, and the code and
docs no longer call it one.** An earlier draft described `2 * se` as a "95%
upper bound". That was wrong twice over: with two residual degrees of freedom
the normal quantile does not apply at all (the one-sided Student-t 95% point is
2.92, so 2.0 is *less* conservative than the claim), and the residual-scale
floor below usually dominates `s` anyway, which makes the interval's nominal
coverage meaningless. Either implement a real small-sample interval or stop
claiming one; this implementation stops claiming one.

- `s` is the residual standard error, floored at 0.05 log units (~5%). Five
  points can agree by luck; without the floor that reads as certainty.
- The `1 +` makes the allowance scale like a *prediction* error rather than an
  error on the fitted line: a future measurement carries its own noise.
- The leverage term `x0' (X'X)^-1 x0` grows quadratically with distance from
  the probe design. This is the whole extrapolation story: nothing special
  happens at the edge of the measured box, the required evidence simply gets
  larger the further out the question is asked. At 1,000x beyond the probes a
  uniform 0.70 ratio no longer clears the entry margin, and it is the same host
  (`test_far_extrapolation_must_beat_a_larger_margin`).

A hard rail refuses any shape more than `log(1024)` outside the measured box in
either axis. It exists for absurd inputs, not for ordinary extrapolation, and
in practice the leverage term binds first.

### 3.4 The decision

Native is chosen when the conservative upper estimate clears the margin:

```
predicted + allowance  <  log(1 - enter_margin)      enter_margin = 0.15
```

One threshold, applied identically every time. **There is no sticky state and
no hysteresis band.** An earlier draft kept a per-key `sticky_native` flag and
judged a later decision against a looser threshold once native had been chosen,
so that a stream near the crossover would not alternate engines. That was the
wrong trade: the decision is already made once per stream, so the band bought
almost nothing, and what it cost was determinism. Two independent estimators
asking about the same shape got different answers depending on what a third
estimator elsewhere in the process had asked earlier — state no caller can see,
set, or reproduce, and a genuine hazard under `GridSearchCV` or any other
harness that fits many models in one process.

`select_cpu_backend` is now a pure function of the batch shape, the policy and
the cached measurement. `DecisionIndependenceTests` pins that: asking the same
question twice gives the same answer and the same reason string, permuting the
order of three questions across a shared cache changes nothing, and every
native answer is checked against `1 - enter_margin` and nothing else.

Judging on the conservative upper estimate rather than the point estimate is
what makes every form of ignorance -- noisy probes, a mis-specified surface, a
distant shape -- push the same way: towards NumPy.

### 3.5 What a measurement is valid for

A timing describes a host *and the execution context it was taken in*. Pin the
process to two cores, or set `OMP_NUM_THREADS=1` between fits, and the ratio
between a threaded native engine and a threaded BLAS moves without anything
about the machine changing. A cache that ignored this would answer confidently
from a measurement that no longer describes anything.

The cache key therefore carries a `RuntimeSignature`:

| field | source | why |
|---|---|---|
| `affinity` | the sorted CPU ids from `os.sched_getaffinity(0)`, or `None` where the platform has no such call | the only one of these that notices a `taskset` or a cgroup quota |
| `usable_cpus` | the mask's length, else `os.cpu_count()` | the portable fallback when there is no mask to record |
| `thread_environment` | the seven `*_NUM_THREADS` variables that BLAS, OpenMP and Rayon read | a caller-set string, sampled not interpreted |
| `threadpools` | optional `threadpoolctl.threadpool_info()` | effective BLAS/OpenMP provider and thread count; catches scikit-learn/joblib limits that do not modify environment variables |
| `threadpool_source` | `threadpoolctl`, `unavailable`, or `error` | makes loss of optional introspection explicit without making it a base dependency |
| `source` | which of the two CPU probes answered, or `"unavailable"` | an unreadable mask degrades to a *named* fallback instead of being indistinguishable from a one-core machine |

The mask, not just its length. Four cores on one socket and four on another
are different machines for cache sharing, memory locality and turbo
behaviour, so recording `4` for both would let one be answered from the
other's measurement.

**The extension's own `parallel_threads` is deliberately not an input**, though
an earlier draft read it out of `sys.modules`. It is not independent — the
*requested* pool is already `CalibrationKey.n_threads`, and what the pool
actually gets is governed by the affinity mask and `RAYON_NUM_THREADS`, both
recorded above — and including it is actively harmful: the value is unknown
until something imports the extension, and the thing that imports it is
calibration. A first calibration would therefore run under signature A, leave
the process in signature B, and be paid for a second time by the next question.
`test_calibrating_does_not_invalidate_its_own_measurement` pins that.

None of it identifies a processor. Where introspection is missing —
`sched_getaffinity` does not exist on macOS or Windows, `cpu_count()` may
return `None` anywhere, the affinity call can raise `OSError` under a
restrictive sandbox — the signature records how far it got and carries on.
The optional thread-pool snapshot excludes file paths and architecture strings;
if `threadpoolctl` is absent or raises, fitting continues with the portable
affinity/environment signature. It is installed transitively in common
scikit-learn environments but remains unnecessary for a NumPy-only install.

A changed signature does not merely miss the cache: `discard_other_signatures`
deletes the stale entries, so a long-lived process that is repeatedly re-pinned
does not accumulate calibrations that can never be valid again.

**`fork` discards everything.** Two independent hazards, one fix. The child
inherits any mutex a parent thread happened to hold at the instant of the fork,
and that thread does not exist in the child, so an inherited lock is a deadlock
waiting to be taken. And a worker child — joblib, `multiprocessing` — is
routinely pinned to a subset of the parent's cores, so the parent's timings do
not describe it. `os.register_at_fork(after_in_child=...)` rebuilds the locks
*and* empties the map; the child re-measures only if it meets the calibration
gate, and only once.

## 4. Bounded cost

This is the constraint that shaped the rest.

**The probes never touch the caller's data.** They are a fixed ladder of five
small shapes -- a 2x2 factorial in `(log n, log p)` with a centre run, the
smallest design that identifies both slopes, stays well conditioned, and leaves
residual freedom for an honest `s`:

| probe | samples | parameters | work units |
|---|---:|---:|---:|
| 1 | 1,024 | 7 | 5.0e4 |
| 2 | 8,192 | 7 | 4.0e5 |
| 3 | 2,896 | 15 | 6.5e5 |
| 4 | 1,024 | 33 | 1.1e6 |
| 5 | 8,192 | 33 | 8.9e6 |

Each is run with `max_iter=3` and a tolerance small enough that both engines
run all three, once as an untimed warmup and twice timed, taking the minimum.
Total: **2.01e8 work units**, reported by `DispatchPolicy.ladder_work_units()`
and asserted against the gate by
`test_the_shipped_gate_amortises_the_shipped_ladder`.

**That total is the hard bound.** It is a product of policy constants —
probes x rounds x engines x iterations — fixed before anything is measured, so
no host, however slow, can make a calibration cost more work than that.
`test_the_hard_bound_is_the_ladder_not_the_clock` runs the same ladder against
simulated hosts a trillion-fold apart in speed and asserts the number of probe
executions is identical.

**Two work gates decide when it is spent, and one soft deadline stops it early:**

1. `minimum_work_units = 5.0e4` -- the smallest probe. Below it the model would
   be answering about a batch smaller than anything it measured, and an engine
   swap is not worth churning. NumPy, no probing, no import of the extension.
2. `calibration_work_units = 1.5e8` -- a first calibration is only started for a
   batch at least this large. `2.01e8 / 1.5e8 = 1.34`, so the ladder costs at
   most about **1.34 single-iteration equivalents of the batch that triggers
   it** -- roughly a quarter of one five-iteration fit, in work terms.
3. `probe_start_deadline_seconds = 0.25` -- a **soft** deadline, and named as
   one. The clock is consulted only *between* probes, cheapest first; a probe
   that has started always runs to completion, so a calibration can and does
   finish after this instant. Earlier drafts of this document called it a "hard
   wall-clock cap", which the implementation never was and could not cheaply
   become: interrupting a running BLAS call is not something this layer can do.
   What it genuinely provides is an early stop on a slow host — a partial
   ladder of four probes still fits a model, fewer refuses to dispatch.
   `test_a_probe_that_has_started_is_never_interrupted` asserts the overrun
   rather than pretending it away.

**And it is paid once.** The result is cached per
`(dtype, penalty, n_threads, runtime signature)` for as long as that context
holds, so the second estimator, and every batch after the first, pays three dot
products. That is also why the two size gates are separate: acquiring evidence
needs a batch big enough to amortise it, *using* evidence already in hand does
not, so a process that has calibrated will dispatch batches 3,000x smaller than
the one that paid.

### 4.1 Probe timing is paired

Within a probe, the two engines are timed as an interleaved paired comparison:
each round runs both back to back, and the rounds alternate which goes first,
so over `probe_repeats` rounds each engine holds each position the same number
of times.

This is not fastidiousness. An earlier draft timed NumPy's warmup and both
repeats, then native's — so on every probe, NumPy paid for the first touch of
that probe's arrays and the first spin-up of its thread pool, and native was
measured on a warm machine. That is a systematic bias, on every probe, in
native's favour: exactly the direction the policy is supposed to be
conservative about. `test_a_first_touch_penalty_does_not_land_on_one_engine`
builds a host where the two engines are identical except for a fixed
first-touch penalty and asserts the policy still answers NumPy; under the old
fixed ordering it answers native.

Measured on this host: **30-44 ms** per calibration, against a triggering batch
of ~30 ms and up. §8 has the end-to-end numbers.

## 5. Where the decision happens

The estimator validates a batch before it knows its shape, so `resolve_backend`
cannot make this call -- it has no workload. `resolve_backend("auto")` is
therefore unchanged and still returns NumPy on CPU.

`partial_fit` resolves NumPy, validates and prepares the batch, and *then* asks
the policy, replacing the backend if the answer is native. That substitution is
free because both CPU engines take, produce and store exactly the same host
NumPy arrays: `NativeCpuBackend` inherits `asarray`, `copy`, `reshape`,
`to_numpy`, `scalar` and `xp` from `NumPyBackend` unchanged, and declares no
`native_design_matrix`, so the prepared design matrix and empty state are
identical either way. `BackendSwapSafetyTests` pins all of that, because if the
native adapter ever stops inheriting them the swap silently changes what the
solver receives and nothing else in the suite notices.

The decision is made **once per stream** -- on the first batch, or on the first
batch after a checkpoint restore -- so an engine is never swapped mid-stream.
Restoring a checkpoint alone never calibrates: there is no batch shape yet.

Failures of an auto-selected engine are absorbed rather than raised, because a
backend the caller did not ask for must not be able to fail their `fit`. This
covers **any ordinary exception** — not only `BackendUnavailableError` — at
either of two points:

- the native backend cannot be constructed after calibration;
- it raises on the first update after being chosen.

An earlier draft caught only `BackendUnavailableError`, which meant a
`PanicException` surfacing as `RuntimeError`, a `MemoryError` from a workspace
allocation, an `OSError` from a missing `libgomp`, or a `TypeError` from a PyO3
signature mismatch all reached a caller who had asked for `auto` and would have
been fine on NumPy. The catch is `Exception`, so `KeyboardInterrupt` and
`SystemExit` still propagate.

Both paths fall back to NumPy, name the exception type in
`auto_dispatch_["reason"]`, and produce the same coefficients a NumPy fit
would. Retrying is safe because nothing has been mutated yet — and if the input
is what is wrong, the NumPy attempt raises its own error, with the native one
attached as `__context__`.

An **explicitly** requested `native_cpu` that fails still raises, whatever it
raises: that is the caller's answer, and substituting another engine would hide
it. Both halves are exercised over the same eight exception types.

## 6. What the caller sees

A fitted estimator that went through the policy gains `auto_dispatch_`, a
JSON-compatible dict: chosen backend, the reason in words, batch work units,
whether a host model was used, predicted ratio, its upper bound, calibration
seconds and probe count. It is present only when the policy ran -- the same
convention `cuda_features_` already uses -- and is cleared by `reset()` and
`set_params()`.

`get_params()["backend"]` stays `"auto"`, and checkpoints continue to record
`"auto"`. The checkpoint format is untouched: a restored model re-decides on
its own host rather than inheriting a decision from the host that saved it.

## 7. Deliberate conservatism, and its cost

The policy will leave speed on the table, by construction:

- a one-shot `fit` below 1.5e8 work units never measures, so it never
  dispatches. On this host `latency-smoke` (2,048 x 8) is 1.7x faster on
  native and auto stays on NumPy. The absolute loss is 0.13 ms.
- a decision is made per stream on the first batch, so a stream of many small
  batches never accumulates its way into a calibration even though it would
  amortise one easily. Re-evaluating once a stream's *cumulative* work crosses
  the gate is the obvious extension; it was left out because it means swapping
  engines mid-stream, which is a larger contract change than this RFC.
- extrapolation demands a bigger margin, so a genuine 20% win a long way from
  the probes will be declined.

All three fail towards today's behaviour. The alternative failure -- auto
silently picking a slower engine, or spending unbounded time deciding -- is the
one worth ruling out.

## 8. Evidence

Deterministic evidence is in `tests/test_cpu_auto_dispatch.py` (102 tests): five
simulated hosts driven through an injected clock and probe runner, including
exact power-law hosts whose ratio the fitted surface must reproduce at shapes
no probe visited.

Host evidence is produced by `scripts/benchmarks/benchmark_auto_dispatch.py`,
which measures `numpy`, `native_cpu`, `auto_cold` (dispatch cache cleared
before every sample, so each pays a full ladder) and `auto_warm` (cache primed)
over the smoke and standard CPU shapes, and reports `auto_cold - auto_warm` as
the isolated calibration cost and `auto_warm / best explicit` as regret. Runs
alternate forward/reverse engine order and require an even repeat count; regret
is the median of aligned per-round ratios, not a ratio of independent medians.
Cold calibration runs in a separate phase so it cannot heat the CPU immediately
before a steady sample:

```bash
.venv/bin/python scripts/benchmarks/benchmark_auto_dispatch.py \
  --profile both --operation both --repeats 8 --warmup 1 \
  --output artifacts/auto-dispatch/auto-dispatch-cpu.json
```

**What a case measures.** The two operations follow the shape sweep's contract,
so a shape means the same work in both tools:

- `fit` is **one call over the whole dataset**. The batches are concatenated by
  the sweep's own `_fit_batch` before any clock starts, once per case, and the
  same arrays are handed to every engine and every repeat -- the estimator
  construction and the solve are timed, assembling the input is not.
- `stream` stays **per batch**: one `partial_fit` per batch, in order.

`summary.work_units` follows from that and is the batch the *policy* is asked
about: all `samples` for `fit`, and the first batch for `stream`, because a
stream decides on its first batch and never revisits. Both are counted at the
design width `features + 1`, so they agree with the `work_units` the estimator
itself reports in `auto_dispatch_`. `AutoDispatchBenchmarkContractTests` in
`tests/test_benchmark_performance_policy.py` pins all of it against a recording
estimator, because a harness that fits one batch while its header and its shape
record describe the whole dataset reports a wrong number instead of failing.

Captured after the contract fix on the WSL2 development host, `--profile both
--operation both --repeats 8 --warmup 1`, float64, `penalty="none"`. Engine
columns are medians in ms; regret is the median aligned ratio to the per-round
best explicit engine:

| case | numpy | native | auto (warm) | chose | regret | calibration |
|---|---:|---:|---:|---|---:|---:|
| latency-smoke 2,048x8 fit | 0.365 | **0.216** | 0.305 | numpy | 1.391 | -- |
| latency-smoke 2,048x8 stream | 0.388 | **0.183** | 0.421 | numpy | 2.177 | -- |
| reference-smoke 8,192x32 fit | **2.678** | 3.360 | 2.646 | numpy | 1.005 | -- |
| reference-smoke 8,192x32 stream | **2.062** | 2.653 | 2.145 | numpy | 1.028 | -- |
| latency 4,096x16 fit | 1.516 | **1.331** | 1.545 | numpy | 1.158 | -- |
| latency 4,096x16 stream | 1.535 | **1.501** | 1.521 | numpy | 1.021 | -- |
| reference 100,000x90 fit | **163.8** | 255.5 | 164.2 | numpy | 1.003 | 34.2 ms |
| reference 100,000x90 stream | **103.6** | 232.9 | 97.2 | numpy | 0.929 | 38.9 ms |
| wide 16,384x256 fit | **129.8** | 212.5 | 114.8 | numpy | 0.948 | 51.5 ms |
| wide 16,384x256 stream | **100.4** | 256.9 | 93.5 | numpy | 1.036 | 43.4 ms |
| streaming 1,000,000x32 fit | 557.7 | **489.7** | 541.3 | numpy | 1.122 | 32.9 ms |
| streaming 1,000,000x32 stream | **285.5** | 475.8 | 265.5 | numpy | 0.928 | -- |

Reading it:

- **No slower native engine was selected.** Native loses materially on the
  reference and wide workloads, and the calibrated upper estimate retained
  NumPy in every case.
- **Conservatism has measurable false negatives.** The one-million-row
  one-shot fit made native about 1.14x faster, but that is below the 15% entry
  margin and far beyond the row range of the probes; auto retained NumPy. The
  small native wins are below the calibration gate. These are missed speedups,
  not cases where auto selected an engine measured slower than NumPy.
- **Calibration is bounded and separable.** Reported calibration time was
  32.9-51.5 ms and is paid once per execution-context key.
- **Cold calibration no longer contaminates steady comparison.** It is measured
  in a separate phase. Where auto and explicit NumPy execute the same backend,
  aligned ratios are 0.928-1.036 on the standard cases, inside the host's
  documented noise band. The accepted JSON is retained under
  a local acceptance artifact under `artifacts/auto-dispatch/` (gitignored and
  intentionally not part of the repository), SHA-256
  `67960356bf1cb11b75cc7b632a8ec9beb82054f2ffd0704fe9af56d19883b28b`.

## 9. Follow-up

1. Re-evaluate the dispatch decision when a stream's cumulative work crosses
   the calibration gate (§7), which needs a decision about mid-stream engine
   changes.
2. The probe ladder is fixed. A host whose crossover sits far outside
   1,024-8,192 rows or 7-33 parameters is answered by extrapolation with a wide
   band; widening the ladder costs calibration time and is a tuning decision,
   not a design one.
3. `dispatch_policy.py` and this policy remain separate. If a third consumer
   appears, the shared abstraction to extract is "compare two timings under a
   stated margin", not the record schema.
