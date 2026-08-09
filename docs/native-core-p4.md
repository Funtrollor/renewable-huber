# P4 native CUDA tuning

P4 keeps the strict numerical engine as the default and adds two explicit,
independent tuning switches for `backend="native_cuda"`:

```python
model = RenewableHuberRegressor(
    backend="native_cuda",
    device="cuda",
    dtype="float32",
    cuda_graphs=True,
    cuda_fast_math=True,
)
```

`cuda_graphs=True` captures the stable candidate-objective DAG for one native
update and replays it across Newton iterations and line-search trials. A graph
is scoped to that update, so captured pointers never outlive a borrowed DLPack
tensor and a new batch shape/configuration cannot replay stale arguments.
Capture is best effort. If the runtime or a library call rejects capture, the
engine clears its enabled graph flag, increments `graph_fallbacks`, and runs
the original stream path. State remains transactional: coefficients and the
information matrix are swapped into the active state only after successful
stream completion.

`cuda_fast_math=True` is valid only with `dtype="float32"`. It opts the main
cuBLAS handle into TF32 Tensor Core math. Objective reductions, convergence
norms, and the float64 path retain strict math. The accepted comparison against
strict float32 is `rtol=5e-3, atol=5e-4`; callers requiring reproducible strict
float32 results must leave the option disabled.

The C ABI remains version 1. Tuning flags use the previously reserved engine
option word, and `rh_cuda_engine_features` is an additive symbol. The Python
payload API is version 3 and reports `supports_cuda_graphs` and
`supports_fast_math`. After fitting, `model.cuda_features_` exposes requested
and enabled flags plus capture/replay/fallback counters. Both constructor
values participate in `get_params`, scikit-learn cloning, and checkpoints.

## RTX 5070 Ti historical development observation

The following local tuning run used a 32,768 x 256 float32 batch, three warmups,
nine samples, and at least 0.5 seconds of fixed work per statistical sample.
Its raw JSON was not committed, so these figures are historical engineering
context, not retained 0.6.1 release evidence or a current performance claim:

```powershell
python scripts/benchmarks/benchmark_cuda_tuning.py `
  --samples 32768 --features 256 --dtype float32 `
  --warmup 3 --repeats 9 --minimum-sample-seconds 0.5 `
  --output artifacts/p4-windows-rtx5070ti-cuda-tuning.json
```

On that development run, strict execution measured 19.416 ms, CUDA
Graph execution 17.021 ms (1.14x), and Graph+TF32 18.162 ms (1.07x versus
strict). The 1.07x TF32 difference is inside this host's approximately 10%
noise band and is not evidence of a speedup. Relative MAD was 0.79%, 3.85%, and 2.49%, respectively. Graph capture
recorded 4,522 replays with zero fallback. TF32 did not beat strict-precision
Graph for this shape, so it remains an experimental opt-in rather than an
automatic dispatch choice. Coefficient error versus strict was also recorded
in the JSON and stayed within the declared tolerance.

## Nsight reproduction

```powershell
.\scripts\profiling\run_nsight_systems.ps1 `
  -Python python `
  -Engine native_cuda -DType float32 -CudaGraphs

.\scripts\profiling\run_nsight_compute.ps1 `
  -Python python `
  -Engine native_cuda -DType float32 -CudaGraphs -CudaFastMath

python scripts/profiling/profile_cuda_update.py `
  --engine native_cuda --input-location host --dtype float32 `
  --cuda-graphs --cuda-fast-math `
  --metadata-output artifacts/p4-nsight-metadata.json
```

Nsight evidence is valid only when the JSON metadata, `.nsys-rep`, GPU,
driver, Toolkit, shape, transport, and tuning flags are retained together.
CUDA API/kernel/memcpy totals overlap and must not be added as wall time.

The uncommitted 2026-08-01 Nsight Systems capture for the same workload was
stored locally under `artifacts/nsight-p4`. All three profiled updates converged in seven iterations;
their median was 18.358 ms and the NVTX repeat window was 56.159 ms. Nsight
observed 3 graph instantiations and 84 graph launches. The report also makes
the remaining bottleneck explicit: 98 `cudaStreamSynchronize` calls consumed
27.451 ms of CUDA API time, so device-side convergence is still the next
architectural optimization. These API totals overlap GPU work and are evidence
for prioritization, not an additive wall-time model.
