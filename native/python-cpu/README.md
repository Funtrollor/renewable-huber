# renewable-huber-native-cpu

Optional Rust/Rayon CPU engine for
[`renewable-huber`](https://github.com/Funtrollor/renewable-huber). Version
0.6.1 requires exactly `renewable-huber==0.6.1`; the base package owns the
public estimator API, validation and portable checkpoint format.

```bash
python -m pip install renewable-huber-native-cpu==0.6.1
```

Published wheels support CPython 3.10–3.12 on Windows x86-64,
manylinux2014 x86-64/aarch64 and macOS x86-64/Apple Silicon. Installing a
wheel does not require Rust.

```python
from renewable_huber import RenewableHuberRegressor

model = RenewableHuberRegressor(backend="native_cpu", n_jobs=-1)
model.fit(X, y)
```

`n_jobs=None` uses the shared Rayon pool, `-1` uses all available logical CPUs,
and a positive integer creates an estimator-local pool of that size. When an
outer joblib or GridSearchCV layer is parallel, use `n_jobs=1` to avoid nested
oversubscription.

CPU `backend="auto"` may select this engine for a sufficiently large workload
when a bounded, host-local runtime calibration demonstrates a conservative
advantage. It never reads a CPU model string or persists a machine-specific
crossover map. Explicit `backend="native_cpu"` never silently falls back.

Inputs are dense float32/float64 host arrays and predictions are NumPy arrays.
See the project [support matrix](https://github.com/Funtrollor/renewable-huber/blob/main/docs/support-matrix.md)
for the complete contract.
