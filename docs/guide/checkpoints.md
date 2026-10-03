# Checkpoints

A checkpoint is a compressed `.npz` archive written without pickle and read
back with `allow_pickle=False`. It holds the estimator's configuration and its
renewable summary state (the coefficients, the accumulated information matrix
and a handful of counters), never the historical `X` or `y`. The normative
list of stored fields is the state contract in [API](../api.md).

## Save and restore

```python
from renewable_huber import RenewableHuberRegressor

model = RenewableHuberRegressor(penalty="l1", lambda_scale=0.5)
for X_batch, y_batch in data_stream:
    model.partial_fit(X_batch, y_batch)
model.save("checkpoints/model.npz")

restored = RenewableHuberRegressor.load("checkpoints/model.npz")
restored.partial_fit(X_next, y_next)  # the stream continues where it stopped
```

`save` creates missing parent directories. A restored model continues the
same stream: its next `partial_fit` uses the saved coefficients and
information matrix exactly as the original model would have.

## Migrating between backends

--8<-- "README.en.md:checkpoint-migration"

The arrays are stored as NumPy data, but the configuration keeps the original
`backend`, `device` and `dtype`, and `load(path)` without overrides rebuilds
that same backend rather than silently downgrading. Restoring a CuPy, PyTorch
or TensorFlow checkpoint as-is therefore needs the same optional dependency,
and a CUDA configuration needs a usable GPU; overriding the target, as above,
lets a GPU checkpoint be migrated to NumPy on a machine without any GPU extra.
Overriding only `backend` resets `device` to `"auto"` instead of carrying an
incompatible CUDA setting forward.

## Formats and reproducibility

- Version 2 checkpoints store the accumulated sample weight and DataFrame
  column names. Version 1 checkpoints still load, and treat every row as unit
  weight.
- L1 checkpoints resume across the NumPy, Rust CPU and native CUDA engines
  (native CUDA runs L1 from 0.7.0, C ABI 2 / Python API 4).
- Continuing with the same backend, dtype and batch order is the reproducible
  workflow. A different backend or dtype agrees within numerical tolerance,
  not bit for bit.

The full rules are in the [support matrix](../support-matrix.md) and the
[API and state contract](../api.md), both in Traditional Chinese.
