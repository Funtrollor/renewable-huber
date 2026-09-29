# renewable-huber

[![CI](https://github.com/Funtrollor/renewable-huber/actions/workflows/ci.yml/badge.svg)](https://github.com/Funtrollor/renewable-huber/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/renewable-huber.svg)](https://pypi.org/project/renewable-huber/)
[![Python versions](https://img.shields.io/pypi/pyversions/renewable-huber.svg)](https://pypi.org/project/renewable-huber/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-D22128.svg)](https://github.com/Funtrollor/renewable-huber/blob/main/LICENSE)

--8<-- "README.en.md:intro"

```bash
python -m pip install renewable-huber
```

--8<-- "README.en.md:quickstart"

`data_stream`, `X_test` and the other names above are placeholders for your own
batches: any iterable of `(X_batch, y_batch)` pairs of NumPy arrays or pandas
objects works.

## Where to go next

<div class="grid cards" markdown>

- **[Installation](guide/installation.md)**: the base package, the optional
  Rust CPU and CUDA engines, and the extras for pandas, scikit-learn, CuPy,
  PyTorch and TensorFlow.
- **[Quick start](guide/quickstart.md)**: streaming updates, pandas and
  scikit-learn input, sample weights and what streaming means for
  reproducibility.
- **[Backends](guide/backends.md)**: how `backend="auto"` chooses, and how to
  run on the Rust CPU engine, CuPy, the native CUDA engine, PyTorch or
  TensorFlow.
- **[Checkpoints](guide/checkpoints.md)**: saving, restoring and migrating a
  stream between backends and dtypes.
- **[Reference](api.md)**: the public API and state contract, the
  [support matrix](support-matrix.md) and the generated
  [Python API](reference/python-api.md).
- **[Development](development/contributing.md)**: contributing, architecture,
  the release process and the native-core design notes.

</div>

!!! note "Languages"

    The user guide and this page are in English. The API contract, support
    matrix, architecture, release documents and maintainability report are
    written in Traditional Chinese and are marked **(中文)** in the navigation.
    A Traditional Chinese README, which is also the PyPI project description,
    is [on GitHub](https://github.com/Funtrollor/renewable-huber/blob/main/README.md).

## Research source and licence

This project is an independent software implementation of the method in
[the original Renewable Huber paper (Electronic Journal of Statistics, DOI)](https://doi.org/10.1214/24-EJS2223).
It is not an official package created, sponsored, approved or endorsed by the
paper's authors or their institutions. The research article is licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); this project's source
code is licensed under the
[Apache License 2.0](https://github.com/Funtrollor/renewable-huber/blob/main/LICENSE),
and detailed attribution notices are in
[NOTICE](https://github.com/Funtrollor/renewable-huber/blob/main/NOTICE).
