# Installation

Requires Python 3.10–3.13 (the released 0.6.1 supports 3.10–3.12; 3.13 is
available from the next release onwards). The base installation depends only
on NumPy:

```bash
python -m pip install renewable-huber
renewable-huber --version
```

## Optional Rust CPU core

--8<-- "README.en.md:native-cpu-install"

How to select the engine, and how `n_jobs` sizes its thread pool, is covered in
[Backends](backends.md#rust-cpu-engine).

## Optional native CUDA engine

--8<-- "README.en.md:native-cuda-install"

Using the engine is covered in [Backends](backends.md#native-cuda-engine).

## Extras

--8<-- "README.en.md:extras"

The GPU extras install only the corresponding framework; they do not install
the NVIDIA driver or the CUDA runtime for you. First check compatibility
between the framework, operating system, Python and GPU driver; see the
[support matrix](../support-matrix.md) for detailed limits.

## From source

For development from source:

```bash
python -m pip install -e ".[dev]"
```

Building the native extensions locally, and the WSL2/Linux toolchain profiles,
are described in [Contributing](../development/contributing.md).
