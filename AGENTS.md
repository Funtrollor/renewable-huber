# Agent working notes

Read this before changing anything under `native/` or `src/renewable_huber/backends/`.
It is deliberately short; the full reasoning lives in
[`docs/maintainability-refactor.md`](docs/maintainability-refactor.md).

`CONTRIBUTING.md` documents the PowerShell workflow. Development has since moved
to WSL2/Linux; the commands below are the Linux equivalents.

## Numbering collision

`P0`–`P3` means two different things in this repository:

| | this file and `docs/maintainability-refactor.md` | `docs/native-core-p*.md` |
|---|---|---|
| P0 | C ABI contract, cross-language manifest, `engine.cu` split | pure-Python baseline |
| P1 | Python backend capability contract | Rust CPU engine |
| P2 | Rust module split | CUDA whole-batch engine |

Always check which scheme a document is using.

## State

Branch `codex/maintainability-p0` carries a structural refactor: P0, P1 and P2
of the maintainability audit are done and verified; **P3 is not started**
(`CheckpointPayload`, test profiles, benchmark sweep split).

## Agent roles and hand-off

- **Claude Code writes implementation code from an agreed engineering plan.**
- **Codex owns architecture, review, acceptance, commits, pushes and pull
  requests.** Claude Code must not commit or push this repository.
- Before starting work, both agents read this file and
  [`docs/agent-handoff.md`](docs/agent-handoff.md), then inspect `git status`
  and the commits made since the hand-off's base SHA.
- After a work session, append one structured entry to `docs/agent-handoff.md`.
  Never use a transcript or an ignored `.claude/` file as the only record of a
  design decision.
- Do not run both agents in the same working tree at the same time. Use
  separate branches/worktrees, and let Codex integrate reviewed commits or
  uncommitted patches into the publishing branch.

Nothing in it changes an algorithm, a kernel order, stream behaviour, or a
public API. Keep it that way: the CPU 1.45x–15.65x and CUDA 1.12x–2.04x results
and the golden corpus must stay bit-identical.

## Things that break silently

Violate any of these and the program keeps running with correct numbers while
the contract quietly stops holding. The full list of 13 is in section 6 of the
report; these are the ones worth memorising.

- **`Failure` must keep its out-of-line destructor and must never sit in an
  anonymous namespace in a header.** It is thrown from several CUDA translation
  units and caught in `c_api.cu`. Per-TU copies are distinct types, the `catch`
  stops matching, and *every* error status degrades to `INTERNAL_ERROR (8)`
  while the numerics stay correct. Guarded by the `status_survives_translation_units`
  smoke case.
- **`RhCudaEngine` stays at global scope.** `rh_cuda.h` forward-declares it
  there and all 17 `extern "C"` entry points take `::RhCudaEngine*`.
- **`mod abi` in `rh-cuda-ffi/src/sys.rs` must not go back behind
  `#[cfg(feature = "cuda")]`.** That gate is what lets the ABI layout tests run
  in CI without a CUDA toolchain.
- **`native/contracts/rh_cuda_contract.json` is the source of truth.** Change it
  first, then the four mirrors it names (C header, C++ `static_assert`s, Rust
  `sys.rs`, PyO3 dict keys). `tests/test_native_cuda_contract.py` refereeing them
  asserts *how many items each parser found* before comparing — never remove
  those count assertions, or a regex that stops matching turns the suite into
  one that always passes.
- **The library exports exactly 17 `rh_cuda_*` symbols.** Baseline in
  `artifacts/baseline-exports.txt`. On Linux a couple of libstdc++ `std::string`
  template instantiations also appear as weak symbols; compare the `rh_cuda_`
  prefix, not the total.
- **`capabilities_of()` is the only place that probes a backend for optional
  behaviour**, and `read_n_jobs` / `read_cuda_features` must stay live accessors.
  Snapshotting them passes every test except
  `tests/test_backend_capabilities.py::LiveAccessorTests`.
- **`allocate<T>` in the CUDA sources is instantiated with `int` as well as
  `float`/`double`** (for `d_pivots` and `d_solver_info`). Do not "tidy" it into
  an explicit instantiation list.

## Verification

```bash
.venv/bin/python -m unittest discover -s tests
.venv/bin/python -m ruff check src tests scripts
.venv/bin/python -m ruff format --check src tests scripts

# rh_cuda.h needs only <stdint.h>, so this needs no GPU and no CUDA toolkit
g++ -std=c++17 -fsyntax-only -I native/cuda/include native/cuda/src/abi_contract.cpp

cd native
cargo fmt --all -- --check
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo check  --locked --workspace --all-targets
# NOT --workspace: PyO3 extension-module crates cannot link as standalone test
# binaries on Linux (unresolved CPython symbols). ci.yml scopes it the same way.
cargo test   --locked -p rh-core -p rh-cpu -p rh-cuda-ffi --all-targets
```

CUDA (needs `nvcc`; `export PATH=/usr/local/cuda/bin:$PATH`):

```bash
cmake -S native/cuda -B build/static -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DRH_CUDA_BUILD_SHARED=OFF -DRH_CUDA_BUILD_TESTS=ON -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build/static && ctest --test-dir build/static --output-on-failure
```

Rebuild the extensions with `bash scripts/setup-wsl-venv.sh --cuda`.

## Measurement

The CUDA benchmark harness on this host drifts 2%–19% between runs of the *same
binary* (`wide float32` is worst). Five runs of A followed by five of B compares
two thermal states, not two binaries. Use
`scripts/benchmarks/run_interleaved_benchmark.py`, which alternates A/B and B/A
and gates aligned paired ratios. **Do not draw conclusions below about ±10%.**

## Corrections to the audit report

- Its P3 proposes pytest markers. No test here imports pytest and CI runs
  `python -m unittest` exclusively, so markers alone would change nothing. A root
  `conftest.py` using `pytest_collection_modifyitems` would work without touching
  the 18 test modules or breaking unittest.
- Its suggested `engine.cu` decomposition had no home for `Blas<T>`/`Solver<T>`
  or for the batch layer, and over-fragmented the memory pool and prediction.
  See section 2.3 of the report for what was done instead and why.
