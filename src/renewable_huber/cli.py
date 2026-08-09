"""Small diagnostic command-line interface for package installations."""

from __future__ import annotations

import argparse

from ._version import __version__


def main(argv: list[str] | None = None) -> int:
    """Print package capability information and return a process status."""

    parser = argparse.ArgumentParser(prog="renewable-huber")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "command",
        nargs="?",
        choices=["info"],
        default="info",
        help="command to run (default: info)",
    )
    parser.parse_args(argv)
    print(f"renewable-huber {__version__}")
    print("available backends:")
    print("  numpy: CPU (base install)")
    print("  native_cpu: opt-in Rust CPU whole-batch engine (install renewable-huber-native-cpu)")
    print("  cupy: CUDA (install the gpu-cupy extra)")
    print("  native_cuda: opt-in Rust/CUDA whole-batch engine (source build)")
    print("  torch: CPU/CUDA (install the gpu-torch extra)")
    print("  tensorflow: CPU/CUDA, eager execution only (install the gpu-tensorflow extra)")
    print("device policy: backend='auto' stays on CPU unless device='cuda' selects CuPy")
    print(
        "cpu policy: backend='auto' may select native_cpu for a large batch when a bounded "
        "runtime measurement on this host shows it faster by a conservative margin. It never "
        "reads the CPU model and never writes a cache file; measurements are kept in memory "
        "and discarded when CPU affinity, thread environment, or an observable BLAS/OpenMP "
        "pool changes, and after fork. Any failure falls back to NumPy. "
        "See auto_dispatch_ on a fitted estimator."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
