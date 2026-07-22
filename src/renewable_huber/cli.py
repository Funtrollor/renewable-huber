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
    print("  cupy: CUDA (install the gpu-cupy extra)")
    print("  torch: CPU/CUDA (install the gpu-torch extra)")
    print("  tensorflow: CPU/CUDA, eager execution only (install the gpu-tensorflow extra)")
    print("device policy: backend='auto' uses NumPy unless device='cuda' selects CuPy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
