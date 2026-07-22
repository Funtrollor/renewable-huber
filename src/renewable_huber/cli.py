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
    print("available backends: numpy (CPU), cupy (CUDA; install the gpu-cupy extra)")
    print("reserved backends: torch, tensorflow")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
