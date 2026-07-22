"""Stream YearPredictionMSD through the refactored public package API."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from renewable_huber import RenewableHuberRegressor  # noqa: E402


def iter_year_prediction_batches(path: Path, batch_size: int):
    """Yield `(X, y)` batches without loading the complete dataset into memory."""

    try:
        import pandas as pd
    except ImportError as error:
        raise SystemExit(
            "This experiment needs pandas. Install it with: pip install .[pandas]"
        ) from error

    for frame in pd.read_csv(path, header=None, chunksize=batch_size):
        yield frame.iloc[:, 1:].to_numpy(dtype=float), frame.iloc[:, 0].to_numpy(dtype=float)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "data" / "renewable_huber" / "raw" / "YearPredictionMSD.txt",
    )
    parser.add_argument("--batch-size", type=int, default=32_768)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--penalty", choices=["none", "l1"], default="none")
    parser.add_argument("--backend", choices=["numpy", "cupy"], default="numpy")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "year_prediction.npz",
    )
    args = parser.parse_args()

    model = RenewableHuberRegressor(
        penalty=args.penalty,
        backend=args.backend,
        device="cuda" if args.backend == "cupy" else "cpu",
        dtype=args.dtype,
    )
    for batch_index, (X_batch, y_batch) in enumerate(
        iter_year_prediction_batches(args.data, args.batch_size), start=1
    ):
        model.partial_fit(X_batch, y_batch)
        diagnostics = model.diagnostics_
        print(
            f"batch={batch_index:4d} samples={model.state_.n_samples_seen:7d} "
            f"iterations={diagnostics.iterations:3d} objective={diagnostics.objective:.6f} "
            f"backend={model.backend_}/{model.device_}"
        )
        if args.max_batches is not None and batch_index >= args.max_batches:
            break

    model.save(args.checkpoint)
    print(f"Saved renewable state to {args.checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
