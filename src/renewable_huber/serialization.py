"""Safe checkpoint format for model configuration and renewable summary state."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .state import RenewableHuberState

if TYPE_CHECKING:
    from .estimator import RenewableHuberRegressor

FORMAT_VERSION = 1


def save_model(model: RenewableHuberRegressor, path: str | Path) -> Path:
    """Write a fitted estimator to a compressed, pickle-free NumPy archive."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = model.state_dict()
    metadata = {
        "format_version": FORMAT_VERSION,
        "config": payload["config"],
        "n_samples_seen": payload["n_samples_seen"],
        "batch_count": payload["batch_count"],
        "previous_lambda": payload["previous_lambda"],
        "n_features_in": payload["n_features_in"],
        "fit_intercept": payload["fit_intercept"],
    }
    with target.open("wb") as file_handle:
        np.savez_compressed(
            file_handle,
            coefficients=payload["coefficients"],
            information=payload["information"],
            metadata=np.asarray(json.dumps(metadata)),
        )
    return target


def load_model(path: str | Path) -> RenewableHuberRegressor:
    """Load a checkpoint made by :func:`save_model` without enabling pickle."""

    from .estimator import RenewableHuberRegressor

    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        if metadata.get("format_version") != FORMAT_VERSION:
            raise ValueError("Unsupported renewable-huber checkpoint format")
        state = RenewableHuberState(
            coefficients=np.asarray(archive["coefficients"], dtype=np.float64),
            information=np.asarray(archive["information"], dtype=np.float64),
            n_samples_seen=int(metadata["n_samples_seen"]),
            batch_count=int(metadata["batch_count"]),
            previous_lambda=float(metadata["previous_lambda"]),
            n_features_in=int(metadata["n_features_in"]),
            fit_intercept=bool(metadata["fit_intercept"]),
        )
    model = RenewableHuberRegressor(**metadata["config"])
    model._restore_state(state)
    return model
