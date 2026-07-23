"""Safe checkpoint format for model configuration and renewable summary state."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .exceptions import ValidationError
from .state import RenewableHuberState

if TYPE_CHECKING:
    from .estimator import RenewableHuberRegressor

FORMAT_VERSION = 2


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
        "weight_sum": payload["weight_sum"],
        "feature_names_in": payload["feature_names_in"],
    }
    with target.open("wb") as file_handle:
        np.savez_compressed(
            file_handle,
            coefficients=payload["coefficients"],
            information=payload["information"],
            metadata=np.asarray(json.dumps(metadata)),
        )
    return target


def load_model(
    path: str | Path,
    *,
    backend: str | None = None,
    device: str | None = None,
    dtype: str | None = None,
    estimator_class: type[RenewableHuberRegressor] | None = None,
) -> RenewableHuberRegressor:
    """Load a checkpoint made by :func:`save_model` without enabling pickle."""

    from .estimator import RenewableHuberRegressor

    source = Path(path)
    try:
        with np.load(source, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            format_version = metadata.get("format_version")
            if format_version not in (1, FORMAT_VERSION):
                raise ValidationError("Unsupported renewable-huber checkpoint format")
            state = RenewableHuberState(
                coefficients=np.asarray(archive["coefficients"], dtype=np.float64),
                information=np.asarray(archive["information"], dtype=np.float64),
                n_samples_seen=int(metadata["n_samples_seen"]),
                batch_count=int(metadata["batch_count"]),
                previous_lambda=float(metadata["previous_lambda"]),
                n_features_in=int(metadata["n_features_in"]),
                fit_intercept=bool(metadata["fit_intercept"]),
                weight_sum=float(metadata.get("weight_sum", metadata["n_samples_seen"])),
            )
    except FileNotFoundError:
        raise
    except ValidationError:
        raise
    except (
        AttributeError,
        EOFError,
        IndexError,
        KeyError,
        OSError,
        OverflowError,
        TypeError,
        ValueError,
    ) as error:
        raise ValidationError("Invalid or corrupted renewable-huber checkpoint") from error

    try:
        config = dict(metadata["config"])
        if backend is not None:
            config["backend"] = backend
            if device is None:
                config["device"] = "auto"
        if device is not None:
            config["device"] = device
        if dtype is not None:
            config["dtype"] = dtype
        model_type = RenewableHuberRegressor if estimator_class is None else estimator_class
        model = model_type(**config)
    except (KeyError, TypeError) as error:
        raise ValidationError("Invalid renewable-huber checkpoint configuration") from error
    model._restore_state(state, feature_names=metadata.get("feature_names_in"))
    return model
