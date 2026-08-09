"""Pickle-free checkpoint codec for configuration and renewable summary state.

This module is the persistence boundary and nothing more. It encodes and
decodes a :class:`CheckpointPayload`; it never imports the estimator class,
never constructs one, and never reaches into an estimator's private lifecycle.
Projecting a fitted model onto a payload, and restoring a payload into a new
model, both belong to the estimator layer, which is where the rules about when
fitted attributes may be created live.

The archive layout is unchanged: two float arrays plus a JSON metadata string,
loaded with ``allow_pickle=False``. Version 1 archives omit ``weight_sum`` and
fall back to unit weights; version 2 stores it explicitly.

``CheckpointPayload.diagnostics`` exists because diagnostics are part of what a
checkpoint *could* describe, but no released format persists them. Decoding a
version 1 or version 2 archive therefore always yields ``None`` rather than an
invented summary of a batch the file never recorded. Persisting diagnostics
requires a new format version and is deliberately not done here.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .exceptions import ValidationError
from .state import RenewableHuberState

if TYPE_CHECKING:
    from .core import UpdateDiagnostics

FORMAT_VERSION = 2
#: Every version this codec can decode. Only ``FORMAT_VERSION`` is written.
SUPPORTED_FORMAT_VERSIONS = (1, FORMAT_VERSION)


@dataclass(frozen=True, slots=True, kw_only=True)
class CheckpointPayload:
    """Everything one checkpoint carries, independent of any estimator object.

    ``state`` holds host arrays: :func:`write_checkpoint` stores them verbatim,
    and :func:`read_checkpoint` decodes them as ``float64`` regardless of the
    dtype the producing model used. Converting to a backend's array type is the
    estimator layer's job.
    """

    format_version: int = FORMAT_VERSION
    config: dict[str, Any]
    state: RenewableHuberState
    feature_names: list[str] | None = None
    diagnostics: UpdateDiagnostics | None = None


def write_checkpoint(payload: CheckpointPayload, path: str | Path) -> Path:
    """Write ``payload`` to a compressed, pickle-free NumPy archive."""

    if payload.format_version != FORMAT_VERSION:
        raise ValidationError("Unsupported renewable-huber checkpoint format")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    state = payload.state
    metadata = {
        "format_version": payload.format_version,
        "config": payload.config,
        "n_samples_seen": state.n_samples_seen,
        "batch_count": state.batch_count,
        "previous_lambda": state.previous_lambda,
        "n_features_in": state.n_features_in,
        "fit_intercept": state.fit_intercept,
        "weight_sum": state.effective_weight,
        "feature_names_in": payload.feature_names,
    }
    with target.open("wb") as file_handle:
        np.savez_compressed(
            file_handle,
            coefficients=state.coefficients,
            information=state.information,
            metadata=np.asarray(json.dumps(metadata)),
        )
    return target


def read_checkpoint(path: str | Path) -> CheckpointPayload:
    """Decode a checkpoint written by :func:`write_checkpoint`.

    A missing file propagates as :class:`FileNotFoundError`; every other decode
    failure becomes a :class:`~renewable_huber.exceptions.ValidationError` so a
    truncated or hand-edited archive cannot reach the estimator layer.
    """

    source = Path(path)
    try:
        # Own the handle rather than letting ``np.load`` open the path: when the
        # container turns out not to be a zip, the descriptor it opened is never
        # closed, and every rejected checkpoint leaks one.
        with source.open("rb") as file_handle, np.load(file_handle, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            format_version = metadata.get("format_version")
            if format_version not in SUPPORTED_FORMAT_VERSIONS:
                raise ValidationError("Unsupported renewable-huber checkpoint format")
            state = RenewableHuberState(
                coefficients=np.asarray(archive["coefficients"], dtype=np.float64),
                information=np.asarray(archive["information"], dtype=np.float64),
                n_samples_seen=int(metadata["n_samples_seen"]),
                batch_count=int(metadata["batch_count"]),
                previous_lambda=float(metadata["previous_lambda"]),
                n_features_in=int(metadata["n_features_in"]),
                fit_intercept=bool(metadata["fit_intercept"]),
                # Version 1 predates frequency weights, so unit weights are the
                # only faithful reading of a stream it recorded.
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
        # An ``.npz`` is a zip container, and a file that is not one at all
        # reaches here as BadZipFile, which inherits only from Exception. It
        # is the plainest case of a corrupted checkpoint and must not escape
        # as a zipfile implementation detail.
        zipfile.BadZipFile,
    ) as error:
        raise ValidationError("Invalid or corrupted renewable-huber checkpoint") from error

    # A structurally sound archive whose configuration is not a mapping is a
    # different failure from a corrupted one, and keeps its own message.
    try:
        config = dict(metadata["config"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValidationError("Invalid renewable-huber checkpoint configuration") from error

    return CheckpointPayload(
        format_version=int(format_version),
        config=config,
        state=state,
        feature_names=metadata.get("feature_names_in"),
        diagnostics=None,
    )
