# Quick start

## Streaming updates

--8<-- "README.en.md:quickstart"

`data_stream` is any iterable of `(X_batch, y_batch)` pairs; `X_test` is a
feature matrix with the same columns. The model keeps only its coefficients and
an accumulated information matrix, so memory use does not grow with the length
of the stream.

## pandas, scikit-learn and sample weights

--8<-- "README.en.md:inputs"

## Streaming semantics

--8<-- "README.en.md:streaming-notes"

Continuing a stream from a saved state is covered in
[Checkpoints](checkpoints.md); running on another engine or device is covered
in [Backends](backends.md).
