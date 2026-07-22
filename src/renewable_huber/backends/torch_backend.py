"""PyTorch backend imported lazily to keep the NumPy installation lightweight."""

from __future__ import annotations

from typing import Any

from ..exceptions import BackendUnavailableError


class _TorchNamespace:
    """Array-API-shaped facade that creates new arrays on one Torch device."""

    def __init__(self, torch: Any, device: Any) -> None:
        self._torch = torch
        self._device = device

    def __getattr__(self, name: str) -> Any:
        return getattr(self._torch, name)

    def zeros(self, *shape: Any, dtype: Any = None) -> Any:
        return self._torch.zeros(*shape, dtype=dtype, device=self._device)

    def ones(self, *shape: Any, dtype: Any = None) -> Any:
        return self._torch.ones(*shape, dtype=dtype, device=self._device)

    def eye(self, size: int, *, dtype: Any = None) -> Any:
        return self._torch.eye(size, dtype=dtype, device=self._device)

    def concatenate(self, values: tuple[Any, ...]) -> Any:
        return self._torch.cat(values)

    def transpose(self, value: Any) -> Any:
        return value.T


class TorchBackend:
    """Eager PyTorch backend for native CPU and CUDA tensor workflows.

    ``device='auto'`` intentionally selects CPU.  CUDA is only used after an
    explicit ``device='cuda'`` request, matching the predictable device policy
    used by the other backends.
    """

    name = "torch"

    def __init__(self, dtype: str = "float64", device: str = "auto") -> None:
        try:
            import torch
        except ImportError as error:
            raise BackendUnavailableError(
                "The PyTorch backend requires PyTorch. "
                "Install it with: pip install 'renewable-huber[gpu-torch]'"
            ) from error

        if device == "cuda":
            if not torch.cuda.is_available():
                raise BackendUnavailableError(
                    "backend='torch', device='cuda' requires a PyTorch build with an available "
                    "CUDA device"
                )
            self._device = torch.device(f"cuda:{torch.cuda.current_device()}")
        else:
            self._device = torch.device("cpu")

        self._torch = torch
        self.dtype = getattr(torch, dtype)
        self.device = str(self._device)
        self.xp = _TorchNamespace(torch, self._device)

    def asarray(self, value: Any) -> Any:
        if hasattr(value, "to_numpy"):
            value = value.to_numpy()
        if isinstance(value, self._torch.Tensor):
            return value.detach().to(device=self._device, dtype=self.dtype)
        return self._torch.as_tensor(value, dtype=self.dtype, device=self._device)

    def copy(self, value: Any) -> Any:
        return value.clone()

    def reshape(self, value: Any, shape: tuple[int, ...]) -> Any:
        return value.reshape(shape)

    def to_numpy(self, value: Any) -> Any:
        return value.detach().cpu().numpy()

    def scalar(self, value: Any) -> float:
        if not isinstance(value, self._torch.Tensor):
            value = self._torch.as_tensor(value)
        return float(value.detach().item())

    def solve(self, matrix: Any, vector: Any) -> Any:
        try:
            return self._torch.linalg.solve(matrix, vector)
        except RuntimeError:
            return self._torch.linalg.lstsq(matrix, vector, rcond=None).solution

    def norm(self, value: Any) -> float:
        return self.scalar(self._torch.linalg.vector_norm(value))

    def is_finite(self, value: Any) -> bool:
        return bool(self._torch.all(self._torch.isfinite(value)).item())

    def synchronize(self) -> None:
        if self._device.type == "cuda":
            self._torch.cuda.synchronize(self._device)
