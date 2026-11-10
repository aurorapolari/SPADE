"""PCA eraser with the native erase/crop/save interface."""
from pathlib import Path

import torch

from ..hooks.base import BaseEraser


class PCAEraser(BaseEraser):
    def __init__(self, U, device=None):
        self.U = U.detach().to(device=device or U.device, dtype=torch.float32)

    def erase(self, X, alpha=None):
        dtype = X.dtype
        values = X.float()
        basis = self.U.to(values.device)
        projected = (values @ basis) @ basis.T
        return (values - projected * (1.0 if alpha is None else alpha)).to(dtype)

    def crop(self, start, end):
        return PCAEraser(self.U[:, start:end])

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"U": self.U.detach().cpu()}, path)


def load_eraser_from_ckpt(ckpt_or_path, device=None):
    checkpoint = torch.load(ckpt_or_path, map_location="cpu", weights_only=True) if isinstance(ckpt_or_path, (str, Path)) else ckpt_or_path
    basis = checkpoint.get("U", checkpoint.get("basis"))
    if basis is None:
        raise ValueError("PCA checkpoint requires U")
    return PCAEraser(basis, device=device)
