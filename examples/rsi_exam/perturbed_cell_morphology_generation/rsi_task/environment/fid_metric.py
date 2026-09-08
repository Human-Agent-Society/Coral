from __future__ import annotations

import numpy as np
import torch
from torchmetrics.image.fid import FrechetInceptionDistance


FEATURE_DIMENSION = 2048
COMPOSITE_WEIGHTS = np.asarray(
    [
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ],
    dtype=np.float32,
)


def _rgb_nchw(images: np.ndarray, device: torch.device) -> torch.Tensor:
    if images.ndim != 4 or images.shape[-1] != 6:
        raise ValueError(f"expected NHWC six-channel images, got {images.shape}")
    values = images.astype(np.float32) / 127.5 - 1.0
    rgb = np.einsum("bhwc,ck->bhwk", values, COMPOSITE_WEIGHTS)
    rgb = np.clip(rgb, -1.0, 1.0)
    rgb = np.floor(np.clip(rgb * 0.5 + 0.5, 0.0, 1.0) * 255.0).astype(np.uint8)
    return torch.from_numpy(rgb).permute(0, 3, 1, 2).contiguous().to(device)


def overall_fid(real: np.ndarray, generated: np.ndarray) -> float:
    if real.shape != generated.shape or real.ndim != 4 or real.shape[-1] != 6:
        raise ValueError(
            f"expected matching NHWC six-channel arrays, got {real.shape} and {generated.shape}"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric = FrechetInceptionDistance(feature=FEATURE_DIMENSION, normalize=False).to(device)
    for start in range(0, len(real), 32):
        metric.update(_rgb_nchw(real[start : start + 32], device), real=True)
        metric.update(_rgb_nchw(generated[start : start + 32], device), real=False)
    value = float(metric.cpu().compute().detach())
    if not np.isfinite(value):
        raise ValueError("FID is not finite")
    return max(0.0, value)


def conditional_fid(
    real: np.ndarray, generated: np.ndarray, condition: np.ndarray
) -> float:
    """Mean per-perturbation FID: FID within each condition, then averaged."""
    if len(condition) != len(real):
        raise ValueError(f"condition length {len(condition)} != {len(real)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric = FrechetInceptionDistance(feature=FEATURE_DIMENSION, normalize=False).to(device)
    values = []
    for label in np.unique(condition):
        rows = np.flatnonzero(condition == label)
        if len(rows) < 2:
            continue
        metric.reset()
        for start in range(0, len(rows), 32):
            chunk = rows[start : start + 32]
            metric.update(_rgb_nchw(real[chunk], device), real=True)
            metric.update(_rgb_nchw(generated[chunk], device), real=False)
        value = float(metric.cpu().compute().detach())
        metric.to(device)
        if not np.isfinite(value):
            raise ValueError(f"conditional FID is not finite for condition {label}")
        values.append(max(0.0, value))
    if not values:
        raise ValueError("no condition had enough samples for conditional FID")
    return float(np.mean(values))
