"""Reject submissions that hand back training images instead of generated ones.

FID-style metrics compare bags of pictures, so a real photograph always beats a
generated one. This check works on provenance rather than appearance: a generated
image is never a near-copy of one specific training image, a retrieved one always is.

The comparison runs in the SAME Inception feature space the metric scores on, and it
is the only space where the check cannot be dodged for free. A pixel-space check is
defeated by a flip or a rotation -- the pixels move, the Inception features barely do,
so the retrieved image keeps its FID advantage. Here the two are tied together: any
edit large enough to escape this check is also large enough to lose the FID advantage
that made retrieval attractive.

Measured on the CellFlux tasks -- share of outputs within DUPLICATE_DISTANCE of some
training image: legitimate methods (sealed ground truth, control passthrough, official
CellFlux) stay far below the threshold; verbatim and transformed retrieval sit at 100%.
"""
from __future__ import annotations

import numpy as np
import torch
from torchmetrics.image.fid import FrechetInceptionDistance

FEATURE_DIMENSION = 2048
DUPLICATE_DISTANCE = 3.0     # L2 in Inception feature space; legit methods sit at 6-9, retrieval at ~0.9
MAX_DUPLICATE_SHARE = 0.05
BATCH = 64


def _to_rgb_uint8(images: np.ndarray) -> np.ndarray:
    """Three channels in, three out; more channels collapse to the first three."""
    if images.shape[-1] == 3:
        return images
    return np.ascontiguousarray(images[..., :3])


def _extractor(device: torch.device):
    metric = FrechetInceptionDistance(feature=FEATURE_DIMENSION, normalize=False).to(device)
    metric.inception.eval()
    return metric.inception


def features(images: np.ndarray, device: torch.device, inception=None) -> torch.Tensor:
    """Inception-2048 features, the representation the FID score itself is built on."""
    model = inception if inception is not None else _extractor(device)
    rgb = _to_rgb_uint8(images)
    out = []
    with torch.no_grad():
        for start in range(0, len(rgb), BATCH):
            block = torch.from_numpy(rgb[start : start + BATCH]).to(device)
            block = block.permute(0, 3, 1, 2).contiguous()
            out.append(model(block).float())
    return torch.cat(out) if out else torch.empty(0, FEATURE_DIMENSION, device=device)


def duplicate_share(generated: np.ndarray, reference_images: np.ndarray) -> tuple[float, float]:
    """(share of outputs that duplicate a reference image, smallest distance seen)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inception = _extractor(device)
    # The eight dihedral views close the "flip it and it is a new picture" route.
    reference = []
    for flip in (False, True):
        for turns in range(4):
            view = reference_images[:, ::-1] if flip else reference_images
            view = np.ascontiguousarray(np.rot90(view, turns, axes=(1, 2)))
            reference.append(features(view, device, inception))
    reference = torch.cat(reference)
    reference_sq = (reference ** 2).sum(1)

    near, smallest = 0, float("inf")
    query = features(generated, device, inception)
    for start in range(0, len(query), 256):
        block = query[start : start + 256]
        distance = (block ** 2).sum(1)[:, None] + reference_sq[None, :] - 2.0 * (block @ reference.T)
        closest = distance.clamp_min(0).min(1).values.sqrt()
        near += int((closest < DUPLICATE_DISTANCE).sum())
        smallest = min(smallest, float(closest.min()))
    return near / len(generated), smallest


def violates(generated: np.ndarray, reference_images: np.ndarray) -> tuple[bool, dict]:
    share, smallest = duplicate_share(generated, reference_images)
    details = {
        "duplicate_share": share,
        "closest_reference_distance": smallest,
        "duplicate_distance_threshold": DUPLICATE_DISTANCE,
        "max_duplicate_share": MAX_DUPLICATE_SHARE,
    }
    return share > MAX_DUPLICATE_SHARE, details
