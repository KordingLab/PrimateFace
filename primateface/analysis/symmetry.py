"""Facial symmetry analysis via fluctuating asymmetry.

Computes fluctuating asymmetry (FA) from 68-point landmarks by comparing
left-right symmetric landmark pairs relative to the facial midline.
FA is proposed as a biomarker of developmental stress in primates.
"""

from typing import Dict

import numpy as np

from .constants import SYMMETRIC_PAIRS, SYMMETRIC_PAIRS_BY_REGION
from .utils import fit_midline, interocular_distance, point_to_line_distance


def facial_symmetry(
    keypoints: np.ndarray,
    method: str = "midline",
) -> float:
    """Compute fluctuating asymmetry score.

    Args:
        keypoints: Array of shape (68, 2) or (68, 3).
        method: Symmetry computation method.
            'midline': Fit line through midline landmarks, compare left/right
                       pair distances to it.
            'procrustes': Reflect landmarks across midline, Procrustes align,
                          measure residual distance.

    Returns:
        FA score >= 0 (0 = perfect symmetry), normalized by interocular distance.
    """
    if method == "midline":
        return _symmetry_midline(keypoints, SYMMETRIC_PAIRS)
    elif method == "procrustes":
        return _symmetry_procrustes(keypoints)
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'midline' or 'procrustes'.")


def per_region_symmetry(keypoints: np.ndarray) -> Dict[str, float]:
    """Compute symmetry broken down by facial region.

    Args:
        keypoints: Array of shape (68, 2) or (68, 3).

    Returns:
        Dict mapping region name to FA score. Regions: jaw, eyebrows,
        eyes, nose, mouth.
    """
    results: Dict[str, float] = {}
    for region, pairs in SYMMETRIC_PAIRS_BY_REGION.items():
        results[region] = _symmetry_midline(keypoints, pairs)
    return results


def _symmetry_midline(
    keypoints: np.ndarray,
    pairs: list,
) -> float:
    """Compute FA using midline-based signed distances.

    For each symmetric pair, compute the absolute difference between
    the signed perpendicular distance of each landmark to the midline.
    Average across all pairs, normalize by interocular distance.
    """
    coords = keypoints[:, :2]
    centroid, direction = fit_midline(keypoints)
    iod = interocular_distance(keypoints)

    if iod < 1e-6:
        return 0.0

    asymmetries = []
    for left_idx, right_idx in pairs:
        d_left = point_to_line_distance(coords[left_idx], centroid, direction)
        d_right = point_to_line_distance(coords[right_idx], centroid, direction)
        # For a symmetric face, |d_left| ≈ |d_right| but with opposite signs
        asymmetries.append(abs(abs(d_left) - abs(d_right)))

    if len(asymmetries) == 0:
        return 0.0

    return float(np.mean(asymmetries)) / iod


def _symmetry_procrustes(keypoints: np.ndarray) -> float:
    """Compute FA by reflecting landmarks and measuring Procrustes residual.

    1. Reflect all landmarks across the fitted midline.
    2. Swap left-right pair indices to get the "mirror" configuration.
    3. Procrustes-align original and mirror.
    4. Return mean residual distance, normalized by IOD.
    """
    coords = keypoints[:, :2].copy()
    centroid, direction = fit_midline(keypoints)
    iod = interocular_distance(keypoints)

    if iod < 1e-6:
        return 0.0

    # Reflect each point across the midline
    reflected = np.empty_like(coords)
    normal = np.array([-direction[1], direction[0]])  # perpendicular to midline

    for i in range(len(coords)):
        diff = coords[i] - centroid
        proj = np.dot(diff, normal)
        reflected[i] = coords[i] - 2 * proj * normal

    # Swap left-right pair indices in the reflected set
    mirrored = reflected.copy()
    for left_idx, right_idx in SYMMETRIC_PAIRS:
        mirrored[left_idx] = reflected[right_idx]
        mirrored[right_idx] = reflected[left_idx]

    # Procrustes alignment (translation + rotation, no scaling)
    # Center both
    orig_centered = coords - coords.mean(axis=0)
    mirror_centered = mirrored - mirrored.mean(axis=0)

    # Optimal rotation via SVD
    h = orig_centered.T @ mirror_centered
    u, _, vt = np.linalg.svd(h)
    d = np.linalg.det(vt.T @ u.T)
    correction = np.diag([1, np.sign(d)])
    rotation = vt.T @ correction @ u.T

    aligned = (rotation @ mirror_centered.T).T

    # Mean residual
    residuals = np.linalg.norm(orig_centered - aligned, axis=-1)
    return float(residuals.mean()) / iod
