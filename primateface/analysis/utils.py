"""Shared utilities for landmark-based analysis.

Provides helpers for loading COCO keypoints, normalization, and
geometric primitives used across analysis modules.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np

from .constants import LEFT_EYE, MIDLINE, RIGHT_EYE


def load_keypoints_from_coco(
    coco_json: Union[str, Path],
) -> Tuple[List[np.ndarray], List[Dict]]:
    """Load 68-point keypoints from a COCO JSON annotation file.

    Args:
        coco_json: Path to COCO JSON file.

    Returns:
        Tuple of:
          - keypoints: List of arrays, each shape (68, 3) with [x, y, visibility].
          - annotations: List of raw annotation dicts (for metadata access).
    """
    with open(coco_json, "r") as f:
        data = json.load(f)

    keypoints_list: List[np.ndarray] = []
    annotations: List[Dict] = []

    for ann in data.get("annotations", []):
        kpts_flat = ann.get("keypoints", [])
        if len(kpts_flat) == 0:
            continue
        kpts = np.array(kpts_flat, dtype=np.float64).reshape(-1, 3)
        keypoints_list.append(kpts)
        annotations.append(ann)

    return keypoints_list, annotations


def get_eye_centers(keypoints: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the center of each eye from landmark coordinates.

    Args:
        keypoints: Array of shape (68, 2) or (68, 3).

    Returns:
        Tuple of (right_eye_center, left_eye_center), each shape (2,).
    """
    coords = keypoints[:, :2]
    right_center = coords[RIGHT_EYE].mean(axis=0)
    left_center = coords[LEFT_EYE].mean(axis=0)
    return right_center, left_center


def interocular_distance(keypoints: np.ndarray) -> float:
    """Distance between eye centers. Primary normalization reference.

    Args:
        keypoints: Array of shape (68, 2) or (68, 3).

    Returns:
        Euclidean distance between left and right eye centers.
    """
    right_center, left_center = get_eye_centers(keypoints)
    dist = float(np.linalg.norm(left_center - right_center))
    return dist


def normalize_distance(distance: float, keypoints: np.ndarray) -> float:
    """Normalize a distance by interocular distance.

    Args:
        distance: Raw pixel distance.
        keypoints: Array of shape (68, 2) or (68, 3).

    Returns:
        Normalized distance. Returns raw distance if IOD is near zero.
    """
    iod = interocular_distance(keypoints)
    if iod < 1e-6:
        return distance
    return distance / iod


def fit_midline(keypoints: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a line through the facial midline landmarks using least squares.

    Args:
        keypoints: Array of shape (68, 2) or (68, 3).

    Returns:
        Tuple of (centroid, direction) where:
          - centroid: shape (2,), mean of midline points.
          - direction: shape (2,), unit vector along the midline (top to bottom).
    """
    coords = keypoints[:, :2]
    midline_pts = coords[MIDLINE]

    centroid = midline_pts.mean(axis=0)
    centered = midline_pts - centroid

    # SVD to find principal direction
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]  # first principal component

    # Ensure direction points downward (positive y)
    if direction[1] < 0:
        direction = -direction

    return centroid, direction


def point_to_line_distance(
    point: np.ndarray,
    line_point: np.ndarray,
    line_direction: np.ndarray,
) -> float:
    """Signed perpendicular distance from a point to a line.

    Positive = right of line direction, negative = left.

    Args:
        point: Shape (2,).
        line_point: A point on the line, shape (2,).
        line_direction: Unit direction vector of the line, shape (2,).

    Returns:
        Signed perpendicular distance.
    """
    diff = point - line_point
    # Cross product in 2D gives signed distance
    return float(diff[0] * line_direction[1] - diff[1] * line_direction[0])


def visibility_ratio(keypoints: np.ndarray) -> float:
    """Fraction of keypoints that are visible (visibility > 0).

    Args:
        keypoints: Array of shape (68, 3) with [x, y, visibility].

    Returns:
        Float in [0, 1].
    """
    if keypoints.shape[1] < 3:
        return 1.0
    return float((keypoints[:, 2] > 0).mean())
