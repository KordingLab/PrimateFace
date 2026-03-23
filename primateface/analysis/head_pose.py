"""Head pose estimation from 68-point facial landmarks.

Estimates yaw, pitch, and roll using cv2.solvePnP with a generic
3D face reference model and 2D landmark correspondences.
"""

from typing import Optional, Tuple

import cv2
import numpy as np

from .constants import POSE_LANDMARK_INDICES, POSE_REFERENCE_3D


def estimate_head_pose(
    keypoints: np.ndarray,
    image_size: Tuple[int, int],
    camera_matrix: Optional[np.ndarray] = None,
    dist_coeffs: Optional[np.ndarray] = None,
) -> Tuple[float, float, float]:
    """Estimate head pose (yaw, pitch, roll) from landmarks.

    Uses 6 reference landmarks (nose tip, chin, eye corners, mouth corners)
    and cv2.solvePnP to recover the rotation of the head.

    Args:
        keypoints: Array of shape (68, 2) or (68, 3). Only x,y used.
        image_size: (width, height) of the source image.
        camera_matrix: Optional 3x3 camera intrinsic matrix. If None, a
            default is constructed assuming the focal length equals the
            image width and the principal point is the image center.
        dist_coeffs: Optional distortion coefficients. Defaults to zero.

    Returns:
        Tuple of (yaw, pitch, roll) in degrees.
        - yaw: horizontal rotation (positive = subject looking left)
        - pitch: vertical rotation (positive = subject looking up)
        - roll: tilt (positive = subject tilting right)
    """
    coords = keypoints[:, :2].astype(np.float64)
    image_points = coords[POSE_LANDMARK_INDICES]

    w, h = image_size
    if camera_matrix is None:
        focal_length = float(w)
        cx, cy = w / 2.0, h / 2.0
        camera_matrix = np.array(
            [[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]],
            dtype=np.float64,
        )

    if dist_coeffs is None:
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rvec, _ = cv2.solvePnP(
        POSE_REFERENCE_3D,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        return 0.0, 0.0, 0.0

    rotation_matrix, _ = cv2.Rodrigues(rvec)
    yaw, pitch, roll = _rotation_matrix_to_euler(rotation_matrix)

    return float(np.degrees(yaw)), float(np.degrees(pitch)), float(np.degrees(roll))


def _rotation_matrix_to_euler(r: np.ndarray) -> Tuple[float, float, float]:
    """Convert a 3x3 rotation matrix to Euler angles (yaw, pitch, roll).

    Uses the ZYX convention.

    Args:
        r: 3x3 rotation matrix.

    Returns:
        Tuple of (yaw, pitch, roll) in radians.
    """
    sy = np.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)

    if sy > 1e-6:
        roll = np.arctan2(r[2, 1], r[2, 2])
        pitch = np.arctan2(-r[2, 0], sy)
        yaw = np.arctan2(r[1, 0], r[0, 0])
    else:
        roll = np.arctan2(-r[1, 2], r[1, 1])
        pitch = np.arctan2(-r[2, 0], sy)
        yaw = 0.0

    return yaw, pitch, roll
