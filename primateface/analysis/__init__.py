"""Landmark-based facial analysis for PrimateFace.

Provides geometric and kinematic analysis derived from 68-point
facial landmarks: kinematics, symmetry, head pose, and quality.

Example:
    >>> from analysis.kinematics import extract_kinematics
    >>> features = extract_kinematics(keypoints)  # keypoints shape (68, 2)
    >>> print(features["mouth_aperture"])
"""

from .head_pose import estimate_head_pose
from .kinematics import (
    brow_height,
    detect_lip_smack,
    extract_kinematics,
    extract_timeseries,
    eye_aperture,
    eye_to_mouth,
    face_aspect_ratio,
    face_height,
    face_width,
    jaw_width,
    mouth_aperture,
    mouth_aspect_ratio,
    mouth_width,
    nose_length,
)
from .quality import face_quality
from .symmetry import facial_symmetry, per_region_symmetry
from .utils import (
    get_eye_centers,
    interocular_distance,
    load_keypoints_from_coco,
    visibility_ratio,
)

__all__ = [
    # kinematics
    "mouth_aperture",
    "mouth_width",
    "mouth_aspect_ratio",
    "eye_aperture",
    "brow_height",
    "face_height",
    "face_width",
    "face_aspect_ratio",
    "jaw_width",
    "nose_length",
    "eye_to_mouth",
    "extract_kinematics",
    "extract_timeseries",
    "detect_lip_smack",
    # symmetry
    "facial_symmetry",
    "per_region_symmetry",
    # head pose
    "estimate_head_pose",
    # quality
    "face_quality",
    # utils
    "interocular_distance",
    "get_eye_centers",
    "load_keypoints_from_coco",
    "visibility_ratio",
]
