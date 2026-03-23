"""Landmark-based facial kinematics.

Computes derived kinematic features from 68-point facial landmarks:
mouth aperture, eye aperture, brow height, mouth aspect ratio,
timeseries extraction, and lip-smack detection.

All functions take keypoints as numpy arrays of shape (68, 2) or (68, 3)
and return normalized scalar values or DataFrames.
"""

from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from .constants import (
    CHIN,
    INNER_LOWER_LIP,
    INNER_UPPER_LIP,
    JAW,
    LEFT_EYEBROW,
    LEFT_EYE_VERTICAL_PAIRS,
    LEFT_MOUTH_CORNER,
    NOSE_BRIDGE,
    NOSE_TIP_CENTER,
    RIGHT_EYEBROW,
    RIGHT_EYE_VERTICAL_PAIRS,
    RIGHT_MOUTH_CORNER,
)
from .utils import get_eye_centers, interocular_distance


def mouth_aperture(keypoints: np.ndarray, normalize: bool = True) -> float:
    """Vertical distance between inner upper and lower lip.

    Uses inner mouth landmarks (62, 66) for more accurate aperture
    measurement than outer lip landmarks.

    Args:
        keypoints: Array of shape (68, 2) or (68, 3).
        normalize: If True, normalize by interocular distance.

    Returns:
        Mouth aperture (normalized if requested).
    """
    coords = keypoints[:, :2]
    dist = float(np.linalg.norm(coords[INNER_UPPER_LIP] - coords[INNER_LOWER_LIP]))
    if normalize:
        iod = interocular_distance(keypoints)
        if iod > 1e-6:
            dist /= iod
    return dist


def mouth_width(keypoints: np.ndarray, normalize: bool = True) -> float:
    """Horizontal distance between mouth corners.

    Args:
        keypoints: Array of shape (68, 2) or (68, 3).
        normalize: If True, normalize by interocular distance.

    Returns:
        Mouth width (normalized if requested).
    """
    coords = keypoints[:, :2]
    dist = float(
        np.linalg.norm(coords[LEFT_MOUTH_CORNER] - coords[RIGHT_MOUTH_CORNER])
    )
    if normalize:
        iod = interocular_distance(keypoints)
        if iod > 1e-6:
            dist /= iod
    return dist


def mouth_aspect_ratio(keypoints: np.ndarray) -> float:
    """Ratio of mouth aperture to mouth width.

    High values indicate open mouth; low values indicate closed or wide mouth.
    Not separately normalized since it is already a ratio.

    Args:
        keypoints: Array of shape (68, 2) or (68, 3).

    Returns:
        Aspect ratio (aperture / width). Returns 0 if width is near zero.
    """
    w = mouth_width(keypoints, normalize=False)
    if w < 1e-6:
        return 0.0
    a = mouth_aperture(keypoints, normalize=False)
    return a / w


def eye_aperture(
    keypoints: np.ndarray,
    side: str = "both",
    normalize: bool = True,
) -> Union[float, Tuple[float, float]]:
    """Vertical opening of the eye(s).

    Computed as the mean vertical distance between upper and lower eyelid
    landmark pairs.

    Args:
        keypoints: Array of shape (68, 2) or (68, 3).
        side: "left", "right", or "both".
        normalize: If True, normalize by interocular distance.

    Returns:
        Single float if side is "left" or "right";
        tuple (right, left) if side is "both".
    """
    coords = keypoints[:, :2]
    iod = interocular_distance(keypoints) if normalize else 1.0
    if iod < 1e-6:
        iod = 1.0

    def _aperture(pairs: list) -> float:
        dists = [
            float(np.linalg.norm(coords[upper] - coords[lower]))
            for upper, lower in pairs
        ]
        val = float(np.mean(dists))
        return val / iod if normalize else val

    if side == "right":
        return _aperture(RIGHT_EYE_VERTICAL_PAIRS)
    elif side == "left":
        return _aperture(LEFT_EYE_VERTICAL_PAIRS)
    else:
        return _aperture(RIGHT_EYE_VERTICAL_PAIRS), _aperture(
            LEFT_EYE_VERTICAL_PAIRS
        )


def brow_height(
    keypoints: np.ndarray,
    side: str = "both",
    normalize: bool = True,
) -> Union[float, Tuple[float, float]]:
    """Distance from eyebrow center to eye center.

    Proxy for AU1 (Inner Brow Raiser) / AU2 (Outer Brow Raiser).
    Higher values indicate raised brows.

    Args:
        keypoints: Array of shape (68, 2) or (68, 3).
        side: "left", "right", or "both".
        normalize: If True, normalize by interocular distance.

    Returns:
        Single float if side is "left" or "right";
        tuple (right, left) if side is "both".
    """
    coords = keypoints[:, :2]
    iod = interocular_distance(keypoints) if normalize else 1.0
    if iod < 1e-6:
        iod = 1.0

    right_eye_center, left_eye_center = get_eye_centers(keypoints)
    right_brow_center = coords[RIGHT_EYEBROW].mean(axis=0)
    left_brow_center = coords[LEFT_EYEBROW].mean(axis=0)

    right_val = float(np.linalg.norm(right_brow_center - right_eye_center))
    left_val = float(np.linalg.norm(left_brow_center - left_eye_center))

    if normalize:
        right_val /= iod
        left_val /= iod

    if side == "right":
        return right_val
    elif side == "left":
        return left_val
    else:
        return right_val, left_val


def face_height(keypoints: np.ndarray, normalize: bool = True) -> float:
    """Vertical distance from chin to brow midpoint.

    Args:
        keypoints: Array of shape (68, 2) or (68, 3).
        normalize: If True, normalize by interocular distance.

    Returns:
        Face height (normalized if requested).
    """
    coords = keypoints[:, :2]
    brow_mid = coords[NOSE_BRIDGE[0]]  # landmark 27, top of nose bridge ≈ brow level
    dist = float(np.linalg.norm(coords[CHIN] - brow_mid))
    if normalize:
        iod = interocular_distance(keypoints)
        if iod > 1e-6:
            dist /= iod
    return dist


def face_width(keypoints: np.ndarray, normalize: bool = True) -> float:
    """Jaw width at the widest point (landmarks 1-15 or 2-14).

    Uses the wider of the two jaw landmark pairs.

    Args:
        keypoints: Array of shape (68, 2) or (68, 3).
        normalize: If True, normalize by interocular distance.

    Returns:
        Face width (normalized if requested).
    """
    coords = keypoints[:, :2]
    w1 = float(np.linalg.norm(coords[JAW[1]] - coords[JAW[15]]))
    w2 = float(np.linalg.norm(coords[JAW[2]] - coords[JAW[14]]))
    dist = max(w1, w2)
    if normalize:
        iod = interocular_distance(keypoints)
        if iod > 1e-6:
            dist /= iod
    return dist


def face_aspect_ratio(keypoints: np.ndarray) -> float:
    """Ratio of face height to face width.

    Higher values indicate elongated faces (typical of adults);
    lower values indicate rounder faces (typical of infants).

    Args:
        keypoints: Array of shape (68, 2) or (68, 3).

    Returns:
        Aspect ratio (height / width). Returns 0 if width is near zero.
    """
    w = face_width(keypoints, normalize=False)
    if w < 1e-6:
        return 0.0
    h = face_height(keypoints, normalize=False)
    return h / w


def jaw_width(keypoints: np.ndarray, normalize: bool = True) -> float:
    """Distance between jaw landmarks 4 and 12 (lower jaw width).

    Narrower than face_width; captures jaw-specific dimorphism.

    Args:
        keypoints: Array of shape (68, 2) or (68, 3).
        normalize: If True, normalize by interocular distance.

    Returns:
        Jaw width (normalized if requested).
    """
    coords = keypoints[:, :2]
    dist = float(np.linalg.norm(coords[JAW[4]] - coords[JAW[12]]))
    if normalize:
        iod = interocular_distance(keypoints)
        if iod > 1e-6:
            dist /= iod
    return dist


def nose_length(keypoints: np.ndarray, normalize: bool = True) -> float:
    """Distance from top of nose bridge (27) to nose tip (30).

    Args:
        keypoints: Array of shape (68, 2) or (68, 3).
        normalize: If True, normalize by interocular distance.

    Returns:
        Nose length (normalized if requested).
    """
    coords = keypoints[:, :2]
    dist = float(np.linalg.norm(coords[NOSE_BRIDGE[0]] - coords[NOSE_TIP_CENTER]))
    if normalize:
        iod = interocular_distance(keypoints)
        if iod > 1e-6:
            dist /= iod
    return dist


def eye_to_mouth(keypoints: np.ndarray, normalize: bool = True) -> float:
    """Distance from eye center midpoint to mouth center.

    Args:
        keypoints: Array of shape (68, 2) or (68, 3).
        normalize: If True, normalize by interocular distance.

    Returns:
        Eye-to-mouth distance (normalized if requested).
    """
    coords = keypoints[:, :2]
    right_eye_c, left_eye_c = get_eye_centers(keypoints)
    eye_mid = (right_eye_c + left_eye_c) / 2.0
    mouth_mid = (coords[LEFT_MOUTH_CORNER] + coords[RIGHT_MOUTH_CORNER]) / 2.0
    dist = float(np.linalg.norm(eye_mid - mouth_mid))
    if normalize:
        iod = interocular_distance(keypoints)
        if iod > 1e-6:
            dist /= iod
    return dist


def extract_kinematics(keypoints: np.ndarray) -> Dict[str, float]:
    """Compute all scalar kinematic and geometric features for a single frame.

    Args:
        keypoints: Array of shape (68, 2) or (68, 3).

    Returns:
        Dict with ~20 named features covering mouth, eyes, brows,
        face proportions, and normalization reference.
    """
    r_eye, l_eye = eye_aperture(keypoints, side="both")
    r_brow, l_brow = brow_height(keypoints, side="both")

    return {
        # Mouth features
        "mouth_aperture": mouth_aperture(keypoints),
        "mouth_width": mouth_width(keypoints),
        "mouth_aspect_ratio": mouth_aspect_ratio(keypoints),
        # Eye features
        "right_eye_aperture": r_eye,
        "left_eye_aperture": l_eye,
        # Brow features
        "right_brow_height": r_brow,
        "left_brow_height": l_brow,
        # Face geometry (new — for demographic prediction)
        "face_height": face_height(keypoints),
        "face_width": face_width(keypoints),
        "face_aspect_ratio": face_aspect_ratio(keypoints),
        "jaw_width": jaw_width(keypoints),
        "nose_length": nose_length(keypoints),
        "eye_to_mouth": eye_to_mouth(keypoints),
        # Normalization reference
        "interocular_distance": interocular_distance(keypoints),
    }


def extract_timeseries(
    keypoints_sequence: np.ndarray,
    fps: float = 30.0,
) -> pd.DataFrame:
    """Compute kinematics for a video sequence.

    Args:
        keypoints_sequence: Array of shape (N_frames, 68, 2) or (N_frames, 68, 3).
        fps: Frames per second (used for time column).

    Returns:
        DataFrame with one row per frame and columns for each kinematic feature
        plus 'frame' and 'time_s'.
    """
    rows: List[Dict[str, float]] = []
    for i in range(keypoints_sequence.shape[0]):
        row = extract_kinematics(keypoints_sequence[i])
        row["frame"] = i
        row["time_s"] = i / fps
        rows.append(row)

    return pd.DataFrame(rows)


def detect_lip_smack(
    timeseries: pd.DataFrame,
    fps: float = 30.0,
    min_freq: float = 3.0,
    max_freq: float = 8.0,
    threshold_std: float = 2.0,
    min_duration_frames: int = 3,
) -> List[Tuple[int, int]]:
    """Detect lip-smacking episodes from mouth aperture timeseries.

    Lip smacking is characterized by rapid, rhythmic mouth opening/closing
    in the 3-8 Hz range (typical for macaques and other Old World monkeys).

    Args:
        timeseries: DataFrame from extract_timeseries with 'mouth_aperture' column.
        fps: Frames per second of the video.
        min_freq: Minimum frequency of lip-smack oscillation (Hz).
        max_freq: Maximum frequency of lip-smack oscillation (Hz).
        threshold_std: Number of standard deviations above mean for detection.
        min_duration_frames: Minimum number of frames for a valid episode.

    Returns:
        List of (start_frame, end_frame) tuples for detected episodes.
    """
    signal = timeseries["mouth_aperture"].values

    if len(signal) < 10 or fps < 2 * max_freq:
        return []

    # Bandpass filter to isolate lip-smack frequency range
    nyquist = fps / 2.0
    low = min_freq / nyquist
    high = min(max_freq / nyquist, 0.99)

    if low >= high or low <= 0:
        return []

    sos = butter(4, [low, high], btype="band", output="sos")
    filtered = sosfiltfilt(sos, signal)

    # Compute envelope via absolute value
    envelope = np.abs(filtered)

    # Threshold: mean + threshold_std * std
    thresh = envelope.mean() + threshold_std * envelope.std()

    # Find contiguous regions above threshold
    above = envelope > thresh
    episodes: List[Tuple[int, int]] = []
    start = None

    for i, val in enumerate(above):
        if val and start is None:
            start = i
        elif not val and start is not None:
            if i - start >= min_duration_frames:
                episodes.append((start, i - 1))
            start = None

    # Handle episode extending to end
    if start is not None and len(above) - start >= min_duration_frames:
        episodes.append((start, len(above) - 1))

    return episodes
