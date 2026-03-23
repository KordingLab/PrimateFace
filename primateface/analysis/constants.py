"""Landmark index definitions for 68-point dlib face landmarks.

Provides named groups, left-right symmetric pairs, midline landmarks,
and 3D reference coordinates for head pose estimation.
"""

from typing import Dict, List, Tuple

import numpy as np

# ============================================================================
# Anatomical region groups (0-indexed, dlib 68-point standard)
# ============================================================================

JAW: List[int] = list(range(0, 17))            # 0-16: jaw contour
RIGHT_EYEBROW: List[int] = list(range(17, 22)) # 17-21
LEFT_EYEBROW: List[int] = list(range(22, 27))  # 22-26
NOSE_BRIDGE: List[int] = list(range(27, 31))    # 27-30
NOSE_TIP: List[int] = list(range(31, 36))       # 31-35
RIGHT_EYE: List[int] = list(range(36, 42))      # 36-41
LEFT_EYE: List[int] = list(range(42, 48))       # 42-47
OUTER_MOUTH: List[int] = list(range(48, 60))    # 48-59
INNER_MOUTH: List[int] = list(range(60, 68))    # 60-67

# ============================================================================
# Key individual landmarks
# ============================================================================

UPPER_LIP_TOP: int = 51         # outer upper lip center
LOWER_LIP_BOTTOM: int = 57     # outer lower lip center
INNER_UPPER_LIP: int = 62      # inner upper lip center
INNER_LOWER_LIP: int = 66      # inner lower lip center
LEFT_MOUTH_CORNER: int = 48
RIGHT_MOUTH_CORNER: int = 54
NOSE_TIP_CENTER: int = 30
CHIN: int = 8
RIGHT_EYE_OUTER: int = 36
RIGHT_EYE_INNER: int = 39
LEFT_EYE_INNER: int = 42
LEFT_EYE_OUTER: int = 45

# ============================================================================
# Left-right symmetric pairs for fluctuating asymmetry
# Format: (left_idx, right_idx) where "left" = subject's left (viewer's right)
# ============================================================================

SYMMETRIC_PAIRS: List[Tuple[int, int]] = [
    # Jaw (mirrored around chin=8)
    (0, 16), (1, 15), (2, 14), (3, 13), (4, 12), (5, 11), (6, 10), (7, 9),
    # Eyebrows
    (17, 26), (18, 25), (19, 24), (20, 23), (21, 22),
    # Eyes
    (36, 45), (37, 44), (38, 43), (39, 42), (40, 47), (41, 46),
    # Nose tip
    (31, 35), (32, 34),
    # Mouth outer
    (48, 54), (49, 53), (50, 52), (55, 59), (56, 58),
    # Mouth inner
    (60, 64), (61, 63), (65, 67),
]

# Symmetric pairs grouped by region (for per-region symmetry)
SYMMETRIC_PAIRS_BY_REGION: Dict[str, List[Tuple[int, int]]] = {
    "jaw": [(0, 16), (1, 15), (2, 14), (3, 13), (4, 12), (5, 11), (6, 10), (7, 9)],
    "eyebrows": [(17, 26), (18, 25), (19, 24), (20, 23), (21, 22)],
    "eyes": [(36, 45), (37, 44), (38, 43), (39, 42), (40, 47), (41, 46)],
    "nose": [(31, 35), (32, 34)],
    "mouth": [(48, 54), (49, 53), (50, 52), (55, 59), (56, 58), (60, 64), (61, 63), (65, 67)],
}

# Midline landmarks (for fitting the symmetry axis)
MIDLINE: List[int] = [27, 28, 29, 30, 33, 51, 62, 66, 57, 8]

# ============================================================================
# Eye vertical pairs (for eye aperture — upper ↔ lower eyelid)
# ============================================================================

RIGHT_EYE_VERTICAL_PAIRS: List[Tuple[int, int]] = [
    (37, 41),  # right eye upper-right ↔ lower-right
    (38, 40),  # right eye upper-left ↔ lower-left
]

LEFT_EYE_VERTICAL_PAIRS: List[Tuple[int, int]] = [
    (43, 47),  # left eye upper-right ↔ lower-right
    (44, 46),  # left eye upper-left ↔ lower-left
]

# ============================================================================
# 3D reference model for head pose estimation via solvePnP
# Generic primate face, approximate. Units are arbitrary but proportional.
# Based on dlib 68-point canonical coordinates.
# ============================================================================

POSE_REFERENCE_3D: np.ndarray = np.array([
    [0.0, 0.0, 0.0],          # Nose tip (30)
    [0.0, -63.6, -12.5],      # Chin (8)
    [-43.3, 32.7, -26.0],     # Left eye outer (45)
    [43.3, 32.7, -26.0],      # Right eye outer (36)
    [-28.9, -28.9, -24.1],    # Left mouth corner (48)
    [28.9, -28.9, -24.1],     # Right mouth corner (54)
], dtype=np.float64)

# Indices into the 68-point array corresponding to POSE_REFERENCE_3D rows
POSE_LANDMARK_INDICES: List[int] = [
    NOSE_TIP_CENTER,     # 30
    CHIN,                # 8
    LEFT_EYE_OUTER,      # 45
    RIGHT_EYE_OUTER,     # 36
    LEFT_MOUTH_CORNER,   # 48
    RIGHT_MOUTH_CORNER,  # 54
]

# ============================================================================
# Total number of landmarks
# ============================================================================

NUM_LANDMARKS: int = 68
