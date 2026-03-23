"""Shared test fixtures for primateface tests."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# Ensure repo root is on sys.path so `analysis`, `demos`, `primateface` are importable
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _make_symmetric_face_68x2() -> np.ndarray:
    """Create a synthetic 68-point face, shape (68, 2).

    Layout: face centered at (100, 100), roughly 80px wide.
    Reuses the pattern from analysis/test_analysis.py.
    """
    kpts = np.zeros((68, 2), dtype=np.float64)

    # Jaw (0-16): semicircle
    for i in range(17):
        angle = np.pi * i / 16
        kpts[i] = [100 + 40 * np.cos(np.pi - angle), 100 + 50 * np.sin(angle)]

    # Right eyebrow (17-21)
    for i, idx in enumerate(range(17, 22)):
        kpts[idx] = [70 + i * 5, 60]

    # Left eyebrow (22-26)
    for i, idx in enumerate(range(22, 27)):
        kpts[idx] = [110 + i * 5, 60]

    # Nose bridge (27-30)
    for i, idx in enumerate(range(27, 31)):
        kpts[idx] = [100, 70 + i * 5]

    # Nose base (31-35)
    kpts[31] = [90, 90]
    kpts[32] = [95, 92]
    kpts[33] = [100, 93]
    kpts[34] = [105, 92]
    kpts[35] = [110, 90]

    # Right eye (36-41)
    kpts[36] = [75, 70]
    kpts[37] = [80, 67]
    kpts[38] = [85, 67]
    kpts[39] = [90, 70]
    kpts[40] = [85, 73]
    kpts[41] = [80, 73]

    # Left eye (42-47)
    kpts[42] = [110, 70]
    kpts[43] = [115, 67]
    kpts[44] = [120, 67]
    kpts[45] = [125, 70]
    kpts[46] = [120, 73]
    kpts[47] = [115, 73]

    # Outer mouth (48-59)
    for i, idx in enumerate(range(48, 60)):
        angle = 2 * np.pi * i / 12
        kpts[idx] = [100 + 12 * np.cos(angle), 115 + 6 * np.sin(angle)]

    # Inner mouth (60-67)
    for i, idx in enumerate(range(60, 68)):
        angle = 2 * np.pi * i / 8
        kpts[idx] = [100 + 8 * np.cos(angle), 115 + 4 * np.sin(angle)]

    return kpts


@pytest.fixture
def synthetic_keypoints_68x2() -> np.ndarray:
    """Synthetic (68, 2) face keypoints."""
    return _make_symmetric_face_68x2()


@pytest.fixture
def synthetic_keypoints_68x3() -> np.ndarray:
    """Synthetic (68, 3) face keypoints with visibility/score column."""
    kpts_2d = _make_symmetric_face_68x2()
    scores = np.ones(68, dtype=np.float64) * 0.95
    return np.column_stack([kpts_2d, scores]).astype(np.float32)


@pytest.fixture
def dummy_image() -> np.ndarray:
    """200x200 random BGR uint8 image."""
    rng = np.random.RandomState(42)
    return rng.randint(0, 255, (200, 200, 3), dtype=np.uint8)


@pytest.fixture
def sample_bbox() -> np.ndarray:
    """Bounding box enclosing the synthetic face."""
    return np.array([60.0, 40.0, 140.0, 160.0], dtype=np.float32)


@pytest.fixture
def mock_processor():
    """Mock PrimateFaceProcessor that returns deterministic results."""
    kpts_2d = _make_symmetric_face_68x2()
    n_dets = 2

    proc = MagicMock()

    # detect_primates returns (bboxes, scores)
    bboxes = np.array(
        [[60, 40, 140, 160], [20, 10, 80, 90]], dtype=np.float32
    )
    scores = np.array([0.95, 0.72], dtype=np.float32)
    proc.detect_primates.return_value = (bboxes, scores)

    # estimate_poses returns an object with .keypoints and .keypoint_scores
    pose_result = MagicMock()
    pose_result.keypoints = np.stack([kpts_2d, kpts_2d]).astype(np.float32)
    pose_result.keypoint_scores = np.ones((n_dets, 68), dtype=np.float32) * 0.9
    proc.estimate_poses.return_value = pose_result

    return proc
