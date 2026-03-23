"""Tests for the analysis module.

Uses synthetic keypoints with known geometry to verify correctness.
"""

import unittest

import numpy as np
import pandas as pd

from primateface.analysis.constants import (
    MIDLINE,
    NUM_LANDMARKS,
    SYMMETRIC_PAIRS,
    SYMMETRIC_PAIRS_BY_REGION,
)
from primateface.analysis.kinematics import (
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
from primateface.analysis.symmetry import facial_symmetry, per_region_symmetry
from primateface.analysis.head_pose import estimate_head_pose
from primateface.analysis.quality import face_quality
from primateface.analysis.utils import (
    fit_midline,
    get_eye_centers,
    interocular_distance,
    point_to_line_distance,
    visibility_ratio,
)


def _make_symmetric_face() -> np.ndarray:
    """Create a perfectly symmetric synthetic face (68, 2).

    Layout: face centered at (100, 100), roughly 80px wide.
    Landmarks are placed at approximate dlib 68-point positions.
    """
    kpts = np.zeros((NUM_LANDMARKS, 2), dtype=np.float64)

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

    # Nose tip (31-35)
    kpts[31] = [92, 88]
    kpts[32] = [95, 90]
    kpts[33] = [100, 92]
    kpts[34] = [105, 90]
    kpts[35] = [108, 88]

    # Right eye (36-41)
    kpts[36] = [75, 72]
    kpts[37] = [80, 68]
    kpts[38] = [85, 68]
    kpts[39] = [90, 72]
    kpts[40] = [85, 75]
    kpts[41] = [80, 75]

    # Left eye (42-47)
    kpts[42] = [110, 72]
    kpts[43] = [115, 68]
    kpts[44] = [120, 68]
    kpts[45] = [125, 72]
    kpts[46] = [120, 75]
    kpts[47] = [115, 75]

    # Outer mouth (48-59)
    kpts[48] = [85, 110]
    kpts[49] = [90, 107]
    kpts[50] = [95, 105]
    kpts[51] = [100, 104]
    kpts[52] = [105, 105]
    kpts[53] = [110, 107]
    kpts[54] = [115, 110]
    kpts[55] = [110, 115]
    kpts[56] = [105, 117]
    kpts[57] = [100, 118]
    kpts[58] = [95, 117]
    kpts[59] = [90, 115]

    # Inner mouth (60-67)
    kpts[60] = [90, 110]
    kpts[61] = [95, 108]
    kpts[62] = [100, 107]
    kpts[63] = [105, 108]
    kpts[64] = [110, 110]
    kpts[65] = [105, 113]
    kpts[66] = [100, 114]
    kpts[67] = [95, 113]

    return kpts


class TestConstants(unittest.TestCase):
    """Verify landmark index constants are valid."""

    def test_symmetric_pairs_in_range(self):
        for left, right in SYMMETRIC_PAIRS:
            self.assertGreaterEqual(left, 0)
            self.assertLess(left, NUM_LANDMARKS)
            self.assertGreaterEqual(right, 0)
            self.assertLess(right, NUM_LANDMARKS)

    def test_symmetric_pairs_not_equal(self):
        for left, right in SYMMETRIC_PAIRS:
            self.assertNotEqual(left, right)

    def test_region_pairs_cover_all_pairs(self):
        all_from_regions = []
        for pairs in SYMMETRIC_PAIRS_BY_REGION.values():
            all_from_regions.extend(pairs)
        self.assertEqual(sorted(all_from_regions), sorted(SYMMETRIC_PAIRS))

    def test_midline_in_range(self):
        for idx in MIDLINE:
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, NUM_LANDMARKS)


class TestUtils(unittest.TestCase):
    """Test shared utility functions."""

    def setUp(self):
        self.kpts = _make_symmetric_face()

    def test_eye_centers(self):
        right_c, left_c = get_eye_centers(self.kpts)
        self.assertEqual(right_c.shape, (2,))
        self.assertEqual(left_c.shape, (2,))
        # Left eye center should be to the right of right eye center
        self.assertGreater(left_c[0], right_c[0])

    def test_interocular_distance_positive(self):
        iod = interocular_distance(self.kpts)
        self.assertGreater(iod, 0)

    def test_visibility_ratio_all_visible(self):
        kpts_3 = np.column_stack([self.kpts, np.full(NUM_LANDMARKS, 2.0)])
        self.assertAlmostEqual(visibility_ratio(kpts_3), 1.0)

    def test_visibility_ratio_half_visible(self):
        vis = np.zeros(NUM_LANDMARKS)
        vis[:34] = 2.0
        kpts_3 = np.column_stack([self.kpts, vis])
        self.assertAlmostEqual(visibility_ratio(kpts_3), 0.5)

    def test_fit_midline_returns_valid(self):
        centroid, direction = fit_midline(self.kpts)
        self.assertEqual(centroid.shape, (2,))
        self.assertEqual(direction.shape, (2,))
        # Direction should be approximately unit length
        self.assertAlmostEqual(np.linalg.norm(direction), 1.0, places=5)

    def test_point_to_line_distance(self):
        # Horizontal line at y=0, direction (1, 0)
        d = point_to_line_distance(
            np.array([5.0, 3.0]),
            np.array([0.0, 0.0]),
            np.array([1.0, 0.0]),
        )
        self.assertAlmostEqual(abs(d), 3.0)


class TestKinematics(unittest.TestCase):
    """Test kinematic feature extraction."""

    def setUp(self):
        self.kpts = _make_symmetric_face()

    def test_mouth_aperture_positive(self):
        val = mouth_aperture(self.kpts)
        self.assertGreater(val, 0)

    def test_mouth_width_positive(self):
        val = mouth_width(self.kpts)
        self.assertGreater(val, 0)

    def test_mouth_aspect_ratio(self):
        ratio = mouth_aspect_ratio(self.kpts)
        self.assertGreater(ratio, 0)
        # For our synthetic face, mouth is wider than tall
        self.assertLess(ratio, 1.0)

    def test_eye_aperture_both(self):
        right, left = eye_aperture(self.kpts, side="both")
        self.assertGreater(right, 0)
        self.assertGreater(left, 0)

    def test_eye_aperture_single(self):
        val = eye_aperture(self.kpts, side="right")
        self.assertIsInstance(val, float)
        self.assertGreater(val, 0)

    def test_brow_height_both(self):
        right, left = brow_height(self.kpts, side="both")
        self.assertGreater(right, 0)
        self.assertGreater(left, 0)

    def test_face_geometry_positive(self):
        self.assertGreater(face_height(self.kpts), 0)
        self.assertGreater(face_width(self.kpts), 0)
        self.assertGreater(face_aspect_ratio(self.kpts), 0)
        self.assertGreater(jaw_width(self.kpts), 0)
        self.assertGreater(nose_length(self.kpts), 0)
        self.assertGreater(eye_to_mouth(self.kpts), 0)

    def test_face_aspect_ratio_range(self):
        # For a realistic face, aspect ratio should be > 0.5 and < 3.0
        ratio = face_aspect_ratio(self.kpts)
        self.assertGreater(ratio, 0.5)
        self.assertLess(ratio, 3.0)

    def test_extract_kinematics_keys(self):
        result = extract_kinematics(self.kpts)
        expected_keys = {
            "mouth_aperture", "mouth_width", "mouth_aspect_ratio",
            "right_eye_aperture", "left_eye_aperture",
            "right_brow_height", "left_brow_height",
            "face_height", "face_width", "face_aspect_ratio",
            "jaw_width", "nose_length", "eye_to_mouth",
            "interocular_distance",
        }
        self.assertEqual(set(result.keys()), expected_keys)

    def test_extract_timeseries(self):
        seq = np.stack([self.kpts] * 10, axis=0)
        df = extract_timeseries(seq, fps=30.0)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 10)
        self.assertIn("frame", df.columns)
        self.assertIn("time_s", df.columns)
        self.assertIn("mouth_aperture", df.columns)

    def test_detect_lip_smack_no_signal(self):
        # Static face = no lip smack
        seq = np.stack([self.kpts] * 60, axis=0)
        df = extract_timeseries(seq, fps=30.0)
        episodes = detect_lip_smack(df, fps=30.0)
        self.assertEqual(len(episodes), 0)


class TestSymmetry(unittest.TestCase):
    """Test fluctuating asymmetry computation."""

    def setUp(self):
        self.kpts = _make_symmetric_face()

    def test_symmetric_face_low_fa(self):
        # Nearly symmetric face should have very low FA
        fa = facial_symmetry(self.kpts, method="midline")
        self.assertGreaterEqual(fa, 0)
        self.assertLess(fa, 0.1)  # low threshold for near-symmetric

    def test_asymmetric_face_higher_fa(self):
        kpts = self.kpts.copy()
        # Shift the entire left jaw outward
        kpts[0:4, 0] -= 10
        fa = facial_symmetry(kpts, method="midline")
        fa_orig = facial_symmetry(self.kpts, method="midline")
        self.assertGreater(fa, fa_orig)

    def test_procrustes_method(self):
        fa = facial_symmetry(self.kpts, method="procrustes")
        self.assertGreaterEqual(fa, 0)

    def test_per_region_symmetry(self):
        result = per_region_symmetry(self.kpts)
        self.assertIn("jaw", result)
        self.assertIn("eyes", result)
        self.assertIn("mouth", result)
        for val in result.values():
            self.assertGreaterEqual(val, 0)

    def test_invalid_method_raises(self):
        with self.assertRaises(ValueError):
            facial_symmetry(self.kpts, method="invalid")


class TestHeadPose(unittest.TestCase):
    """Test head pose estimation."""

    def setUp(self):
        self.kpts = _make_symmetric_face()

    def test_returns_three_angles(self):
        yaw, pitch, roll = estimate_head_pose(self.kpts, image_size=(200, 200))
        self.assertIsInstance(yaw, float)
        self.assertIsInstance(pitch, float)
        self.assertIsInstance(roll, float)

    def test_frontal_face_small_angles(self):
        yaw, pitch, roll = estimate_head_pose(self.kpts, image_size=(200, 200))
        # Frontal synthetic face should have small angles
        self.assertLess(abs(roll), 30)


class TestQuality(unittest.TestCase):
    """Test face quality assessment."""

    def test_quality_with_valid_image(self):
        # Create a simple test image
        img = np.random.randint(50, 200, (200, 200, 3), dtype=np.uint8)
        bbox = np.array([60, 50, 140, 160])
        kpts = _make_symmetric_face()
        vis = np.full(NUM_LANDMARKS, 2.0)
        kpts_3 = np.column_stack([kpts, vis])

        result = face_quality(img, bbox, kpts_3)
        self.assertIn("blur", result)
        self.assertIn("size", result)
        self.assertIn("visibility", result)
        self.assertIn("brightness", result)
        self.assertIn("score", result)
        self.assertGreaterEqual(result["score"], 0)
        self.assertLessEqual(result["score"], 1)

    def test_quality_with_all_visible(self):
        img = np.full((200, 200, 3), 128, dtype=np.uint8)
        bbox = np.array([60, 50, 140, 160])
        kpts = _make_symmetric_face()
        vis = np.full(NUM_LANDMARKS, 2.0)
        kpts_3 = np.column_stack([kpts, vis])

        result = face_quality(img, bbox, kpts_3)
        self.assertAlmostEqual(result["visibility"], 1.0)

    def test_quality_empty_bbox(self):
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        bbox = np.array([0, 0, 0, 0])
        kpts = np.zeros((NUM_LANDMARKS, 3))
        result = face_quality(img, bbox, kpts)
        self.assertAlmostEqual(result["score"], 0.0, places=2)


if __name__ == "__main__":
    unittest.main()
