"""Tests for the Face dataclass."""

import numpy as np
import pytest

from primateface.face import Face


class TestFaceConstruction:
    """Test Face object creation."""

    def test_basic_construction(
        self, synthetic_keypoints_68x3, dummy_image, sample_bbox
    ):
        face = Face(
            bbox=sample_bbox,
            score=0.95,
            keypoints=synthetic_keypoints_68x3,
            _image=dummy_image,
            _image_size=(200, 200),
        )
        assert face.score == 0.95
        assert face.bbox.shape == (4,)
        assert face.keypoints.shape == (68, 3)

    def test_repr(self, synthetic_keypoints_68x3, dummy_image, sample_bbox):
        face = Face(
            bbox=sample_bbox,
            score=0.95,
            keypoints=synthetic_keypoints_68x3,
            _image=dummy_image,
            _image_size=(200, 200),
        )
        r = repr(face)
        assert "Face(" in r
        assert "score=0.95" in r
        assert "landmarks=68" in r

    def test_to_dict(self, synthetic_keypoints_68x3, dummy_image, sample_bbox):
        face = Face(
            bbox=sample_bbox,
            score=0.95,
            keypoints=synthetic_keypoints_68x3,
            _image=dummy_image,
            _image_size=(200, 200),
        )
        d = face.to_dict()
        assert isinstance(d, dict)
        assert isinstance(d["bbox"], list)
        assert isinstance(d["keypoints"], list)
        assert d["score"] == 0.95


class TestFaceCrop:
    """Test face crop extraction."""

    def test_crop_shape(self, synthetic_keypoints_68x3, dummy_image, sample_bbox):
        face = Face(
            bbox=sample_bbox,
            score=0.95,
            keypoints=synthetic_keypoints_68x3,
            _image=dummy_image,
            _image_size=(200, 200),
        )
        crop = face.crop
        # bbox is [60, 40, 140, 160] → 80x120 region
        assert crop.shape == (120, 80, 3)

    def test_crop_clamps_to_image_bounds(
        self, synthetic_keypoints_68x3, dummy_image
    ):
        """Bbox extending beyond image should be clamped."""
        bbox = np.array([-10.0, -10.0, 50.0, 50.0], dtype=np.float32)
        face = Face(
            bbox=bbox,
            score=0.9,
            keypoints=synthetic_keypoints_68x3,
            _image=dummy_image,
            _image_size=(200, 200),
        )
        crop = face.crop
        assert crop.shape == (50, 50, 3)  # clamped from [-10,-10] to [0,0]


class TestFaceAnalysisProperties:
    """Test lazy analysis properties using synthetic data."""

    @pytest.fixture
    def face(self, synthetic_keypoints_68x3, dummy_image, sample_bbox):
        return Face(
            bbox=sample_bbox,
            score=0.95,
            keypoints=synthetic_keypoints_68x3,
            _image=dummy_image,
            _image_size=(200, 200),
        )

    def test_head_pose_returns_3_tuple(self, face):
        yaw, pitch, roll = face.head_pose
        assert isinstance(yaw, float)
        assert isinstance(pitch, float)
        assert isinstance(roll, float)

    def test_symmetry_returns_float(self, face):
        sym = face.symmetry
        assert isinstance(sym, float)
        assert sym >= 0.0

    def test_kinematics_returns_dict(self, face):
        kin = face.kinematics
        assert isinstance(kin, dict)
        expected_keys = {
            "mouth_aperture", "mouth_width", "mouth_aspect_ratio",
            "right_eye_aperture", "left_eye_aperture",
            "right_brow_height", "left_brow_height",
            "face_height", "face_width", "face_aspect_ratio",
            "jaw_width", "nose_length", "eye_to_mouth",
            "interocular_distance",
        }
        assert expected_keys.issubset(kin.keys())

    def test_mouth_aperture_delegates(self, face):
        ma = face.mouth_aperture
        assert isinstance(ma, float)
        assert ma == face.kinematics["mouth_aperture"]

    def test_eye_aperture_returns_tuple(self, face):
        right, left = face.eye_aperture
        assert isinstance(right, float)
        assert isinstance(left, float)

    def test_brow_position_returns_tuple(self, face):
        right, left = face.brow_position
        assert isinstance(right, float)
        assert isinstance(left, float)

    def test_interocular_distance(self, face):
        iod = face.interocular_distance
        assert isinstance(iod, float)
        assert iod > 0

    def test_quality_returns_dict(self, face):
        q = face.quality
        assert isinstance(q, dict)
        for key in ("blur", "size", "visibility", "brightness", "score"):
            assert key in q

    def test_region_symmetry_returns_dict(self, face):
        rs = face.region_symmetry
        assert isinstance(rs, dict)
        for key in ("jaw", "eyebrows", "eyes", "nose", "mouth"):
            assert key in rs

    def test_lazy_evaluation_caches(self, face):
        """Accessing kinematics twice should return the same object."""
        kin1 = face.kinematics
        kin2 = face.kinematics
        assert kin1 is kin2


class TestFaceEmbedding:
    """Test face embedding and verify."""

    def test_embedding_raises_without_model(
        self, synthetic_keypoints_68x3, dummy_image, sample_bbox
    ):
        """Embedding should raise if no model configured."""
        face = Face(
            bbox=sample_bbox,
            score=0.95,
            keypoints=synthetic_keypoints_68x3,
            _image=dummy_image,
            _image_size=(200, 200),
        )
        with pytest.raises(RuntimeError, match="No embedding model"):
            _ = face.embedding

    def test_embedding_with_mock_fn(
        self, synthetic_keypoints_68x3, dummy_image, sample_bbox
    ):
        """Embedding should work when _embedding_fn is provided."""
        import numpy as np

        fake_embedding = np.random.randn(512).astype(np.float32)
        face = Face(
            bbox=sample_bbox,
            score=0.95,
            keypoints=synthetic_keypoints_68x3,
            _image=dummy_image,
            _image_size=(200, 200),
            _embedding_fn=lambda crop: fake_embedding,
        )
        emb = face.embedding
        assert emb.shape == (512,)
        np.testing.assert_array_equal(emb, fake_embedding)

    def test_verify_same_face(
        self, synthetic_keypoints_68x3, dummy_image, sample_bbox
    ):
        """Verifying a face against itself should return True."""
        import numpy as np

        fake_emb = np.random.randn(512).astype(np.float32)
        face = Face(
            bbox=sample_bbox,
            score=0.95,
            keypoints=synthetic_keypoints_68x3,
            _image=dummy_image,
            _image_size=(200, 200),
            _embedding_fn=lambda crop: fake_emb,
        )
        is_same, distance = face.verify(face)
        assert is_same is True
        assert distance < 0.01

    def test_verify_different_faces(
        self, synthetic_keypoints_68x3, dummy_image, sample_bbox
    ):
        """Verifying different embeddings should return larger distance."""
        import numpy as np

        emb1 = np.array([1.0, 0.0, 0.0] + [0.0] * 509, dtype=np.float32)
        emb2 = np.array([0.0, 1.0, 0.0] + [0.0] * 509, dtype=np.float32)

        face1 = Face(
            bbox=sample_bbox, score=0.9, keypoints=synthetic_keypoints_68x3,
            _image=dummy_image, _image_size=(200, 200),
            _embedding_fn=lambda crop: emb1,
        )
        face2 = Face(
            bbox=sample_bbox, score=0.9, keypoints=synthetic_keypoints_68x3,
            _image=dummy_image, _image_size=(200, 200),
            _embedding_fn=lambda crop: emb2,
        )
        is_same, distance = face1.verify(face2)
        assert is_same is False
        assert distance > 0.5
