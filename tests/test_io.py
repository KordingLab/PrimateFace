"""Tests for primateface.io export utilities."""

import json

import pandas as pd
import pytest

from primateface.face import Face
from primateface.io import (
    DLIB_68_EDGES,
    from_coco_json,
    from_dlc,
    to_coco_json,
    to_csv,
    to_dataframe,
    to_dlc_csv,
    to_dlc_h5,
)


@pytest.fixture
def sample_faces(synthetic_keypoints_68x3, dummy_image, sample_bbox):
    """Two Face objects for testing exports."""
    faces = []
    for score in [0.95, 0.72]:
        faces.append(Face(
            bbox=sample_bbox.copy(),
            score=score,
            keypoints=synthetic_keypoints_68x3.copy(),
            _image=dummy_image,
            _image_size=(200, 200),
        ))
    return faces


class TestToDataFrame:
    """Test to_dataframe conversion."""

    def test_returns_dataframe(self, sample_faces):
        df = to_dataframe(sample_faces)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_has_detection_columns(self, sample_faces):
        df = to_dataframe(sample_faces)
        for col in ("face_idx", "score", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"):
            assert col in df.columns

    def test_has_keypoint_columns(self, sample_faces):
        df = to_dataframe(sample_faces)
        assert "kpt_0_x" in df.columns
        assert "kpt_67_score" in df.columns

    def test_has_kinematic_columns(self, sample_faces):
        df = to_dataframe(sample_faces)
        assert "mouth_aperture" in df.columns
        assert "interocular_distance" in df.columns

    def test_has_head_pose_columns(self, sample_faces):
        df = to_dataframe(sample_faces)
        for col in ("yaw", "pitch", "roll"):
            assert col in df.columns

    def test_has_quality_columns(self, sample_faces):
        df = to_dataframe(sample_faces)
        assert "quality_blur" in df.columns
        assert "quality_score" in df.columns

    def test_has_symmetry_columns(self, sample_faces):
        df = to_dataframe(sample_faces)
        assert "symmetry_fa" in df.columns
        assert "symmetry_jaw" in df.columns

    def test_image_path_column(self, sample_faces):
        df = to_dataframe(sample_faces, image_path="monkey.jpg")
        assert "image" in df.columns
        assert df["image"].iloc[0] == "monkey.jpg"

    def test_exclude_features(self, sample_faces):
        df = to_dataframe(
            sample_faces,
            include_kinematics=False,
            include_head_pose=False,
            include_quality=False,
            include_symmetry=False,
        )
        assert "mouth_aperture" not in df.columns
        assert "yaw" not in df.columns
        assert "quality_blur" not in df.columns
        assert "symmetry_fa" not in df.columns


class TestToCsv:
    """Test CSV export."""

    def test_creates_file(self, sample_faces, tmp_path):
        out = to_csv(sample_faces, tmp_path / "test.csv")
        assert out.exists()

    def test_roundtrip(self, sample_faces, tmp_path):
        out = to_csv(sample_faces, tmp_path / "test.csv")
        df = pd.read_csv(out)
        assert len(df) == 2
        assert "score" in df.columns


class TestToCocoJson:
    """Test COCO JSON export."""

    def test_creates_file(self, sample_faces, tmp_path):
        out = to_coco_json(sample_faces, tmp_path / "test.json", image_path="img.jpg")
        assert out.exists()

    def test_valid_coco_structure(self, sample_faces, tmp_path):
        out = to_coco_json(sample_faces, tmp_path / "test.json", image_path="img.jpg")
        with open(out) as f:
            coco = json.load(f)
        assert "images" in coco
        assert "annotations" in coco
        assert "categories" in coco
        assert len(coco["annotations"]) == 2

    def test_coco_bbox_is_xywh(self, sample_faces, tmp_path):
        out = to_coco_json(sample_faces, tmp_path / "test.json")
        with open(out) as f:
            coco = json.load(f)
        bbox = coco["annotations"][0]["bbox"]
        assert len(bbox) == 4
        # bbox should be [x, y, w, h] where w, h > 0
        assert bbox[2] > 0
        assert bbox[3] > 0

    def test_keypoints_length(self, sample_faces, tmp_path):
        out = to_coco_json(sample_faces, tmp_path / "test.json")
        with open(out) as f:
            coco = json.load(f)
        kpts = coco["annotations"][0]["keypoints"]
        assert len(kpts) == 68 * 3


class TestFromCocoJson:
    """Test COCO JSON import."""

    def test_roundtrip(self, sample_faces, tmp_path):
        out = to_coco_json(sample_faces, tmp_path / "test.json", image_path="img.jpg")
        df = from_coco_json(out)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "kpt_0_x" in df.columns


class TestToDlcCsv:
    """Test DeepLabCut CSV export."""

    def test_creates_file(self, sample_faces, tmp_path):
        out = to_dlc_csv(sample_faces, tmp_path / "dlc.csv")
        assert out.exists()

    def test_multiindex_header(self, sample_faces, tmp_path):
        out = to_dlc_csv(sample_faces, tmp_path / "dlc.csv")
        df = pd.read_csv(out, header=[0, 1, 2], index_col=0)
        assert df.columns.nlevels == 3
        assert len(df) == 2


class TestDlib68Edges:
    """Test skeleton edge constant."""

    def test_edges_are_tuples(self):
        assert isinstance(DLIB_68_EDGES, list)
        for edge in DLIB_68_EDGES:
            assert len(edge) == 2
            assert all(0 <= idx < 68 for idx in edge)

    def test_edges_include_jaw(self):
        jaw_edges = [(i, i + 1) for i in range(16)]
        for e in jaw_edges:
            assert e in DLIB_68_EDGES

    def test_eyes_are_closed_loops(self):
        assert (41, 36) in DLIB_68_EDGES  # right eye closure
        assert (47, 42) in DLIB_68_EDGES  # left eye closure


class TestFromDlc:
    """Test DLC import."""

    def test_from_dlc_csv(self, sample_faces, tmp_path):
        out = to_dlc_csv(sample_faces, tmp_path / "dlc.csv", scorer="test")
        df = from_dlc(out)
        assert df.columns.nlevels == 3
        assert len(df) == 2

    def test_from_dlc_h5(self, sample_faces, tmp_path):
        pytest.importorskip("tables")  # pytables needed for HDF5
        out = to_dlc_h5(sample_faces, tmp_path / "dlc.h5", scorer="test")
        df = from_dlc(out)
        assert df.columns.nlevels == 3
        assert len(df) == 2

    def test_from_dlc_bad_extension(self, tmp_path):
        bad = tmp_path / "file.txt"
        bad.touch()
        with pytest.raises(ValueError, match="Unsupported file extension"):
            from_dlc(bad)


class TestSleapInterop:
    """Test SLEAP export/import (skipped if sleap-io not installed)."""

    @pytest.fixture(autouse=True)
    def _require_sleap(self):
        pytest.importorskip("sleap_io")

    def test_to_sleap_creates_file(self, sample_faces, tmp_path):
        from primateface.io import to_sleap
        out = to_sleap(sample_faces, tmp_path / "test.slp")
        assert out.exists()

    def test_sleap_roundtrip(self, sample_faces, tmp_path):
        from primateface.io import to_sleap, from_sleap
        slp_path = to_sleap(sample_faces, tmp_path / "test.slp")
        df = from_sleap(slp_path)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2  # 2 faces = 2 instances
        assert "kpt_0_x" in df.columns
        assert "kpt_67_x" in df.columns


class TestNwbInterop:
    """Test NWB export/import (skipped if pynwb/ndx-pose not installed)."""

    @pytest.fixture(autouse=True)
    def _require_nwb(self):
        pytest.importorskip("pynwb")
        pytest.importorskip("ndx_pose")

    def test_to_nwb_creates_file(self, sample_faces, tmp_path):
        from primateface.io import to_nwb
        out = to_nwb(sample_faces, tmp_path / "test.nwb")
        assert out.exists()

    def test_nwb_roundtrip(self, sample_faces, tmp_path):
        from primateface.io import to_nwb, from_nwb
        nwb_path = to_nwb(sample_faces, tmp_path / "test.nwb")
        df = from_nwb(nwb_path)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "kpt_0_x" in df.columns
