"""Tests for the PrimateFace CLI."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from primateface.cli import main


class TestCLIModels:
    """Test 'primateface models' command."""

    def test_models_list(self, capsys):
        ret = main(["models", "list"])
        assert ret == 0
        captured = capsys.readouterr()
        assert "fparodi/primateface-models" in captured.out
        assert "detection" in captured.out
        assert "pose" in captured.out
        assert "hrnet" in captured.out.lower() or "HRNet" in captured.out

    def test_models_default_action(self, capsys):
        ret = main(["models"])
        assert ret == 0


class TestCLIHelp:
    """Test CLI help and no-args behavior."""

    def test_no_args_returns_zero(self, capsys):
        ret = main([])
        assert ret == 0

    def test_analyze_missing_input(self):
        """analyze without input should fail."""
        try:
            main(["analyze"])
            assert False, "Should have raised SystemExit"
        except SystemExit as e:
            assert e.code == 2  # argparse error


def _make_mock_face(score: float = 0.95) -> MagicMock:
    """Create a mock Face object for CLI tests."""
    face = MagicMock()
    face.score = score
    face.bbox = np.array([100, 50, 200, 150], dtype=np.float32)
    face.keypoints = np.random.rand(68, 3).astype(np.float32)
    face.head_pose = (5.0, -3.0, 1.0)
    face.symmetry = 0.012
    face.kinematics = {
        "mouth_aperture": 0.15,
        "mouth_width": 0.8,
        "interocular_distance": 60.0,
    }
    face.__repr__ = lambda self: f"Face(score={score:.2f}, bbox=[100, 50, 200, 150], landmarks=68)"
    return face


class TestCLIAnalyzeSingleImage:
    """Test 'primateface analyze image.jpg' with mocked PrimateFace."""

    @pytest.fixture
    def mock_pf(self):
        """Patch PrimateFace so it doesn't load real models."""
        mock_instance = MagicMock()
        mock_instance.analyze.return_value = [_make_mock_face(0.95)]
        with patch("primateface.PrimateFace", return_value=mock_instance) as mock_cls:
            mock_cls.draw = MagicMock()
            yield mock_cls, mock_instance

    def test_analyze_single_image(self, mock_pf, capsys, tmp_path):
        # Create a dummy image file
        img = tmp_path / "test.jpg"
        img.touch()

        ret = main(["analyze", str(img)])
        assert ret == 0
        captured = capsys.readouterr()
        assert "1 face(s)" in captured.out
        assert "head_pose" in captured.out
        assert "symmetry" in captured.out

    def test_analyze_with_output(self, mock_pf, capsys, tmp_path):
        img = tmp_path / "test.jpg"
        img.touch()
        out = tmp_path / "result.jpg"

        ret = main(["analyze", str(img), "--output", str(out)])
        assert ret == 0
        captured = capsys.readouterr()
        assert "Visualization saved" in captured.out
        # draw() should have been called
        mock_pf[0].draw.assert_called_once()

    def test_analyze_no_faces(self, mock_pf, capsys, tmp_path):
        mock_pf[1].analyze.return_value = []
        img = tmp_path / "test.jpg"
        img.touch()

        ret = main(["analyze", str(img)])
        assert ret == 0
        captured = capsys.readouterr()
        assert "0 face(s)" in captured.out

    def test_analyze_nonexistent_file(self, mock_pf):
        mock_pf[1].analyze.side_effect = FileNotFoundError("not found")
        with pytest.raises(FileNotFoundError):
            main(["analyze", "/nonexistent/image.jpg"])


class TestCLIAnalyzeDirectory:
    """Test 'primateface analyze ./images/' with mocked PrimateFace."""

    @pytest.fixture
    def mock_pf(self):
        mock_instance = MagicMock()
        mock_instance.analyze.return_value = [_make_mock_face(0.9)]
        with patch("primateface.PrimateFace", return_value=mock_instance) as mock_cls:
            mock_cls.draw = MagicMock()
            yield mock_cls, mock_instance

    def test_analyze_directory(self, mock_pf, capsys, tmp_path):
        # Create dummy images
        (tmp_path / "img1.jpg").touch()
        (tmp_path / "img2.png").touch()
        (tmp_path / "not_image.txt").touch()

        ret = main(["analyze", str(tmp_path)])
        assert ret == 0
        captured = capsys.readouterr()
        # Should process 2 images (jpg + png), skip txt
        assert "img1.jpg" in captured.out
        assert "img2.png" in captured.out
        assert "not_image.txt" not in captured.out

    def test_analyze_directory_with_output(self, mock_pf, capsys, tmp_path):
        (tmp_path / "img1.jpg").touch()
        out_dir = tmp_path / "results"

        ret = main(["analyze", str(tmp_path), "--output", str(out_dir)])
        assert ret == 0
        # draw should have been called for the image
        mock_pf[0].draw.assert_called()

    def test_analyze_empty_directory(self, mock_pf, capsys, tmp_path):
        ret = main(["analyze", str(tmp_path)])
        assert ret == 1  # should fail with "No images found"
        captured = capsys.readouterr()
        assert "No images found" in captured.err


class TestCLIAnalyzeOptions:
    """Test CLI option passing."""

    @pytest.fixture
    def mock_pf(self):
        mock_instance = MagicMock()
        mock_instance.analyze.return_value = [_make_mock_face()]
        with patch("primateface.PrimateFace", return_value=mock_instance) as mock_cls:
            mock_cls.draw = MagicMock()
            yield mock_cls, mock_instance

    def test_pose_model_option(self, mock_pf, tmp_path):
        img = tmp_path / "test.jpg"
        img.touch()
        main(["analyze", str(img), "--pose-model", "vitpose"])
        mock_pf[0].assert_called_once()
        call_kwargs = mock_pf[0].call_args
        assert call_kwargs.kwargs.get("pose_model") == "vitpose" or \
               call_kwargs[1].get("pose_model") == "vitpose"

    def test_det_threshold_option(self, mock_pf, tmp_path):
        img = tmp_path / "test.jpg"
        img.touch()
        main(["analyze", str(img), "--det-threshold", "0.8"])
        call_kwargs = mock_pf[0].call_args
        assert call_kwargs.kwargs.get("det_threshold") == 0.8 or \
               call_kwargs[1].get("det_threshold") == 0.8

    def test_device_option(self, mock_pf, tmp_path):
        img = tmp_path / "test.jpg"
        img.touch()
        main(["analyze", str(img), "--device", "cpu"])
        call_kwargs = mock_pf[0].call_args
        assert call_kwargs.kwargs.get("device") == "cpu" or \
               call_kwargs[1].get("device") == "cpu"
