"""Tests for the model manager."""

from pathlib import Path
from unittest.mock import patch

from primateface._model_manager import ModelManager


class TestModelManager:
    """Test model download and path resolution."""

    @patch("primateface._model_manager.hf_hub_download")
    def test_ensure_models_calls_hf_download(self, mock_download, tmp_path):
        """ensure_models should call hf_hub_download for each model file."""
        # Make mock return a fake path
        fake_path = tmp_path / "fake_model.pth"
        fake_path.touch()
        mock_download.return_value = str(fake_path)

        mgr = ModelManager()
        det_cfg, det_ckpt, pose_cfg, pose_ckpt = mgr.ensure_models()

        # Should be called 4 times (2 detection + 2 pose)
        assert mock_download.call_count == 4

        # Check that repo_id is correct
        for call in mock_download.call_args_list:
            assert call.kwargs["repo_id"] == "fparodi/primateface-models"
            assert call.kwargs["library_name"] == "primateface"

    @patch("primateface._model_manager.hf_hub_download")
    def test_ensure_models_returns_4_paths(self, mock_download, tmp_path):
        fake_path = tmp_path / "fake.pth"
        fake_path.touch()
        mock_download.return_value = str(fake_path)

        mgr = ModelManager()
        result = mgr.ensure_models()
        assert len(result) == 4
        assert all(isinstance(p, Path) for p in result)

    @patch("primateface._model_manager.hf_hub_download")
    def test_skips_download_when_files_exist(self, mock_download, tmp_path):
        """If model_dir has all files, should skip downloads."""
        # Create all expected local files
        for name in [
            "mmdet_config.py", "mmdet_checkpoint.pth",
            "mmpose_config.py", "mmpose_checkpoint.pth",
        ]:
            (tmp_path / name).touch()

        mgr = ModelManager(model_dir=tmp_path)
        mgr.ensure_models()

        # Should not call hf_hub_download at all
        mock_download.assert_not_called()

    @patch("primateface._model_manager.hf_hub_download")
    def test_creates_model_dir_if_missing(self, mock_download, tmp_path):
        fake_path = tmp_path / "fake.pth"
        fake_path.touch()
        mock_download.return_value = str(fake_path)

        new_dir = tmp_path / "models" / "subdir"
        mgr = ModelManager(model_dir=new_dir)
        mgr.ensure_models()

        assert new_dir.exists()
