"""Model download and path resolution for PrimateFace.

Wraps the ``demos.model_registry`` metadata and ``huggingface_hub``
download logic into a single ``ensure_models()`` call.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional, Tuple, Union

from ._model_registry import (
    HF_REPO_ID,
    LIBRARY_NAME,
    LIBRARY_VERSION,
    get_model_entry,
)
from huggingface_hub import hf_hub_download

# Mapping from user-facing pose model names to registry variants
POSE_MODEL_VARIANTS = {
    "hrnet": "default",
    "vitpose": "vitpose",
}


class ModelManager:
    """Handles model file resolution and automatic downloading.

    On construction, does NOT download anything. Call
    :meth:`ensure_models` to check availability and download if needed.

    Args:
        model_dir: Optional custom directory for model files. If *None*,
            uses the HuggingFace Hub cache (``~/.cache/huggingface/``).
        pose_variant: Registry variant for the pose model
            (``"default"`` for HRNet, ``"vitpose"`` for ViTPose).
    """

    def __init__(
        self,
        model_dir: Optional[Union[str, Path]] = None,
        pose_variant: str = "default",
    ) -> None:
        self._model_dir = Path(model_dir) if model_dir else None
        self._pose_variant = pose_variant

    def ensure_models(self) -> Tuple[Path, Path, Path, Path]:
        """Ensure all required model files are available locally.

        Downloads from HuggingFace Hub if not already cached.

        Returns:
            Tuple of ``(det_config, det_checkpoint, pose_config,
            pose_checkpoint)`` as absolute Paths.
        """
        required = [
            ("detection", "config", "default"),
            ("detection", "checkpoint", "default"),
            ("pose", "config", self._pose_variant),
            ("pose", "checkpoint", self._pose_variant),
        ]

        paths = {}
        for task, file_type, variant in required:
            entry = get_model_entry(task, file_type, variant)

            # Check custom model_dir first
            if self._model_dir is not None:
                local_path = self._model_dir / entry.local_name
                if local_path.exists():
                    paths[(task, file_type)] = local_path
                    continue

            # Download via HF Hub
            cached_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=entry.hf_filename,
                subfolder=entry.hf_subfolder,
                library_name=LIBRARY_NAME,
                library_version=LIBRARY_VERSION,
            )

            if self._model_dir is not None:
                self._model_dir.mkdir(parents=True, exist_ok=True)
                local_path = self._model_dir / entry.local_name
                shutil.copy2(cached_path, str(local_path))
                paths[(task, file_type)] = local_path
            else:
                paths[(task, file_type)] = Path(cached_path)

        return (
            paths[("detection", "config")],
            paths[("detection", "checkpoint")],
            paths[("pose", "config")],
            paths[("pose", "checkpoint")],
        )
