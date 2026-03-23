"""Centralized registry of PrimateFace models hosted on Hugging Face Hub.

Single source of truth for model metadata, HF repo location, and file
mappings. All download utilities should import from here rather than
defining their own constants.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple


# ---------- HuggingFace Hub coordinates ----------

HF_REPO_ID: str = "fparodi/primateface-models"
HF_REPO_URL: str = f"https://huggingface.co/{HF_REPO_ID}"
LIBRARY_NAME: str = "primateface"
LIBRARY_VERSION: str = "0.1.0"


# ---------- Model entries ----------

@dataclass(frozen=True)
class ModelEntry:
    """Metadata for a single model file hosted on HF Hub.

    Attributes:
        local_name: Filename used locally (e.g. ``mmdet_checkpoint.pth``).
        hf_subfolder: Subdirectory in the HF repo (e.g. ``detection``).
        hf_filename: Filename within the HF subfolder.
        task: One of ``detection`` or ``pose``.
        file_type: One of ``config`` or ``checkpoint``.
        description: Human-readable one-liner.
        variant: Model variant name (``"default"`` for the standard model).
    """

    local_name: str
    hf_subfolder: str
    hf_filename: str
    task: str
    file_type: str
    description: str
    variant: str = "default"


MODEL_ENTRIES: List[ModelEntry] = [
    ModelEntry(
        local_name="mmdet_config.py",
        hf_subfolder="detection",
        hf_filename="cascade_rcnn_r101_fpn_config.py",
        task="detection",
        file_type="config",
        description="Cascade R-CNN R101-FPN detection config (MMDetection)",
    ),
    ModelEntry(
        local_name="mmdet_checkpoint.pth",
        hf_subfolder="detection",
        hf_filename="cascade_rcnn_r101_fpn.pth",
        task="detection",
        file_type="checkpoint",
        description="Cascade R-CNN R101-FPN detection weights (340 MB)",
    ),
    ModelEntry(
        local_name="mmpose_config.py",
        hf_subfolder="pose",
        hf_filename="hrnetv2_w18_dark_68kpt_config.py",
        task="pose",
        file_type="config",
        description="HRNetV2-W18-DARK 68-keypoint pose config (MMPose)",
    ),
    ModelEntry(
        local_name="mmpose_checkpoint.pth",
        hf_subfolder="pose",
        hf_filename="hrnetv2_w18_dark_68kpt.pth",
        task="pose",
        file_type="checkpoint",
        description="HRNetV2-W18-DARK 68-keypoint pose weights (38 MB)",
    ),
    # -- Alternative pose model: ViTPose --
    ModelEntry(
        local_name="vitpose_config.py",
        hf_subfolder="pose",
        hf_filename="vitpose_base_68kpt_config.py",
        task="pose",
        file_type="config",
        description="ViTPose-Base 68-keypoint pose config (MMPose)",
        variant="vitpose",
    ),
    ModelEntry(
        local_name="vitpose_checkpoint.pth",
        hf_subfolder="pose",
        hf_filename="vitpose_base_68kpt.pth",
        task="pose",
        file_type="checkpoint",
        description="ViTPose-Base 68-keypoint pose weights (1.2 GB)",
        variant="vitpose",
    ),
]


# ---------- Derived convenience mappings ----------

# {local_name: (hf_subfolder, hf_filename)} — used by download functions
MODELS: Dict[str, Tuple[str, str]] = {
    e.local_name: (e.hf_subfolder, e.hf_filename) for e in MODEL_ENTRIES
}

LOCAL_FILENAMES: List[str] = [e.local_name for e in MODEL_ENTRIES]

DET_FILES: List[str] = [e.local_name for e in MODEL_ENTRIES if e.task == "detection"]
POSE_FILES: List[str] = [e.local_name for e in MODEL_ENTRIES if e.task == "pose"]


def get_model_entry(
    task: str, file_type: str, variant: str = "default"
) -> ModelEntry:
    """Look up a single model entry by task, file type, and variant.

    Args:
        task: ``"detection"`` or ``"pose"``.
        file_type: ``"config"`` or ``"checkpoint"``.
        variant: Model variant (``"default"`` or e.g. ``"vitpose"``).

    Returns:
        The matching ModelEntry.

    Raises:
        KeyError: If no entry matches.
    """
    for entry in MODEL_ENTRIES:
        if (
            entry.task == task
            and entry.file_type == file_type
            and entry.variant == variant
        ):
            return entry
    raise KeyError(
        f"No model entry for task={task!r}, file_type={file_type!r}, "
        f"variant={variant!r}"
    )


AVAILABLE_VARIANTS: Dict[str, List[str]] = {}
for _e in MODEL_ENTRIES:
    AVAILABLE_VARIANTS.setdefault(_e.task, [])
    if _e.variant not in AVAILABLE_VARIANTS[_e.task]:
        AVAILABLE_VARIANTS[_e.task].append(_e.variant)
