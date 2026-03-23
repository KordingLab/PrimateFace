"""Face embedding backends for re-identification.

Provides wrapper functions for ArcFace and MegaDescriptor embedding
models. Each backend is lazily loaded and requires optional dependencies.
"""

from __future__ import annotations

from typing import Callable, Tuple

import cv2
import numpy as np


def _require_insightface():
    """Import insightface or raise a helpful error."""
    try:
        import insightface
        return insightface
    except ImportError:
        raise ImportError(
            "insightface is required for ArcFace embeddings. "
            "Install with: uv pip install 'primateface[embedding]'"
        )


def _require_timm():
    """Import timm or raise a helpful error."""
    try:
        import timm
        return timm
    except ImportError:
        raise ImportError(
            "timm is required for MegaDescriptor embeddings. "
            "Install with: uv pip install 'primateface[megadescriptor]'"
        )


def _parse_device_index(device: str) -> int:
    """Parse InsightFace ctx_id from a PyTorch device string.

    Args:
        device: PyTorch device string ("cpu", "cuda", "cuda:0", "cuda:1").

    Returns:
        InsightFace ctx_id: -1 for CPU, 0+ for GPU index.
    """
    device_lower = device.lower()
    if "cpu" in device_lower:
        return -1
    if "cuda" in device_lower:
        if ":" in device_lower:
            return int(device_lower.split(":")[1])
        return 0
    return -1


def load_arcface(device: str = "cpu") -> Tuple[object, Callable]:
    """Load ArcFace recognition model via InsightFace.

    Args:
        device: Device string (used to select ONNX provider).

    Returns:
        Tuple of (model_object, embedding_function).
        The embedding function takes a BGR ndarray and returns a 512-d vector.
    """
    _require_insightface()
    from insightface.app import FaceAnalysis

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "cpu" in device.lower():
        providers = ["CPUExecutionProvider"]

    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=_parse_device_index(device), det_size=(640, 640))
    rec_model = app.models["recognition"]

    def embed_fn(crop_bgr: np.ndarray) -> np.ndarray:
        """Compute 512-d ArcFace embedding from a face crop."""
        if crop_bgr.shape[:2] != (112, 112):
            crop_bgr = cv2.resize(crop_bgr, (112, 112))
        return rec_model.get_feat(crop_bgr).flatten()

    return rec_model, embed_fn


def load_megadescriptor(device: str = "cpu") -> Tuple[object, Callable]:
    """Load MegaDescriptor-L-384 wildlife re-ID model via timm.

    Args:
        device: PyTorch device string.

    Returns:
        Tuple of (model_object, embedding_function).
        The embedding function takes a BGR ndarray and returns a 1536-d vector.
    """
    timm = _require_timm()
    import torch
    from PIL import Image

    model = timm.create_model(
        "hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True
    )
    model.eval()
    if "cuda" in device:
        model = model.to(device)

    data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
    transform = timm.data.create_transform(**data_cfg)

    def embed_fn(crop_bgr: np.ndarray) -> np.ndarray:
        """Compute 1536-d MegaDescriptor embedding from a face crop."""
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        tensor = transform(pil_img).unsqueeze(0)
        if "cuda" in device:
            tensor = tensor.to(device)
        with torch.no_grad():
            emb = model(tensor)
        return emb.cpu().numpy().flatten()

    return model, embed_fn


# Registry of available embedding backends
EMBEDDING_BACKENDS = {
    "arcface": load_arcface,
    "megadescriptor": load_megadescriptor,
}
