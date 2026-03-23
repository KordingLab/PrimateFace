"""Face dataclass with lazy analysis properties.

Each ``Face`` is produced by :meth:`PrimateFace.analyze` and carries
detection outputs (bbox, score, keypoints) plus derived features that
are computed on first access and cached thereafter.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np


@dataclass
class Face:
    """A single detected primate face with landmarks and derived features.

    Core detection fields are set at construction time. Analysis features
    (head_pose, symmetry, quality, kinematics) are computed lazily on first
    access via ``cached_property``.

    Attributes:
        bbox: Bounding box as ``[x1, y1, x2, y2]`` in pixel coordinates.
        score: Detection confidence in ``[0, 1]``.
        keypoints: Facial landmarks, shape ``(68, 3)`` with
            ``[x, y, confidence]``.
    """

    bbox: np.ndarray
    score: float
    keypoints: np.ndarray
    _image: np.ndarray = field(repr=False, compare=False)
    _image_size: Tuple[int, int] = field(repr=False, compare=False)
    _embedding_fn: Optional[Callable[[np.ndarray], np.ndarray]] = field(
        repr=False, compare=False, default=None
    )
    _embedding_backend: Optional[str] = field(
        repr=False, compare=False, default=None
    )

    # -- Eager convenience properties --

    @functools.cached_property
    def crop(self) -> np.ndarray:
        """Face crop extracted from the source image using the bbox."""
        x1, y1, x2, y2 = self.bbox.astype(int)
        h, w = self._image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        return self._image[y1:y2, x1:x2].copy()

    # -- Lazy analysis properties --

    @functools.cached_property
    def head_pose(self) -> Tuple[float, float, float]:
        """Head pose as ``(yaw, pitch, roll)`` in degrees."""
        from .analysis.head_pose import estimate_head_pose

        return estimate_head_pose(self.keypoints, self._image_size)

    @functools.cached_property
    def quality(self) -> Dict[str, float]:
        """Face quality metrics: blur, size, visibility, brightness, score."""
        from .analysis.quality import face_quality

        return face_quality(self._image, self.bbox, self.keypoints)

    @functools.cached_property
    def symmetry(self) -> float:
        """Fluctuating asymmetry score (0 = perfect symmetry)."""
        from .analysis.symmetry import facial_symmetry

        return facial_symmetry(self.keypoints)

    @functools.cached_property
    def region_symmetry(self) -> Dict[str, float]:
        """Per-region symmetry: jaw, eyebrows, eyes, nose, mouth."""
        from .analysis.symmetry import per_region_symmetry

        return per_region_symmetry(self.keypoints)

    @functools.cached_property
    def kinematics(self) -> Dict[str, float]:
        """All scalar kinematic/geometric features (~14 keys)."""
        from .analysis.kinematics import extract_kinematics

        return extract_kinematics(self.keypoints)

    # -- Shortcut properties that delegate to kinematics --

    @functools.cached_property
    def mouth_aperture(self) -> float:
        """Mouth opening (normalized by interocular distance)."""
        return self.kinematics["mouth_aperture"]

    @functools.cached_property
    def eye_aperture(self) -> Tuple[float, float]:
        """Eye opening as ``(right, left)``, normalized."""
        return (
            self.kinematics["right_eye_aperture"],
            self.kinematics["left_eye_aperture"],
        )

    @functools.cached_property
    def brow_position(self) -> Tuple[float, float]:
        """Brow height as ``(right, left)``, normalized."""
        return (
            self.kinematics["right_brow_height"],
            self.kinematics["left_brow_height"],
        )

    @functools.cached_property
    def interocular_distance(self) -> float:
        """Distance between eye centers in pixels."""
        return self.kinematics["interocular_distance"]

    # -- Embedding / re-identification --

    def _get_aligned_crop(self) -> Optional[np.ndarray]:
        """Align face to canonical 112x112 for ArcFace embedding.

        Returns:
            Aligned 112x112 BGR image, or None if alignment fails.
        """
        from ._processor import align_face
        from ._constants import TARGET_LANDMARKS_5PT_112X112

        landmarks_68 = self.keypoints[:, :2]
        if landmarks_68.shape[0] != 68 or np.any(np.isnan(landmarks_68)):
            return None
        aligned, _ = align_face(
            self._image,
            landmarks_68,
            output_size=112,
            target_landmarks=TARGET_LANDMARKS_5PT_112X112,
        )
        return aligned

    @functools.cached_property
    def embedding(self) -> np.ndarray:
        """Face embedding vector for re-identification.

        For ArcFace, the face is aligned to a canonical 112x112 pose before
        embedding extraction. Falls back to the raw crop if alignment fails.

        Returns:
            1-D numpy array (512-d for ArcFace, 1536-d for MegaDescriptor).

        Raises:
            RuntimeError: If no embedding model was configured.
        """
        if self._embedding_fn is None:
            raise RuntimeError(
                "No embedding model loaded. Initialize with: "
                "PrimateFace(embedding_model='arcface')"
            )
        if self._embedding_backend == "arcface":
            aligned = self._get_aligned_crop()
            if aligned is not None:
                return self._embedding_fn(aligned)
        return self._embedding_fn(self.crop)

    def verify(
        self, other: "Face", threshold: float = 0.4
    ) -> Tuple[bool, float]:
        """Compare this face to another for identity verification.

        Uses cosine distance between embeddings.

        Args:
            other: Another Face object to compare against.
            threshold: Maximum cosine distance to consider same identity.
                Default 0.4 works well for ArcFace.

        Returns:
            Tuple of ``(is_same_person, cosine_distance)``.
        """
        e1 = self.embedding
        e2 = other.embedding
        cos_sim = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8))
        distance = 1.0 - cos_sim
        return distance < threshold, distance

    # -- Serialization --

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict (excludes image reference)."""
        return {
            "bbox": self.bbox.tolist(),
            "score": self.score,
            "keypoints": self.keypoints.tolist(),
        }

    def __repr__(self) -> str:
        x1, y1, x2, y2 = self.bbox.astype(int)
        return (
            f"Face(score={self.score:.2f}, "
            f"bbox=[{x1}, {y1}, {x2}, {y2}], "
            f"landmarks=68)"
        )
