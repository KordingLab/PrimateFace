"""High-level PrimateFace interface.

Provides the ``PrimateFace`` class — the "3 lines of code" entry point
for primate face detection, landmark estimation, and analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Sequence, Union

import cv2
import numpy as np

from .face import Face
from ._model_manager import ModelManager, POSE_MODEL_VARIANTS
from ._embedding import EMBEDDING_BACKENDS

InputType = Union[str, Path, np.ndarray]


class PrimateFace:
    """High-level interface for primate face analysis.

    Handles model download, initialization, and provides a simple
    :meth:`analyze` method that returns rich :class:`Face` objects.

    Example:
        >>> import primateface
        >>> pf = primateface.PrimateFace()
        >>> faces = pf.analyze("monkey.jpg")
        >>> faces[0].head_pose
        (5.2, -3.1, 1.0)

    Args:
        device: PyTorch device string. Defaults to ``"cuda:0"`` if
            available, else ``"cpu"``.
        pose_model: Pose estimation backend. ``"hrnet"`` (default,
            38 MB) or ``"vitpose"`` (1.2 GB, more accurate).
        embedding_model: Face embedding backend for re-identification.
            ``None`` (default, no embedding), ``"arcface"`` (512-d,
            requires insightface), or ``"megadescriptor"`` (1536-d,
            requires timm).
        det_threshold: Detection confidence threshold.
        nms_threshold: NMS threshold for overlapping detections.
        model_dir: Directory to cache model files. If *None*, uses
            HuggingFace Hub default cache (``~/.cache/huggingface/``).
    """

    def __init__(
        self,
        device: Optional[str] = None,
        pose_model: str = "hrnet",
        embedding_model: Optional[str] = None,
        det_threshold: float = 0.5,
        nms_threshold: float = 0.3,
        model_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        import torch

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.pose_model = pose_model
        self.embedding_model = embedding_model
        self.det_threshold = det_threshold
        self.nms_threshold = nms_threshold

        # Validate pose_model
        if pose_model not in POSE_MODEL_VARIANTS:
            raise ValueError(
                f"Unknown pose_model={pose_model!r}. "
                f"Choose from: {list(POSE_MODEL_VARIANTS.keys())}"
            )

        # Validate and load embedding model
        self._embedding_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None
        if embedding_model is not None:
            if embedding_model not in EMBEDDING_BACKENDS:
                raise ValueError(
                    f"Unknown embedding_model={embedding_model!r}. "
                    f"Choose from: {list(EMBEDDING_BACKENDS.keys())}"
                )
            _, self._embedding_fn = EMBEDDING_BACKENDS[embedding_model](device)

        # Resolve model paths (downloads from HF if needed)
        pose_variant = POSE_MODEL_VARIANTS[pose_model]
        mgr = ModelManager(model_dir=model_dir, pose_variant=pose_variant)
        det_cfg, det_ckpt, pose_cfg, pose_ckpt = mgr.ensure_models()

        # Initialize the underlying processor
        from ._processor import PrimateFaceProcessor

        self._processor = PrimateFaceProcessor(
            det_config=str(det_cfg),
            det_checkpoint=str(det_ckpt),
            pose_config=str(pose_cfg),
            pose_checkpoint=str(pose_ckpt),
            device=self.device,
        )

    def analyze(
        self,
        image: InputType,
        det_threshold: Optional[float] = None,
        nms_threshold: Optional[float] = None,
    ) -> List[Face]:
        """Detect and analyze all primate faces in an image.

        Args:
            image: Input image as a file path (``str``/``Path``),
                numpy array (BGR), or PIL Image.
            det_threshold: Override default detection threshold.
            nms_threshold: Override default NMS threshold.

        Returns:
            List of :class:`Face` objects, sorted by detection
            confidence (highest first). Empty list if no faces detected.

        Raises:
            FileNotFoundError: If image path does not exist.
            ValueError: If image cannot be loaded or decoded.
        """
        bgr = self._load_image(image)
        h, w = bgr.shape[:2]
        image_size = (w, h)

        det_thr = det_threshold if det_threshold is not None else self.det_threshold
        nms_thr = nms_threshold if nms_threshold is not None else self.nms_threshold

        # Run detection
        bboxes, scores = self._processor.detect_primates(
            bgr, bbox_thr=det_thr, nms_thr=nms_thr
        )

        if len(bboxes) == 0:
            return []

        # Run pose estimation
        pose_results = self._processor.estimate_poses(bgr, bboxes)
        keypoints_xy = pose_results.keypoints  # (N, 68, 2)
        keypoint_scores = pose_results.keypoint_scores  # (N, 68)

        # Build Face objects
        faces: List[Face] = []
        for i in range(len(bboxes)):
            kpts_with_vis = np.column_stack([
                keypoints_xy[i],
                keypoint_scores[i][:, np.newaxis],
            ]).astype(np.float32)

            face = Face(
                bbox=bboxes[i].astype(np.float32),
                score=float(scores[i]),
                keypoints=kpts_with_vis,
                _image=bgr,
                _image_size=image_size,
                _embedding_fn=self._embedding_fn,
                _embedding_backend=self.embedding_model,
            )
            faces.append(face)

        faces.sort(key=lambda f: f.score, reverse=True)
        return faces

    def analyze_batch(
        self,
        images: Sequence[InputType],
    ) -> List[List[Face]]:
        """Analyze multiple images.

        Convenience wrapper that calls :meth:`analyze` for each image.

        Args:
            images: Sequence of image inputs.

        Returns:
            List of Face lists, one per input image.
        """
        return [self.analyze(img) for img in images]

    @staticmethod
    def draw(
        faces: List[Face],
        image: InputType,
        output: Optional[Union[str, Path]] = None,
        draw_keypoints: bool = True,
        draw_skeleton: bool = True,
        draw_bbox: bool = True,
        show_pose: bool = False,
    ) -> np.ndarray:
        """Draw detected faces on an image.

        Args:
            faces: List of Face objects from :meth:`analyze`.
            image: Source image (path, numpy BGR, or PIL Image).
            output: If provided, save the visualization to this path.
            draw_keypoints: Draw landmark points.
            draw_skeleton: Draw skeleton connections.
            draw_bbox: Draw bounding boxes.
            show_pose: Draw 3D head pose wireframe cube + axes.
                When True, keypoints/skeleton/bbox are disabled by
                default for a cleaner view.

        Returns:
            BGR numpy array with faces drawn.
        """
        # When show_pose is on, turn off other overlays unless explicitly set
        if show_pose:
            draw_keypoints = False
            draw_skeleton = False
            draw_bbox = False

        bgr = PrimateFace._load_image(image)
        canvas = bgr.copy()

        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            kpts = face.keypoints[:, :2]  # (68, 2)
            scores = face.keypoints[:, 2]  # (68,)

            if draw_bbox:
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{face.score:.2f}"
                cv2.putText(
                    canvas, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                )

            if draw_keypoints:
                for j in range(68):
                    if scores[j] > 0.3:
                        px, py = int(kpts[j, 0]), int(kpts[j, 1])
                        cv2.circle(canvas, (px, py), 2, (0, 0, 255), -1)

            if draw_skeleton:
                # Jaw contour
                for j in range(16):
                    if scores[j] > 0.3 and scores[j + 1] > 0.3:
                        p1 = (int(kpts[j, 0]), int(kpts[j, 1]))
                        p2 = (int(kpts[j + 1, 0]), int(kpts[j + 1, 1]))
                        cv2.line(canvas, p1, p2, (255, 200, 0), 1)
                # Eye, brow, nose, mouth contours
                for start, end in [
                    (17, 22), (22, 27),  # eyebrows
                    (27, 31),            # nose bridge
                    (31, 36),            # nose base
                    (36, 42), (42, 48),  # eyes
                    (48, 60), (60, 68),  # mouth
                ]:
                    for j in range(start, end - 1):
                        if scores[j] > 0.3 and scores[j + 1] > 0.3:
                            p1 = (int(kpts[j, 0]), int(kpts[j, 1]))
                            p2 = (int(kpts[j + 1, 0]), int(kpts[j + 1, 1]))
                            cv2.line(canvas, p1, p2, (255, 200, 0), 1)

            if show_pose:
                _draw_pose_axes(canvas, face.keypoints, face._image_size, face.bbox)

        if output is not None:
            cv2.imwrite(str(output), canvas)

        return canvas

    @staticmethod
    def _load_image(image: InputType) -> np.ndarray:
        """Convert various input types to BGR numpy array.

        Args:
            image: File path, numpy array, or PIL Image.

        Returns:
            BGR numpy array.

        Raises:
            FileNotFoundError: If path does not exist.
            ValueError: If image cannot be decoded.
            TypeError: If input type is not supported.
        """
        if isinstance(image, (str, Path)):
            path = Path(image)
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")
            bgr = cv2.imread(str(path))
            if bgr is None:
                raise ValueError(f"Could not decode image: {path}")
            return bgr

        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            if image.ndim == 3 and image.shape[2] == 4:
                return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            if image.ndim == 3 and image.shape[2] == 3:
                return image
            raise ValueError(
                f"Unexpected array shape: {image.shape}. "
                "Expected (H, W), (H, W, 3), or (H, W, 4)."
            )

        # PIL Image
        try:
            import PIL.Image

            if isinstance(image, PIL.Image.Image):
                rgb = np.array(image.convert("RGB"))
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except ImportError:
            pass

        raise TypeError(
            f"Unsupported image type: {type(image)}. "
            "Expected str, Path, numpy array, or PIL Image."
        )


def _draw_pose_axes(
    canvas: np.ndarray,
    keypoints: np.ndarray,
    image_size: tuple,
    bbox: Optional[np.ndarray] = None,
) -> None:
    """Draw 3D head pose wireframe cube on the canvas.

    Projects a 3D bounding box around the head using the estimated
    rotation from cv2.solvePnP, plus RGB axis arrows from the origin.
    The cube is scaled to match the face bounding box.

    Args:
        canvas: BGR image to draw on (modified in place).
        keypoints: Shape (68, 3) with [x, y, score].
        image_size: (width, height) of the image.
        bbox: Optional [x1, y1, x2, y2] to scale the cube.
    """
    from primateface.analysis.constants import POSE_LANDMARK_INDICES, POSE_REFERENCE_3D

    coords = keypoints[:, :2].astype(np.float64)
    image_points = coords[POSE_LANDMARK_INDICES]

    w, h = image_size
    focal_length = float(w)
    cx, cy = w / 2.0, h / 2.0
    camera_matrix = np.array(
        [[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(
        POSE_REFERENCE_3D, image_points,
        camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return

    # Scale cube to match bbox size
    if bbox is not None:
        bw = float(bbox[2] - bbox[0])
        bh = float(bbox[3] - bbox[1])
        s = max(bw, bh) / 2.0 * 0.7  # slightly smaller than bbox
    else:
        s = 40.0

    # 3D wireframe cube vertices (8 corners)
    cube_3d = np.float64([
        [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],  # back face
        [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s],       # front face
    ])

    # Project to 2D
    cube_2d, _ = cv2.projectPoints(cube_3d, rvec, tvec, camera_matrix, dist_coeffs)
    pts = cube_2d.reshape(-1, 2).astype(int)

    # Draw back face (darker)
    for i, j in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        cv2.line(canvas, tuple(pts[i]), tuple(pts[j]), (100, 200, 100), 1)
    # Draw front face (brighter)
    for i, j in [(4, 5), (5, 6), (6, 7), (7, 4)]:
        cv2.line(canvas, tuple(pts[i]), tuple(pts[j]), (0, 255, 0), 2)
    # Draw connecting edges
    for i, j in [(0, 4), (1, 5), (2, 6), (3, 7)]:
        cv2.line(canvas, tuple(pts[i]), tuple(pts[j]), (0, 255, 0), 1)

    # Also draw RGB axis arrows from origin
    axis_len = s * 1.5
    axis_3d = np.float64([
        [axis_len, 0, 0],
        [0, axis_len, 0],
        [0, 0, axis_len],
    ])
    origin_3d = np.float64([[0, 0, 0]])

    axis_2d, _ = cv2.projectPoints(axis_3d, rvec, tvec, camera_matrix, dist_coeffs)
    origin_2d, _ = cv2.projectPoints(origin_3d, rvec, tvec, camera_matrix, dist_coeffs)

    origin = tuple(origin_2d[0].ravel().astype(int))
    for i, color in enumerate([(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
        endpoint = tuple(axis_2d[i].ravel().astype(int))
        cv2.arrowedLine(canvas, origin, endpoint, color, 2, tipLength=0.2)
