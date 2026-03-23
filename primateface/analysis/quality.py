"""Face quality assessment from image crops and landmarks.

Computes quality metrics for detected faces: blur, size, visibility,
brightness, and a composite score. Useful for filtering low-quality
detections before downstream analysis.
"""

from typing import Dict

import cv2
import numpy as np

from .utils import visibility_ratio


def face_quality(
    image: np.ndarray,
    bbox: np.ndarray,
    keypoints: np.ndarray,
) -> Dict[str, float]:
    """Compute face quality metrics.

    Args:
        image: Full image as HxWxC uint8 array (BGR or RGB).
        bbox: Bounding box as [x1, y1, x2, y2] or [x, y, w, h].
            If all values <= 1.0, assumed to be normalized coordinates.
        keypoints: Array of shape (68, 3) with [x, y, visibility].
            If shape is (68, 2), visibility is assumed to be 1.0 for all.

    Returns:
        Dict with keys:
          'blur': Laplacian variance (higher = sharper). Range ~0-2000+.
          'size': Face area as fraction of image area. Range [0, 1].
          'visibility': Fraction of visible keypoints. Range [0, 1].
          'brightness': Mean pixel intensity of face crop. Range [0, 255].
          'score': Composite quality score. Range [0, 1].
    """
    h_img, w_img = image.shape[:2]

    # Parse bbox
    bbox = np.asarray(bbox, dtype=np.float64)
    if len(bbox) == 4:
        x1, y1, x2_or_w, y2_or_h = bbox
        # Heuristic: if x2 > x1 and y2 > y1 and values are large, it's xyxy
        # Otherwise treat as xywh
        if x2_or_w > x1 and y2_or_h > y1 and max(x2_or_w, y2_or_h) > 1.0:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2_or_w), int(y2_or_h)
        else:
            x1, y1 = int(x1), int(y1)
            x2, y2 = int(x1 + x2_or_w), int(y1 + y2_or_h)
    else:
        return _empty_quality()

    # Clamp to image bounds
    x1 = max(0, min(x1, w_img - 1))
    y1 = max(0, min(y1, h_img - 1))
    x2 = max(x1 + 1, min(x2, w_img))
    y2 = max(y1 + 1, min(y2, h_img))

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return _empty_quality()

    # Blur: Laplacian variance
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Size: face area / image area
    face_area = (x2 - x1) * (y2 - y1)
    img_area = w_img * h_img
    size = face_area / img_area if img_area > 0 else 0.0

    # Visibility
    vis = visibility_ratio(keypoints)

    # Brightness: mean intensity
    brightness = float(gray.mean())

    # Composite score: weighted combination, each term mapped to [0, 1]
    blur_score = min(blur / 500.0, 1.0)          # 500+ Laplacian var = sharp
    size_score = min(size / 0.05, 1.0)            # 5%+ of image = good size
    brightness_score = 1.0 - abs(brightness - 127.5) / 127.5  # best at mid-gray
    score = 0.3 * blur_score + 0.3 * vis + 0.2 * size_score + 0.2 * brightness_score
    score = max(0.0, min(1.0, score))

    return {
        "blur": blur,
        "size": size,
        "visibility": vis,
        "brightness": brightness,
        "score": score,
    }


def _empty_quality() -> Dict[str, float]:
    """Return a zero-quality result for degenerate inputs."""
    return {
        "blur": 0.0,
        "size": 0.0,
        "visibility": 0.0,
        "brightness": 0.0,
        "score": 0.0,
    }
