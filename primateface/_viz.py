"""
Fast visualization utilities for pose estimation results.

This module provides optimized OpenCV-based visualization classes for
drawing pose estimation results including keypoints, skeletons, and
bounding boxes with consistent color coding for tracked instances.

Example:
    visualizer = FastPoseVisualizer(draw_keypoints=True, draw_skeleton=True)
    frame_with_poses = visualizer.draw_poses(frame, pred_instances)
"""

from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

from ._constants import (
    BBOX_THICKNESS,
    DEFAULT_COLOR_BGR,
    ID_COLORS_BGR,
    KEYPOINT_RADIUS,
    LINE_THICKNESS,
    TEXT_COLOR_BGR,
)


class FastPoseVisualizer:
    """Fast visualization for pose estimation results using optimized OpenCV operations.
    
    This class provides efficient rendering of pose estimation results with
    configurable drawing options and consistent color coding for tracked instances.
    
    Attributes:
        draw_keypoints: Whether to draw keypoint circles
        draw_skeleton: Whether to draw skeleton connections
        draw_bbox: Whether to draw bounding boxes
        keypoint_radius: Radius for keypoint circles
        line_thickness: Thickness for skeleton lines
        bbox_thickness: Thickness for bounding box lines
        show_bbox_conf: Whether to show confidence scores on bboxes
        keypoint_threshold: Confidence threshold for displaying keypoints
    """

    def __init__(
        self,
        draw_keypoints: bool = True,
        draw_skeleton: bool = True,
        draw_bbox: bool = True,
        keypoint_radius: int = 3,
        line_thickness: int = 1,
        bbox_thickness: int = 2,
        show_bbox_conf: bool = True,
        keypoint_threshold: float = 0.3,
    ) -> None:
        """Initialize the visualizer with drawing parameters.
        
        Args:
            draw_keypoints: Whether to draw keypoint circles
            draw_skeleton: Whether to draw skeleton line connections
            draw_bbox: Whether to draw bounding boxes around detections
            keypoint_radius: Radius in pixels for keypoint circles
            line_thickness: Thickness in pixels for skeleton lines
            bbox_thickness: Thickness in pixels for bounding box lines
            show_bbox_conf: Whether to display confidence scores on bboxes
            keypoint_threshold: Minimum confidence to display a keypoint
        """
        self.draw_keypoints = draw_keypoints
        self.draw_skeleton = draw_skeleton
        self.draw_bbox = draw_bbox
        self.keypoint_radius = keypoint_radius
        self.line_thickness = line_thickness
        self.bbox_thickness = bbox_thickness
        self.show_bbox_conf = show_bbox_conf
        self.keypoint_threshold = keypoint_threshold

        self.ID_COLORS = ID_COLORS_BGR
        self.DEFAULT_COLOR = DEFAULT_COLOR_BGR
        self.TEXT_COLOR = TEXT_COLOR_BGR

    def draw_poses(
        self,
        frame: np.ndarray,
        pred_instances: Any,
        skeleton_links: Optional[List[Tuple[int, int]]] = None,
        instance_ids: Optional[List[int]] = None,
        bbox_scores: Optional[List[float]] = None,
    ) -> np.ndarray:
        """Draw detected poses on frame efficiently.
        
        Args:
            frame: Input image/frame as numpy array (BGR format).
            pred_instances: Object containing keypoints and scores attributes.
            skeleton_links: List of (start_idx, end_idx) tuples defining skeleton.
            instance_ids: List of instance IDs for consistent coloring.
            bbox_scores: List of bounding box confidence scores.
            
        Returns:
            Frame with poses drawn on it (copy of input).
        """
        if pred_instances is None:
            return frame

        frame = frame.copy()

        keypoints = pred_instances.keypoints
        keypoint_scores = pred_instances.keypoint_scores

        for i in range(len(keypoints)):
            stable_id = instance_ids[i] if instance_ids is not None and i < len(instance_ids) else -1
            color = self.ID_COLORS.get(stable_id, self.DEFAULT_COLOR)

            kpts = keypoints[i]
            scores = keypoint_scores[i]

            if self.draw_skeleton and skeleton_links:
                for start_idx, end_idx in skeleton_links:
                    if (
                        scores[start_idx] > self.keypoint_threshold
                        and scores[end_idx] > self.keypoint_threshold
                    ):
                        start_pt = tuple(map(int, kpts[start_idx]))
                        end_pt = tuple(map(int, kpts[end_idx]))
                        cv2.line(frame, start_pt, end_pt, color, self.line_thickness)

            if self.draw_keypoints:
                for kpt, score in zip(kpts, scores):
                    if score > self.keypoint_threshold:
                        center = tuple(map(int, kpt))
                        cv2.circle(frame, center, self.keypoint_radius, color, -1)

            if self.draw_bbox and hasattr(pred_instances, "bboxes"):
                bbox = pred_instances.bboxes[i]
                pt1 = tuple(map(int, bbox[:2]))
                pt2 = tuple(map(int, bbox[2:]))
                cv2.rectangle(frame, pt1, pt2, color, self.bbox_thickness)

                id_text = f"ID: {stable_id}"
                cv2.putText(
                    frame,
                    id_text,
                    (pt1[0], pt1[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    self.TEXT_COLOR,
                    1,
                )

                if self.show_bbox_conf and bbox_scores and i < len(bbox_scores):
                    conf_score = bbox_scores[i]
                    cv2.putText(
                        frame,
                        f"{conf_score:.2f}",
                        (pt1[0], pt2[1] + 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1,
                    )

        return frame