"""Unified processing script for PrimateFace detection and pose estimation.

This module provides a unified interface for processing both videos and image
directories with primate detection and pose estimation models. It supports
MMDetection/MMPose frameworks and includes visualization and smoothing options.

Example:
    # Process a single image
    python process.py --input image.jpg --input-type image \\
        --det-config config.py --det-checkpoint model.pth \\
        --output-dir results/ --save-viz

    # Process a video
    python process.py --input video.mp4 --input-type video \\
        --det-config config.py --det-checkpoint model.pth \\
        --pose-config pose_config.py --pose-checkpoint pose_model.pth \\
        --output-dir results/ --save-viz --viz-pose
    
    # Process image directory
    python process.py --input ./images/ --input-type images \\
        --det-config config.py --det-checkpoint model.pth \\
        --pose-config pose_config.py --pose-checkpoint pose_model.pth \\
        --output-dir results/ --save-predictions
"""

import argparse
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import json_tricks
import mmcv
import natsort
import numpy as np
import torch
from mmdet.apis import inference_detector, init_detector
from mmengine.structures import InstanceData
from mmpose.apis import inference_topdown, init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.structures import merge_data_samples
from mmpose.utils import adapt_mmdet_pipeline
from tqdm import tqdm

from ._constants import (
    ALIGNED_FACE_SIZE,
    DEFAULT_BBOX_THR,
    DEFAULT_KPT_THR,
    DEFAULT_NMS_THR,
    DEFAULT_MEDIAN_WINDOW,
    DEFAULT_SAVGOL_ORDER,
    DEFAULT_SAVGOL_WINDOW,
    DET_CAT_ID,
    IMAGE_EXTENSIONS,
    LANDMARK_5PT_FROM_68_INDICES,
    TARGET_LANDMARKS_5PT_256X256,
    VIDEO_EXTENSIONS,
)
from ._smooth import MedianSavgolSmoother
from ._viz import FastPoseVisualizer


def get_5_source_landmarks_from_68(landmarks_68: np.ndarray) -> np.ndarray:
    """Extract 5 alignment landmarks from 68 points."""
    if landmarks_68.shape[0] != 68:
        return np.full((5, 2), np.nan, dtype=np.float32)
    idx = LANDMARK_5PT_FROM_68_INDICES
    pts = np.zeros((5, 2), dtype=np.float32)
    pts[0] = np.mean(landmarks_68[idx["left_eye_indices"], :2], axis=0)
    pts[1] = np.mean(landmarks_68[idx["right_eye_indices"], :2], axis=0)
    pts[2] = landmarks_68[idx["nose_tip_idx"], :2]
    pts[3] = landmarks_68[idx["left_mouth_corner_idx"], :2]
    pts[4] = landmarks_68[idx["right_mouth_corner_idx"], :2]
    return pts


def transform_keypoints(keypoints: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
    """Apply a 2x3 affine transform to 2-D keypoints."""
    if keypoints is None or transform_matrix is None:
        return np.full_like(keypoints, np.nan) if keypoints is not None else None
    homogeneous = np.hstack([keypoints[:, :2], np.ones((keypoints.shape[0], 1))])
    return (homogeneous @ transform_matrix.T).astype(np.float32)


def align_face(
    image_bgr: np.ndarray,
    landmarks_68: np.ndarray,
    output_size: int = ALIGNED_FACE_SIZE,
    target_landmarks: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Warp a face to canonical aligned space using 5-point alignment.

    Args:
        image_bgr: Input image in BGR format.
        landmarks_68: Array of shape (68, 2) or (68, 3).
        output_size: Size of the square output image (default 256).
        target_landmarks: Target 5-point landmarks for alignment.
            Defaults to TARGET_LANDMARKS_5PT_256X256. Use
            TARGET_LANDMARKS_5PT_112X112 for ArcFace-compatible output.

    Returns:
        Tuple of (aligned_image, affine_matrix) or (None, None) on failure.
    """
    if target_landmarks is None:
        target_landmarks = TARGET_LANDMARKS_5PT_256X256
    src_5pts = get_5_source_landmarks_from_68(landmarks_68)
    if np.any(np.isnan(src_5pts)):
        return None, None
    matrix, _ = cv2.estimateAffinePartial2D(
        src_5pts, target_landmarks, method=cv2.LMEDS
    )
    if matrix is None:
        matrix, _ = cv2.estimateAffine2D(src_5pts, target_landmarks)
    if matrix is None:
        return None, None
    aligned = cv2.warpAffine(image_bgr, matrix, (output_size, output_size), borderValue=(0, 0, 0))
    return aligned, matrix


class PrimateFaceProcessor:
    """Main processor for primate detection and pose estimation.
    
    This class handles both video and image processing with configurable
    detection and pose estimation models from MMDetection/MMPose.
    
    Attributes:
        det_model: Loaded detection model
        pose_model: Loaded pose estimation model
        device: Device for inference (cuda/cpu)
        visualizer: FastPoseVisualizer instance for drawing results
        smoother: Optional MedianSavgolSmoother for temporal smoothing
    """
    
    def __init__(
        self,
        det_config: str,
        det_checkpoint: str,
        pose_config: Optional[str] = None,
        pose_checkpoint: Optional[str] = None,
        device: str = "cuda:0",
        use_smoothing: bool = False,
        median_window: int = DEFAULT_MEDIAN_WINDOW,
        savgol_window: int = DEFAULT_SAVGOL_WINDOW,
        savgol_order: int = DEFAULT_SAVGOL_ORDER,
    ):
        """Initialize the processor with models and configurations.
        
        Args:
            det_config: Path to detection model config file.
            det_checkpoint: Path to detection model checkpoint.
            pose_config: Path to pose model config file (optional).
            pose_checkpoint: Path to pose model checkpoint (optional).
            device: Device to use for inference.
            use_smoothing: Whether to apply temporal smoothing (for videos).
            median_window: Window size for median filter.
            savgol_window: Window size for Savitzky-Golay filter.
            savgol_order: Polynomial order for Savitzky-Golay filter.
        """
        self.device = device
        
        # Initialize detection model
        self.det_model = init_detector(
            det_config,
            det_checkpoint,
            device=device,
            cfg_options=dict(model=dict(test_cfg=dict(score_thr=0.01)))
        )
        self.det_model.cfg = adapt_mmdet_pipeline(self.det_model.cfg)
        
        # Initialize pose model if provided
        self.pose_model = None
        if pose_config and pose_checkpoint:
            self.pose_model = init_pose_estimator(
                pose_config,
                pose_checkpoint,
                device=device,
                cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=False)))
            )
            print("Pose model loaded. Running detection + pose estimation.")
        else:
            print("No pose model provided. Running face detection only.")
        
        # Initialize visualizer
        self.visualizer = FastPoseVisualizer(
            draw_keypoints=True,
            draw_skeleton=True,
            draw_bbox=True
        )
        
        # Initialize smoother if needed
        self.smoother = None
        if use_smoothing:
            self.smoother = MedianSavgolSmoother(
                median_window=median_window,
                savgol_window=savgol_window,
                savgol_order=savgol_order
            )
    
    def detect_primates(
        self,
        image: np.ndarray,
        bbox_thr: float = DEFAULT_BBOX_THR,
        nms_thr: float = DEFAULT_NMS_THR
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Detect primates in an image.
        
        Args:
            image: Input image as numpy array (BGR format).
            bbox_thr: Detection confidence threshold.
            nms_thr: NMS threshold for overlapping detections.
            
        Returns:
            Tuple of (bboxes, scores) where bboxes is array of [x1, y1, x2, y2]
            and scores is array of confidence values.
        """
        det_result = inference_detector(self.det_model, image)
        pred_instances = det_result.pred_instances.cpu().numpy()
        
        # Filter by detection category
        bboxes = pred_instances.bboxes
        labels = pred_instances.labels
        scores = pred_instances.scores
        
        # Filter by category and threshold
        keep_idxs = np.logical_and(
            labels == DET_CAT_ID,
            scores > bbox_thr
        )
        
        bboxes = bboxes[keep_idxs]
        scores = scores[keep_idxs]
        
        # Apply NMS
        if len(bboxes) > 0:
            bboxes_xyxy = np.column_stack([
                bboxes[:, 0], bboxes[:, 1],
                bboxes[:, 2], bboxes[:, 3],
                scores
            ])
            keep_idxs = nms(bboxes_xyxy, nms_thr)
            bboxes = bboxes[keep_idxs]
            scores = scores[keep_idxs]
        
        return bboxes, scores
    
    def estimate_poses(
        self,
        image: np.ndarray,
        bboxes: np.ndarray,
        bbox_format: str = 'xyxy'
    ) -> Any:
        """Estimate poses for detected primates.
        
        Args:
            image: Input image as numpy array (BGR format).
            bboxes: Array of bounding boxes.
            bbox_format: Format of bboxes ('xyxy' or 'xywh').
            
        Returns:
            Pose estimation results containing keypoints and scores.
        """
        if len(bboxes) == 0:
            # Return empty result
            empty_instances = InstanceData()
            empty_instances.keypoints = np.array([])
            empty_instances.keypoint_scores = np.array([])
            return empty_instances
        
        # Prepare bboxes for pose estimation
        pose_results = inference_topdown(self.pose_model, image, bboxes, bbox_format)
        merged_results = merge_data_samples(pose_results)
        
        return merged_results.pred_instances
    
    def process_frame(
        self,
        frame: np.ndarray,
        bbox_thr: float = DEFAULT_BBOX_THR,
        kpt_thr: float = DEFAULT_KPT_THR,
        nms_thr: float = DEFAULT_NMS_THR,
        instance_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Process a single frame/image.
        
        Args:
            frame: Input frame/image as numpy array (BGR format).
            bbox_thr: Detection confidence threshold.
            kpt_thr: Keypoint confidence threshold.
            nms_thr: NMS threshold for overlapping detections.
            instance_id: Optional instance ID for smoothing (videos only).
            
        Returns:
            Dictionary containing detection and pose results.
        """
        # Detect primates
        bboxes, bbox_scores = self.detect_primates(frame, bbox_thr, nms_thr)
        
        # Initialize empty pose results
        keypoints = np.array([])
        keypoint_scores = np.array([])
        
        # Estimate poses if pose model is available
        if self.pose_model and len(bboxes) > 0:
            pose_results = self.estimate_poses(frame, bboxes)
            keypoints = pose_results.keypoints
            keypoint_scores = pose_results.keypoint_scores
            
            # Apply smoothing if available and instance_id provided
            if self.smoother and instance_id is not None and len(keypoints) > 0:
                smoothed_kpts = []
                for i, (kpts, scores) in enumerate(zip(keypoints, keypoint_scores)):
                    smooth_kpts = self.smoother.update(
                        instance_id=instance_id + i,
                        keypoints=kpts,
                        keypoint_scores=scores,
                        kpt_thr=kpt_thr
                    )
                    smoothed_kpts.append(smooth_kpts)
                keypoints = np.array(smoothed_kpts) if smoothed_kpts else keypoints
        
        return {
            'bboxes': bboxes.tolist() if len(bboxes) > 0 else [],
            'bbox_scores': bbox_scores.tolist() if len(bbox_scores) > 0 else [],
            'keypoints': keypoints.tolist() if len(keypoints) > 0 else [],
            'keypoint_scores': keypoint_scores.tolist() if len(keypoint_scores) > 0 else []
        }
    
    def visualize_frame(
        self,
        frame: np.ndarray,
        results: Dict[str, Any],
        viz_pose: bool = True
    ) -> np.ndarray:
        """Visualize detection/pose results on frame.
        
        Args:
            frame: Input frame/image as numpy array (BGR format).
            results: Dictionary containing detection and pose results.
            viz_pose: Whether to visualize poses AND bboxes (True) or just bboxes (False).
            
        Returns:
            Frame with visualizations drawn on it.
        """
        if viz_pose and results['keypoints']:
            # Create pseudo instances for visualization
            pred_instances = type('', (), {})()
            pred_instances.keypoints = np.array(results['keypoints'])
            pred_instances.keypoint_scores = np.array(results['keypoint_scores'])
            pred_instances.bboxes = np.array(results['bboxes'])  # Add bboxes to instances
            
            # Draw poses and bboxes (visualizer.draw_bbox defaults to True)
            vis_frame = self.visualizer.draw_poses(
                frame,
                pred_instances,
                bbox_scores=results['bbox_scores']
            )
        else:
            # Draw bounding boxes only (when viz_pose=False)
            vis_frame = frame.copy()
            for bbox, score in zip(results['bboxes'], results['bbox_scores']):
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{score:.2f}"
                cv2.putText(
                    vis_frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
                )
        
        return vis_frame
    
    def process_video(
        self,
        video_path: str,
        output_dir: Optional[str] = None,
        save_predictions: bool = False,
        save_viz: bool = False,
        viz_pose: bool = True,
        bbox_thr: float = DEFAULT_BBOX_THR,
        kpt_thr: float = DEFAULT_KPT_THR,
        nms_thr: float = DEFAULT_NMS_THR
    ) -> Dict[str, Any]:
        """Process a video file.
        
        Args:
            video_path: Path to input video file.
            output_dir: Directory to save outputs.
            save_predictions: Whether to save predictions as JSON.
            save_viz: Whether to save visualization video.
            viz_pose: Whether to visualize poses AND bboxes (True) or just bboxes (False). Default: True
            bbox_thr: Detection confidence threshold.
            kpt_thr: Keypoint confidence threshold.
            nms_thr: NMS threshold.
            
        Returns:
            Dictionary containing all frame predictions.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Setup output directory
        if output_dir:
            output_dir = Path(output_dir)
        else:
            output_dir = video_path.parent / f"{video_path.stem}_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer if needed
        video_writer = None
        if save_viz:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_video_path = output_dir / f"{video_path.stem}_processed.mp4"
            video_writer = cv2.VideoWriter(
                str(output_video_path), fourcc, fps, (width, height)
            )
        
        # Process frames
        all_predictions = []
        with tqdm(total=total_frames, desc="Processing video") as pbar:
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                results = self.process_frame(
                    frame, bbox_thr, kpt_thr, nms_thr,
                    instance_id=0  # Simple ID for smoothing
                )
                results['frame_idx'] = frame_idx
                all_predictions.append(results)
                
                # Visualize if needed
                if video_writer:
                    vis_frame = self.visualize_frame(frame, results, viz_pose)
                    video_writer.write(vis_frame)
                
                frame_idx += 1
                pbar.update(1)
        
        # Cleanup
        cap.release()
        if video_writer:
            video_writer.release()
        
        # Save predictions
        if save_predictions:
            pred_path = output_dir / f"{video_path.stem}_predictions.json"
            with open(pred_path, 'w') as f:
                json_tricks.dump(all_predictions, f, indent=2)
        
        return {'video': str(video_path), 'predictions': all_predictions}
    
    def process_image_directory(
        self,
        image_dir: str,
        output_dir: Optional[str] = None,
        save_predictions: bool = False,
        save_viz: bool = False,
        viz_pose: bool = True,
        bbox_thr: float = DEFAULT_BBOX_THR,
        kpt_thr: float = DEFAULT_KPT_THR,
        nms_thr: float = DEFAULT_NMS_THR
    ) -> Dict[str, Any]:
        """Process a directory of images.
        
        Args:
            image_dir: Path to directory containing images.
            output_dir: Directory to save outputs.
            save_predictions: Whether to save predictions as JSON.
            save_viz: Whether to save visualization images.
            viz_pose: Whether to visualize poses AND bboxes (True) or just bboxes (False). Default: True
            bbox_thr: Detection confidence threshold.
            kpt_thr: Keypoint confidence threshold.
            nms_thr: NMS threshold.
            
        Returns:
            Dictionary containing predictions for all images.
        """
        image_path = Path(image_dir)
        if not image_path.exists():
            raise FileNotFoundError(f"Path not found: {image_path}")
        
        # Check if it's a single file or directory
        if image_path.is_file():
            # Single image file
            image_files = [image_path]
            # Setup output directory for single file
            if output_dir:
                output_dir = Path(output_dir)
            else:
                output_dir = image_path.parent / f"{image_path.stem}_output"
        else:
            # Directory of images
            # Setup output directory
            if output_dir:
                output_dir = Path(output_dir)
            else:
                output_dir = image_path.parent / f"{image_path.name}_output"
            
            # Find all images
            image_files = []
            for ext in IMAGE_EXTENSIONS:
                image_files.extend(image_path.glob(ext))
            
            # Sort files
            image_files = natsort.natsorted(image_files)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not image_files:
            warnings.warn(f"No images found in {image_dir}")
            return {'directory': str(image_dir), 'predictions': {}}
        
        # Process images
        all_predictions = {}
        for img_path in tqdm(image_files, desc="Processing images"):
            # Read image
            image = cv2.imread(str(img_path))
            if image is None:
                warnings.warn(f"Failed to read {img_path}")
                continue
            
            # Process image
            results = self.process_frame(image, bbox_thr, kpt_thr, nms_thr)
            results['image_path'] = str(img_path)
            all_predictions[img_path.name] = results
            
            # Save visualization if needed
            if save_viz:
                vis_image = self.visualize_frame(image, results, viz_pose)
                viz_path = output_dir / f"{img_path.stem}_viz{img_path.suffix}"
                cv2.imwrite(str(viz_path), vis_image)
        
        # Save predictions
        if save_predictions:
            pred_path = output_dir / "predictions.json"
            with open(pred_path, 'w') as f:
                json_tricks.dump(all_predictions, f, indent=2)
        
        return {'directory': str(image_dir), 'predictions': all_predictions}


def main() -> int:
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Process videos or images for primate detection and pose estimation',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--input', required=True, help='Input video file or image directory')
    parser.add_argument('--input-type', required=True, choices=['video', 'image', 'images'],
                    help='Type of input: video file, single image, or image directory')
    parser.add_argument('--det-config', required=True, help='Detection model config file')
    parser.add_argument('--det-checkpoint', required=True, help='Detection model checkpoint')
    parser.add_argument('--pose-config', help='Pose model config file (optional, if not provided only detection is run)')
    parser.add_argument('--pose-checkpoint', help='Pose model checkpoint (optional, if not provided only detection is run)')
    
    # Optional arguments
    parser.add_argument('--output-dir', help='Output directory for results')
    parser.add_argument('--save-predictions', action='store_true',
                    help='Save predictions as JSON')
    parser.add_argument('--save-viz', action='store_true',
                    help='Save visualization output')
    parser.add_argument('--viz-pose', action='store_true', default=True,
                    help='Visualize both poses and bboxes together (default: True). Use --no-viz-pose for bboxes only')
    
    # Thresholds
    parser.add_argument('--bbox-thr', type=float, default=DEFAULT_BBOX_THR,
                    help=f'Detection confidence threshold (default: {DEFAULT_BBOX_THR})')
    parser.add_argument('--kpt-thr', type=float, default=DEFAULT_KPT_THR,
                    help=f'Keypoint confidence threshold (default: {DEFAULT_KPT_THR})')
    parser.add_argument('--nms-thr', type=float, default=DEFAULT_NMS_THR,
                    help=f'NMS threshold (default: {DEFAULT_NMS_THR})')
    
    # Smoothing (for videos)
    parser.add_argument('--smooth', action='store_true',
                    help='Apply temporal smoothing to keypoints (videos only)')
    parser.add_argument('--smooth-median-window', type=int, default=DEFAULT_MEDIAN_WINDOW,
                    help=f'Window size for median filter (default: {DEFAULT_MEDIAN_WINDOW})')
    parser.add_argument('--smooth-savgol-window', type=int, default=DEFAULT_SAVGOL_WINDOW,
                    help=f'Window size for Savitzky-Golay filter (default: {DEFAULT_SAVGOL_WINDOW})')
    parser.add_argument('--smooth-savgol-order', type=int, default=DEFAULT_SAVGOL_ORDER,
                    help=f'Polynomial order for Savitzky-Golay filter (default: {DEFAULT_SAVGOL_ORDER})')
    
    # Device
    parser.add_argument('--device', default='cuda:0',
                    help='Device to use for inference')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = PrimateFaceProcessor(
        det_config=args.det_config,
        det_checkpoint=args.det_checkpoint,
        pose_config=args.pose_config,
        pose_checkpoint=args.pose_checkpoint,
        device=args.device,
        use_smoothing=args.smooth and args.input_type == 'video',
        median_window=args.smooth_median_window,
        savgol_window=args.smooth_savgol_window,
        savgol_order=args.smooth_savgol_order
    )
    
    # Process based on input type
    if args.input_type == 'video':
        results = processor.process_video(
            video_path=args.input,
            output_dir=args.output_dir,
            save_predictions=args.save_predictions,
            save_viz=args.save_viz,
            viz_pose=args.viz_pose,
            bbox_thr=args.bbox_thr,
            kpt_thr=args.kpt_thr,
            nms_thr=args.nms_thr
        )
    elif args.input_type == 'image':
        # For single image, process as a directory with one file
        results = processor.process_image_directory(
            image_dir=args.input,
            output_dir=args.output_dir,
            save_predictions=args.save_predictions,
            save_viz=args.save_viz,
            viz_pose=args.viz_pose,
            bbox_thr=args.bbox_thr,
            kpt_thr=args.kpt_thr,
            nms_thr=args.nms_thr
        )
    else:  # images (directory)
        results = processor.process_image_directory(
            image_dir=args.input,
            output_dir=args.output_dir,
            save_predictions=args.save_predictions,
            save_viz=args.save_viz,
            viz_pose=args.viz_pose,
            bbox_thr=args.bbox_thr,
            kpt_thr=args.kpt_thr,
            nms_thr=args.nms_thr
        )
    
    print(f"Processing complete. Output saved to: {args.output_dir or 'default location'}")


if __name__ == '__main__':
    main()