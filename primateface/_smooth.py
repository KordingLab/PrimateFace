"""Temporal smoothing utilities for keypoint trajectories.

This module provides smoothing classes for removing noise and jitter from
keypoint trajectories in video sequences. It combines median filtering
for outlier removal with Savitzky-Golay filtering for smooth interpolation.

Example:
    smoother = MedianSavgolSmoother(median_window=5, savgol_window=7)
    smoothed_keypoints = smoother.smooth_frame(keypoints, scores, instance_id=1)
"""

from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import medfilt, savgol_filter

class MedianSavgolSmoother:
    """Applies Median filtering followed by Savitzky-Golay filtering to keypoint time series.
    
    This class maintains a history of keypoints for each tracked instance and applies
    temporal smoothing to reduce noise and jitter in pose estimation results.
    
    Attributes:
        median_window: Window size for median filter (must be odd)
        savgol_window: Window size for Savitzky-Golay filter (must be odd)
        savgol_order: Polynomial order for Savitzky-Golay filter
        min_history: Minimum number of frames needed for smoothing
        history: Dictionary storing recent keypoints for each instance ID
    """
    
    def __init__(
        self,
        median_window: int = 5,
        savgol_window: int = 7,
        savgol_order: int = 3
    ) -> None:
        """Initialize the smoother with filtering parameters.

        Args:
            median_window: Window size for the median filter (must be odd)
            savgol_window: Window size for the Savitzky-Golay filter (must be odd)
            savgol_order: Polynomial order for the Savitzky-Golay filter
                        (must be less than savgol_window)

        Raises:
            ValueError: If window sizes are not positive odd integers or if
                    savgol_order >= savgol_window
        """
        if median_window % 2 == 0 or median_window <= 0:
            raise ValueError("median_window must be a positive odd integer.")
        if savgol_window % 2 == 0 or savgol_window <= 0:
            raise ValueError("savgol_window must be a positive odd integer.")
        if savgol_order >= savgol_window:
            raise ValueError("savgol_order must be less than savgol_window.")

        self.median_window = median_window
        self.savgol_window = savgol_window
        self.savgol_order = savgol_order

        # Determine the required history length
        self.min_history = max(self.median_window, self.savgol_window)

        # Stores recent keypoints for each instance ID
        # history[instance_id] = deque(maxlen=required_history_length)
        # Each item in the deque is a tuple: (keypoints_array, scores_array)
        self.history: Dict[int, Deque[Tuple[np.ndarray, np.ndarray]]] = {}

    def _apply_filters(self, time_series: np.ndarray) -> np.ndarray:
        """Apply median and Savitzky-Golay filters to a 1D time series.
        
        Args:
            time_series: 1D array of values to filter
            
        Returns:
            Filtered time series of the same shape
            
        Note:
            If all values are NaN, returns the input unchanged.
            Savitzky-Golay filter is only applied if the time series
            is long enough (>= savgol_window).
        """
        # Handle NaNs: Apply filters only to valid segments if necessary,
        # but for simplicity, let's rely on the filters' default handling or
        # simply return NaNs if the input contains too many.
        # A more robust approach might involve interpolation first.
        if np.all(np.isnan(time_series)):
            return time_series # All NaNs, nothing to filter

        # Median Filter (scipy handles ends)
        filtered = medfilt(time_series, kernel_size=self.median_window)

        # Savitzky-Golay Filter (applied to median filter output)
        # Requires window_length <= len(time_series)
        if len(time_series) >= self.savgol_window:
            filtered = savgol_filter(filtered,
                                    window_length=self.savgol_window,
                                    polyorder=self.savgol_order,
                                    mode='interp')
        else:
            # If series is too short for Savgol, return median filtered result
            return filtered

        return filtered

    def update(
        self,
        instance_id: int,
        keypoints: np.ndarray,
        keypoint_scores: np.ndarray,
        kpt_thr: float = 0.3
    ) -> Optional[np.ndarray]:
        """Update smoother state and return smoothed keypoints for current frame.

        Args:
            instance_id: Unique ID of the tracked instance
            keypoints: Current detected keypoints, shape (num_keypoints, 2 or 3)
            keypoint_scores: Current confidence scores, shape (num_keypoints,)
            kpt_thr: Confidence threshold for reliable keypoints

        Returns:
            Smoothed keypoints array with same shape as input, or the original
            keypoints if insufficient history exists for smoothing
            
        Note:
            Keypoints with scores below kpt_thr are treated as unreliable
            and may be excluded from smoothing calculations.
        """
        if instance_id not in self.history:
            # Initialize deque with appropriate max length if needed
            # Let's use a slightly larger buffer than min_history for safety
            self.history[instance_id] = deque(maxlen=self.min_history * 2)

        # Store current valid data
        self.history[instance_id].append((keypoints.copy(), keypoint_scores.copy()))

        # Check if we have enough history
        if len(self.history[instance_id]) < self.min_history:
            # Not enough data, return the original keypoints for this frame
            return keypoints

        # --- Extract history for filtering ---
        hist_kps = [item[0] for item in self.history[instance_id]]
        hist_scores = [item[1] for item in self.history[instance_id]]

        # Stack history into (Time, NumKeypoints, Dims) and (Time, NumKeypoints)
        # Make sure all keypoints arrays have the same shape
        num_kps = hist_kps[0].shape[0]
        dims = hist_kps[0].shape[1]
        if not all(kp.shape == (num_kps, dims) for kp in hist_kps):
            print(f"Warning (ID: {instance_id}): Keypoint shape inconsistency in history. Cannot smooth.")
            # Return current raw keypoints
            self.reset_history(instance_id)
            return keypoints

        keypoints_history = np.stack(hist_kps, axis=0)      # (Time, NumKps, Dims)
        scores_history = np.stack(hist_scores, axis=0)      # (Time, NumKps)

        # --- Apply filters per keypoint coordinate ---
        smoothed_keypoints_current = np.copy(keypoints)

        for k in range(num_kps):
            # Extract time series for this keypoint's x and y
            x_series = keypoints_history[:, k, 0]
            y_series = keypoints_history[:, k, 1]
            score_series = scores_history[:, k]

            # Mark low-confidence points as NaN for filtering
            x_series[score_series < kpt_thr] = np.nan
            y_series[score_series < kpt_thr] = np.nan

            # Apply filters (handle potential all-NaN series within _apply_filters)
            smoothed_x = self._apply_filters(x_series)
            smoothed_y = self._apply_filters(y_series)

            # Get the last value from the filtered series (corresponding to current frame)
            # Use the original value if smoothing resulted in NaN
            current_x_smooth = smoothed_x[-1] if not np.isnan(smoothed_x[-1]) else keypoints[k, 0]
            current_y_smooth = smoothed_y[-1] if not np.isnan(smoothed_y[-1]) else keypoints[k, 1]

            smoothed_keypoints_current[k, 0] = current_x_smooth
            smoothed_keypoints_current[k, 1] = current_y_smooth
            # Keep original score if dims=3
            if dims == 3:
                smoothed_keypoints_current[k, 2] = keypoints[k, 2]


        return smoothed_keypoints_current


    def reset_history(self, instance_id: Optional[int] = None) -> None:
        """Resets the history for a specific instance or all instances."""
        if instance_id is not None:
            if instance_id in self.history:
                del self.history[instance_id]
        else:
            self.history.clear()
            print("MedianSavgolSmoother history reset for all IDs.")