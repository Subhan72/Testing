"""
Temporal stability detection module
Detects when video content becomes stable (no significant changes)
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional
from skimage.metrics import structural_similarity as ssim
import config


class StabilityDetector:
    """Detects temporal stability in video frames"""
    
    def __init__(self):
        """Initialize stability detector"""
        self.frame_history: List[Tuple[np.ndarray, float, int]] = []  # (frame, timestamp, frame_num)
        self.stability_window_duration = config.STABILITY_WINDOW_DURATION
        self.stability_threshold_ssim = config.STABILITY_THRESHOLD_SSIM
        self.stability_threshold_mse = config.STABILITY_THRESHOLD_MSE
        self.min_stable_frames = config.STABILITY_FRAME_COUNT
    
    def add_frame(self, frame: np.ndarray, timestamp: float, frame_number: int):
        """
        Add a frame to the history
        
        Args:
            frame: Frame image
            timestamp: Timestamp in seconds
            frame_number: Frame number
        """
        # Convert to grayscale for comparison
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        # Resize for faster comparison (optional optimization)
        if gray.shape[0] > 480 or gray.shape[1] > 640:
            scale = min(480 / gray.shape[0], 640 / gray.shape[1])
            new_h, new_w = int(gray.shape[0] * scale), int(gray.shape[1] * scale)
            gray = cv2.resize(gray, (new_w, new_h))
        
        self.frame_history.append((gray, timestamp, frame_number))
        
        # Keep only frames within stability window
        current_time = timestamp
        self.frame_history = [
            (f, t, n) for f, t, n in self.frame_history
            if current_time - t <= self.stability_window_duration + 1.0  # Keep some buffer
        ]
    
    def is_stable(self) -> Tuple[bool, float]:
        """
        Check if current content is stable
        
        Returns:
            Tuple of (is_stable, stability_score)
        """
        if len(self.frame_history) < self.min_stable_frames:
            return False, 0.0
        
        # Get frames within stability window
        current_time = self.frame_history[-1][1]
        window_frames = [
            f for f, t, n in self.frame_history
            if current_time - t <= self.stability_window_duration
        ]
        
        if len(window_frames) < self.min_stable_frames:
            return False, 0.0
        
        # Calculate stability metrics
        stability_scores = []
        
        # Compare consecutive frames
        for i in range(1, len(window_frames)):
            frame1 = window_frames[i-1]
            frame2 = window_frames[i]
            
            # Ensure same size
            if frame1.shape != frame2.shape:
                h, w = min(frame1.shape[0], frame2.shape[0]), min(frame1.shape[1], frame2.shape[1])
                frame1 = frame1[:h, :w]
                frame2 = frame2[:h, :w]
            
            # Calculate SSIM
            ssim_score = ssim(frame1, frame2, data_range=255)
            
            # Calculate MSE
            mse = np.mean((frame1.astype(float) - frame2.astype(float)) ** 2)
            
            # Combined stability score
            ssim_ok = ssim_score >= self.stability_threshold_ssim
            mse_ok = mse <= self.stability_threshold_mse
            
            if ssim_ok and mse_ok:
                stability_scores.append(ssim_score)
            else:
                stability_scores.append(0.0)
        
        if not stability_scores:
            return False, 0.0
        
        avg_stability = np.mean(stability_scores)
        is_stable = avg_stability >= self.stability_threshold_ssim and len(stability_scores) >= self.min_stable_frames - 1
        
        return is_stable, avg_stability
    
    def get_stable_frame(self) -> Optional[Tuple[np.ndarray, float, int]]:
        """
        Get the most recent stable frame
        
        Returns:
            Tuple of (frame, timestamp, frame_number) or None
        """
        if not self.frame_history:
            return None
        
        # Return the most recent frame (which should be stable if is_stable() returns True)
        return self.frame_history[-1]
    
    def calculate_frame_difference(self, frame1: np.ndarray, frame2: np.ndarray) -> Tuple[float, float]:
        """
        Calculate difference between two frames
        
        Args:
            frame1: First frame
            frame2: Second frame
            
        Returns:
            Tuple of (ssim_score, mse)
        """
        # Ensure same size
        if frame1.shape != frame2.shape:
            h, w = min(frame1.shape[0], frame2.shape[0]), min(frame1.shape[1], frame2.shape[1])
            frame1 = frame1[:h, :w]
            frame2 = frame2[:h, :w]
        
        # Convert to grayscale if needed
        if len(frame1.shape) == 3:
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        if len(frame2.shape) == 3:
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Calculate SSIM
        ssim_score = ssim(frame1, frame2, data_range=255)
        
        # Calculate MSE
        mse = np.mean((frame1.astype(float) - frame2.astype(float)) ** 2)
        
        return ssim_score, mse
    
    def reset(self):
        """Reset frame history"""
        self.frame_history = []
