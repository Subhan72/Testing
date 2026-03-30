"""
Video processing module for extracting frames from video files
"""
import cv2
import numpy as np
from typing import Generator, Tuple, Optional
import os
from tqdm import tqdm
import config


class VideoProcessor:
    """Handles video file reading and frame extraction"""
    
    def __init__(self, video_path: str):
        """
        Initialize video processor
        
        Args:
            video_path: Path to input video file
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0
        
        # Calculate frame interval
        self.frame_interval = int(self.fps * config.FRAME_EXTRACTION_INTERVAL)
        if self.frame_interval < 1:
            self.frame_interval = 1
        
        if config.VERBOSE:
            print(f"Video Info:")
            print(f"  Resolution: {self.width}x{self.height}")
            print(f"  FPS: {self.fps:.2f}")
            print(f"  Duration: {self.duration:.2f} seconds")
            print(f"  Total frames: {self.frame_count}")
            print(f"  Frame extraction interval: {config.FRAME_EXTRACTION_INTERVAL}s ({self.frame_interval} frames)")
    
    def extract_frames(self) -> Generator[Tuple[np.ndarray, int, float], None, None]:
        """
        Extract frames from video at configured intervals
        
        Yields:
            Tuple of (frame, frame_number, timestamp)
        """
        frame_number = 0
        extracted_count = 0
        
        if config.VERBOSE:
            pbar = tqdm(total=self.frame_count, desc="Extracting frames")
        
        while True:
            ret, frame = self.cap.read()
            
            if not ret:
                break
            
            # Extract frame at intervals
            if frame_number % self.frame_interval == 0:
                timestamp = frame_number / self.fps if self.fps > 0 else 0
                extracted_count += 1
                
                if config.MAX_FRAMES_TO_PROCESS and extracted_count > config.MAX_FRAMES_TO_PROCESS:
                    break
                
                yield frame.copy(), frame_number, timestamp
            
            frame_number += 1
            
            if config.VERBOSE:
                pbar.update(1)
        
        if config.VERBOSE:
            pbar.close()
        
        self.cap.release()
    
    def get_frame_at_timestamp(self, timestamp: float) -> Optional[np.ndarray]:
        """
        Get a specific frame at given timestamp
        
        Args:
            timestamp: Timestamp in seconds
            
        Returns:
            Frame at timestamp or None if not found
        """
        frame_number = int(timestamp * self.fps)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        
        if ret:
            return frame
        return None
    
    def get_frame_at_index(self, frame_index: int) -> Optional[np.ndarray]:
        """
        Get a specific frame by frame number
        
        Args:
            frame_index: Frame number
            
        Returns:
            Frame at index or None if not found
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self.cap.read()
        
        if ret:
            return frame
        return None
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'cap'):
            self.cap.release()
