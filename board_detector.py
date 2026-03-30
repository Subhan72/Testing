"""
Board detection module
Detects blackboard/whiteboard regions in video frames
"""
import cv2
import numpy as np
from typing import Tuple, Optional, List
import config


class BoardDetector:
    """Detects and extracts board regions from frames"""
    
    def __init__(self):
        """Initialize board detector"""
        self.blackboard_lower = np.array(config.BLACKBOARD_LOWER_HSV)
        self.blackboard_upper = np.array(config.BLACKBOARD_UPPER_HSV)
        self.whiteboard_lower = np.array(config.WHITEBOARD_LOWER_HSV)
        self.whiteboard_upper = np.array(config.WHITEBOARD_UPPER_HSV)
        self.min_board_area_ratio = config.MIN_BOARD_AREA_RATIO
    
    def detect_board(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], str, float]:
        """
        Detect board region in frame
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (board_region, board_type, confidence)
            board_region: Extracted board region or None
            board_type: "blackboard", "whiteboard", or "unknown"
            confidence: Detection confidence (0-1)
        """
        if not config.BOARD_DETECTION_ENABLED:
            # Return full frame if board detection is disabled
            return frame, "unknown", 1.0
        
        h, w = frame.shape[:2]
        total_area = h * w
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Try to detect blackboard
        blackboard_mask = cv2.inRange(hsv, self.blackboard_lower, self.blackboard_upper)
        blackboard_area = np.sum(blackboard_mask > 0)
        blackboard_ratio = blackboard_area / total_area
        
        # Try to detect whiteboard
        whiteboard_mask = cv2.inRange(hsv, self.whiteboard_lower, self.whiteboard_upper)
        whiteboard_area = np.sum(whiteboard_mask > 0)
        whiteboard_ratio = whiteboard_area / total_area
        
        # Determine board type
        if blackboard_ratio > whiteboard_ratio and blackboard_ratio >= self.min_board_area_ratio:
            board_type = "blackboard"
            mask = blackboard_mask
            confidence = min(blackboard_ratio * 2, 1.0)  # Normalize confidence
        elif whiteboard_ratio >= self.min_board_area_ratio:
            board_type = "whiteboard"
            mask = whiteboard_mask
            confidence = min(whiteboard_ratio * 2, 1.0)
        else:
            # No clear board detected, return full frame
            return frame, "unknown", 0.5
        
        # Find contours to get board region
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return frame, board_type, confidence
        
        # Find largest contour (likely the board)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        x, y, w_rect, h_rect = cv2.boundingRect(largest_contour)
        
        # Expand rectangle slightly to include edges
        margin = 10
        x = max(0, x - margin)
        y = max(0, y - margin)
        w_rect = min(w - x, w_rect + 2 * margin)
        h_rect = min(h - y, h_rect + 2 * margin)
        
        # Extract board region
        board_region = frame[y:y+h_rect, x:x+w_rect]
        
        # If extracted region is too small, return full frame
        if board_region.size < total_area * self.min_board_area_ratio:
            return frame, board_type, confidence * 0.5
        
        return board_region, board_type, confidence
    
    def detect_board_edges(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect board using edge detection (alternative method)
        
        Args:
            frame: Input frame
            
        Returns:
            Board region or None
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find largest rectangular contour
        largest_contour = None
        max_area = 0
        
        for contour in contours:
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if it's roughly rectangular (4 corners)
            if len(approx) >= 4:
                area = cv2.contourArea(contour)
                if area > max_area:
                    max_area = area
                    largest_contour = contour
        
        if largest_contour is None:
            return None
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Extract region
        board_region = frame[y:y+h, x:x+w]
        
        return board_region
    
    def enhance_board_contrast(self, board_region: np.ndarray, board_type: str) -> np.ndarray:
        """
        Enhance contrast of board region for better diagram detection
        
        Args:
            board_region: Board region image
            board_type: Type of board ("blackboard" or "whiteboard")
            
        Returns:
            Enhanced board region
        """
        if board_type == "blackboard":
            # Enhance contrast for blackboard (dark background, light content)
            gray = cv2.cvtColor(board_region, cv2.COLOR_BGR2GRAY)
            enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
            return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        elif board_type == "whiteboard":
            # Enhance contrast for whiteboard (light background, dark content)
            gray = cv2.cvtColor(board_region, cv2.COLOR_BGR2GRAY)
            enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
            return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        else:
            return board_region
