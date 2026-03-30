"""
Content analysis module
Analyzes frame content to detect diagram-like structures
"""
import cv2
import numpy as np
from typing import Dict, List, Tuple
import config


class ContentAnalyzer:
    """Analyzes content to identify diagram-like structures"""
    
    def __init__(self):
        """Initialize content analyzer"""
        self.canny_low = config.CANNY_LOW_THRESHOLD
        self.canny_high = config.CANNY_HIGH_THRESHOLD
        self.min_contour_area = config.MIN_CONTOUR_AREA
        self.min_edge_density = config.MIN_EDGE_DENSITY
        self.min_contour_count = config.MIN_CONTOUR_COUNT
        self.min_shape_variety = config.MIN_SHAPE_VARIETY
    
    def analyze(self, frame: np.ndarray) -> Dict:
        """
        Analyze frame content for diagram-like features
        
        Args:
            frame: Input frame
            
        Returns:
            Dictionary with analysis results
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        # Edge detection
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        significant_contours = [c for c in contours if cv2.contourArea(c) >= self.min_contour_area]
        
        # Analyze shapes
        shape_analysis = self._analyze_shapes(significant_contours, gray.shape)
        
        # Calculate complexity metrics
        complexity_score = self._calculate_complexity(edges, significant_contours, gray.shape)
        
        # Detect geometric structures
        geometric_features = self._detect_geometric_structures(edges, gray)
        
        # Calculate overall diagram score
        diagram_score = self._calculate_diagram_score(
            edge_density,
            len(significant_contours),
            shape_analysis,
            complexity_score,
            geometric_features
        )
        
        return {
            'edge_density': edge_density,
            'contour_count': len(significant_contours),
            'shape_analysis': shape_analysis,
            'complexity_score': complexity_score,
            'geometric_features': geometric_features,
            'diagram_score': diagram_score,
            'is_diagram_like': diagram_score >= config.MIN_COMPLETENESS_SCORE
        }
    
    def _analyze_shapes(self, contours: List, image_shape: Tuple[int, int]) -> Dict:
        """
        Analyze shapes in contours
        
        Args:
            contours: List of contours
            image_shape: Shape of image (height, width)
            
        Returns:
            Dictionary with shape analysis
        """
        if not contours:
            return {
                'circles': 0,
                'rectangles': 0,
                'lines': 0,
                'complex_shapes': 0,
                'shape_variety': 0
            }
        
        circles = 0
        rectangles = 0
        lines = 0
        complex_shapes = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_contour_area:
                continue
            
            # Approximate contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check shape type
            if len(approx) == 2:
                lines += 1
            elif len(approx) == 4:
                # Check if it's roughly rectangular
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h if h > 0 else 0
                rect_area = w * h
                extent = float(area) / rect_area if rect_area > 0 else 0
                
                if extent > 0.7:  # Close to rectangular
                    rectangles += 1
                else:
                    complex_shapes += 1
            elif len(approx) >= 8:
                # Check if it's roughly circular
                (x, y), radius = cv2.minEnclosingCircle(contour)
                circle_area = np.pi * radius * radius
                extent = float(area) / circle_area if circle_area > 0 else 0
                
                if extent > 0.7:  # Close to circular
                    circles += 1
                else:
                    complex_shapes += 1
            else:
                complex_shapes += 1
        
        # Calculate shape variety
        shape_types = sum([
            1 if circles > 0 else 0,
            1 if rectangles > 0 else 0,
            1 if lines > 0 else 0,
            1 if complex_shapes > 0 else 0
        ])
        
        return {
            'circles': circles,
            'rectangles': rectangles,
            'lines': lines,
            'complex_shapes': complex_shapes,
            'shape_variety': shape_types
        }
    
    def _detect_geometric_structures(self, edges: np.ndarray, gray: np.ndarray) -> Dict:
        """
        Detect geometric structures (lines, circles) using Hough transforms
        
        Args:
            edges: Edge image
            gray: Grayscale image
            
        Returns:
            Dictionary with geometric features
        """
        # Hough lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
        line_count = len(lines) if lines is not None else 0
        
        # Hough circles
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
            param1=50, param2=30, minRadius=5, maxRadius=100
        )
        circle_count = len(circles[0]) if circles is not None else 0
        
        # Detect grid-like patterns (multiple parallel lines)
        grid_score = 0.0
        if lines is not None and len(lines) > 5:
            # Check for parallel lines (similar angles)
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                angles.append(angle)
            
            # Group similar angles
            angle_groups = {}
            for angle in angles:
                rounded = round(angle / 10) * 10
                angle_groups[rounded] = angle_groups.get(rounded, 0) + 1
            
            # If many lines have similar angles, might be a grid
            max_group_size = max(angle_groups.values()) if angle_groups else 0
            if max_group_size > 3:
                grid_score = min(max_group_size / 10.0, 1.0)
        
        return {
            'line_count': line_count,
            'circle_count': circle_count,
            'grid_score': grid_score,
            'has_geometric_structure': line_count > 5 or circle_count > 2 or grid_score > 0.3
        }
    
    def _calculate_complexity(self, edges: np.ndarray, contours: List, image_shape: Tuple[int, int]) -> float:
        """
        Calculate complexity score of the content
        
        Args:
            edges: Edge image
            contours: List of contours
            image_shape: Shape of image
            
        Returns:
            Complexity score (0-1)
        """
        if not contours:
            return 0.0
        
        # Edge density component
        edge_density = np.sum(edges > 0) / edges.size
        edge_score = min(edge_density / 0.1, 1.0)  # Normalize
        
        # Contour count component
        contour_score = min(len(contours) / 20.0, 1.0)  # Normalize
        
        # Contour area distribution
        areas = [cv2.contourArea(c) for c in contours]
        if areas:
            area_variance = np.var(areas) / (np.mean(areas) ** 2 + 1e-6)
            variance_score = min(area_variance, 1.0)
        else:
            variance_score = 0.0
        
        # Combined complexity
        complexity = (edge_score * 0.4 + contour_score * 0.3 + variance_score * 0.3)
        
        return complexity
    
    def _calculate_diagram_score(
        self,
        edge_density: float,
        contour_count: int,
        shape_analysis: Dict,
        complexity_score: float,
        geometric_features: Dict
    ) -> float:
        """
        Calculate overall diagram likelihood score
        
        Args:
            edge_density: Edge density
            contour_count: Number of contours
            shape_analysis: Shape analysis results
            complexity_score: Complexity score
            geometric_features: Geometric features
            
        Returns:
            Diagram score (0-1)
        """
        # Edge density component
        edge_score = 1.0 if edge_density >= self.min_edge_density else edge_density / self.min_edge_density
        
        # Contour count component
        contour_score = 1.0 if contour_count >= self.min_contour_count else contour_count / self.min_contour_count
        
        # Shape variety component
        shape_variety = shape_analysis['shape_variety']
        variety_score = 1.0 if shape_variety >= self.min_shape_variety else shape_variety / self.min_shape_variety
        
        # Geometric structure component
        geo_score = 1.0 if geometric_features['has_geometric_structure'] else 0.5
        
        # Weighted combination
        diagram_score = (
            edge_score * 0.25 +
            contour_score * 0.25 +
            variety_score * 0.25 +
            complexity_score * 0.15 +
            geo_score * 0.10
        )
        
        return min(diagram_score, 1.0)
