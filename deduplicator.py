"""
Deduplication module
Removes duplicate diagrams using image similarity
Handles pointer variations and progressive diagram building
"""
import cv2
import numpy as np
import imagehash
from PIL import Image
from typing import List, Tuple, Optional, Dict
from skimage.metrics import structural_similarity as ssim
import config


class Deduplicator:
    """Removes duplicate diagrams with temporal context for progressive building"""
    
    def __init__(self):
        """Initialize deduplicator"""
        self.saved_diagrams: List[Tuple[np.ndarray, str]] = []  # (image, hash)
        # Track recent diagrams with metadata for progressive detection
        self.recent_diagrams: List[Dict] = []  # List of {image, hash, timestamp, edge_density, contour_count, diagram_id}
        self.phash_threshold = config.SIMILARITY_THRESHOLD_PHASH
        self.ssim_threshold = config.SIMILARITY_THRESHOLD_SSIM
        self.hist_threshold = config.SIMILARITY_THRESHOLD_HIST
        # Temporal window for progressive diagram detection (seconds)
        self.temporal_window = 600.0  # Check diagrams within 10 minutes (increased for longer sequences)
        # Content increase thresholds for progressive building (more lenient)
        self.min_content_increase = 0.003  # Minimum edge density increase (very reduced for blackboard diagrams)
        self.min_contour_increase = 1  # Minimum contour count increase
        # Pointer variation detection - if content is very similar (within small range), it's likely pointer
        self.pointer_variation_edge_tolerance = 0.003  # Edge density variation for pointer (increased tolerance)
        self.pointer_variation_contour_tolerance = 4  # Contour count variation for pointer (increased)
    
    def check_diagram(
        self, 
        image: np.ndarray, 
        timestamp: float,
        edge_density: float,
        contour_count: int
    ) -> Tuple[str, Optional[int], Optional[str], float]:
        """
        Check if diagram is duplicate or progressive version
        
        Args:
            image: Image to check
            timestamp: Timestamp in video
            edge_density: Edge density of current diagram
            contour_count: Contour count of current diagram
            
        Returns:
            Tuple of (action, diagram_id_to_replace, similar_hash, similarity_score)
            action: 'skip' (duplicate/pointer), 'replace' (progressive), 'save' (new)
            diagram_id_to_replace: ID of diagram to replace if action is 'replace'
        """
        if not config.DEDUPLICATION_ENABLED:
            return 'save', None, None, 0.0
        
        # Calculate perceptual hash
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        current_hash = str(imagehash.phash(pil_image))
        
        # First check recent diagrams for progressive building (within temporal window)
        best_recent_match = None
        best_recent_similarity = 0.0
        
        for recent in self.recent_diagrams:
            # Check if within temporal window
            time_diff = abs(timestamp - recent['timestamp'])
            if time_diff > self.temporal_window:
                continue
            
            # Check similarity
            hash_distance = imagehash.hex_to_hash(current_hash) - imagehash.hex_to_hash(recent['hash'])
            
            # More lenient threshold for recent diagrams
            if hash_distance <= self.phash_threshold * 2.0:  # More lenient for recent
                ssim_score, hist_score = self._calculate_similarity(image, recent['image'])
                similarity = (
                    (1.0 - hash_distance / 64.0) * 0.4 +
                    ssim_score * 0.4 +
                    hist_score * 0.2
                )
                
                if similarity > best_recent_similarity:
                    best_recent_similarity = similarity
                    best_recent_match = recent
        
        # If found similar recent diagram, check if it's progressive building or pointer variation
        if best_recent_match and best_recent_similarity >= 0.80:  # Lowered threshold (was 0.85)
            old_edge = best_recent_match['edge_density']
            old_contour = best_recent_match['contour_count']
            
            # Calculate content differences
            edge_increase = edge_density - old_edge
            contour_increase = contour_count - old_contour
            edge_decrease = old_edge - edge_density
            contour_decrease = old_contour - contour_count
            
            # Check for pointer variation: content is very similar (within tolerance)
            is_pointer_variation = (
                abs(edge_increase) <= self.pointer_variation_edge_tolerance and
                abs(contour_increase) <= self.pointer_variation_contour_tolerance
            )
            
            if is_pointer_variation:
                # Pointer variation - skip
                return 'skip', None, best_recent_match['hash'], best_recent_similarity
            
            # Check if new diagram has significantly more content (progressive building)
            if edge_increase >= self.min_content_increase or contour_increase >= self.min_contour_increase:
                # Progressive building detected - replace old diagram
                return 'replace', best_recent_match['diagram_id'], best_recent_match['hash'], best_recent_similarity
            elif edge_decrease >= self.min_content_increase or contour_decrease >= self.min_contour_increase:
                # Old diagram has more content - skip new one (it's an intermediate)
                return 'skip', None, best_recent_match['hash'], best_recent_similarity
            else:
                # Similar content levels - likely pointer variation, skip
                return 'skip', None, best_recent_match['hash'], best_recent_similarity
        
        # Check all saved diagrams for duplicates (including checking content similarity)
        # Also check for progressive building across all saved diagrams (not just recent)
        # Track all matches to find the best one to replace (earliest in sequence)
        all_matches = []  # List of (similarity, hash, diagram_id, content, timestamp)
        
        for saved_image, saved_hash in self.saved_diagrams:
            hash_distance = imagehash.hex_to_hash(current_hash) - imagehash.hex_to_hash(saved_hash)
            
            # More lenient threshold for checking all saved diagrams
            if hash_distance <= self.phash_threshold * 2.0:  # Even more lenient
                ssim_score, hist_score = self._calculate_similarity(image, saved_image)
                similarity = (
                    (1.0 - hash_distance / 64.0) * 0.4 +
                    ssim_score * 0.4 +
                    hist_score * 0.2
                )
                
                # Try to get content info from recent diagrams
                match_content = None
                match_diagram_id = None
                match_timestamp = None
                for recent in self.recent_diagrams:
                    if recent['hash'] == saved_hash:
                        match_content = {
                            'edge_density': recent['edge_density'],
                            'contour_count': recent['contour_count']
                        }
                        match_diagram_id = recent['diagram_id']
                        match_timestamp = recent['timestamp']
                        break
                
                # Store match for later processing
                if similarity >= 0.70:  # Lower threshold to catch all potential matches
                    all_matches.append({
                        'similarity': similarity,
                        'hash': saved_hash,
                        'diagram_id': match_diagram_id,
                        'content': match_content,
                        'timestamp': match_timestamp
                    })
        
        # Process all matches to find best action
        if all_matches:
            # First, check for pointer variations (highest priority - skip immediately)
            for match in all_matches:
                similarity = match['similarity']
                saved_hash = match['hash']
                match_content = match['content']
                
                if match_content and similarity >= 0.70:
                    old_edge = match_content['edge_density']
                    old_contour = match_content['contour_count']
                    edge_diff = abs(edge_density - old_edge)
                    contour_diff = abs(contour_count - old_contour)
                    
                    # Check for pointer variation: content is very similar
                    # This should catch diagrams 24-30 (all have edge ~0.037-0.038, contours 14-17)
                    if (edge_diff <= self.pointer_variation_edge_tolerance and
                        contour_diff <= self.pointer_variation_contour_tolerance):
                        return 'skip', None, saved_hash, similarity
            
            # Then check for progressive building - find earliest diagram in sequence
            progressive_matches = []
            for match in all_matches:
                similarity = match['similarity']
                saved_hash = match['hash']
                match_diagram_id = match['diagram_id']
                match_content = match['content']
                match_timestamp = match['timestamp']
                
                if match_content and similarity >= 0.75:
                    old_edge = match_content['edge_density']
                    old_contour = match_content['contour_count']
                    edge_increase = edge_density - old_edge
                    contour_increase = contour_count - old_contour
                    
                    # Check for progressive building: new has more content
                    time_diff = abs(timestamp - match_timestamp) if match_timestamp else float('inf')
                    if (time_diff <= self.temporal_window and
                        (edge_increase >= self.min_content_increase or 
                         contour_increase >= self.min_contour_increase)):
                        progressive_matches.append({
                            'similarity': similarity,
                            'hash': saved_hash,
                            'diagram_id': match_diagram_id,
                            'timestamp': match_timestamp or float('inf')
                        })
            
            # If progressive building detected, replace the earliest one (lowest timestamp)
            if progressive_matches:
                # Sort by timestamp (earliest first)
                progressive_matches.sort(key=lambda x: x['timestamp'])
                best_progressive = progressive_matches[0]
                return 'replace', best_progressive['diagram_id'], best_progressive['hash'], best_progressive['similarity']
            
            # Finally, check for duplicates (high similarity but no content change)
            for match in all_matches:
                similarity = match['similarity']
                saved_hash = match['hash']
                if similarity >= self.ssim_threshold:
                    return 'skip', None, saved_hash, similarity
        
        # Use best match for reporting
        best_similarity = all_matches[0]['similarity'] if all_matches else 0.0
        best_match_hash = all_matches[0]['hash'] if all_matches else None
        
        # New unique diagram
        return 'save', None, best_match_hash, best_similarity
    
    def _calculate_similarity(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[float, float]:
        """
        Calculate similarity between two images
        
        Args:
            img1: First image
            img2: Second image
            
        Returns:
            Tuple of (ssim_score, histogram_correlation)
        """
        # Resize to same size for comparison
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        target_h = min(h1, h2)
        target_w = min(w1, w2)
        
        img1_resized = cv2.resize(img1, (target_w, target_h))
        img2_resized = cv2.resize(img2, (target_w, target_h))
        
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY) if len(img1_resized.shape) == 3 else img1_resized
        gray2 = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY) if len(img2_resized.shape) == 3 else img2_resized
        
        # Calculate SSIM
        try:
            ssim_score = ssim(gray1, gray2, data_range=255)
        except:
            ssim_score = 0.0
        
        # Calculate histogram correlation
        hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
        hist_corr = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        return ssim_score, hist_corr
    
    def add_diagram(
        self, 
        image: np.ndarray, 
        diagram_id: int,
        timestamp: float,
        edge_density: float,
        contour_count: int
    ):
        """
        Add diagram to saved list and recent diagrams
        
        Args:
            image: Diagram image to add
            diagram_id: ID of the diagram
            timestamp: Timestamp in video
            edge_density: Edge density
            contour_count: Contour count
        """
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        diagram_hash = str(imagehash.phash(pil_image))
        self.saved_diagrams.append((image.copy(), diagram_hash))
        
        # Add to recent diagrams for progressive detection
        self.recent_diagrams.append({
            'image': image.copy(),
            'hash': diagram_hash,
            'diagram_id': diagram_id,
            'timestamp': timestamp,
            'edge_density': edge_density,
            'contour_count': contour_count
        })
        
        # Clean up old recent diagrams (outside temporal window)
        current_time = timestamp
        self.recent_diagrams = [
            r for r in self.recent_diagrams
            if abs(current_time - r['timestamp']) <= self.temporal_window
        ]
    
    def remove_diagram_from_recent(self, diagram_id: int):
        """
        Remove diagram from recent list (when replaced)
        
        Args:
            diagram_id: ID of diagram to remove
        """
        self.recent_diagrams = [
            r for r in self.recent_diagrams
            if r['diagram_id'] != diagram_id
        ]
    
    def replace_diagram_in_saved(self, old_hash: str, new_image: np.ndarray, new_hash: str):
        """
        Replace diagram in saved_diagrams list
        
        Args:
            old_hash: Hash of diagram to replace
            new_image: New diagram image
            new_hash: Hash of new diagram
        """
        # Find and replace the diagram with matching hash
        for i, (saved_image, saved_hash) in enumerate(self.saved_diagrams):
            if saved_hash == old_hash:
                self.saved_diagrams[i] = (new_image.copy(), new_hash)
                break
    
    def get_similarity_stats(self) -> Dict:
        """
        Get statistics about saved diagrams
        
        Returns:
            Dictionary with statistics
        """
        return {
            'total_diagrams': len(self.saved_diagrams),
            'phash_threshold': self.phash_threshold,
            'ssim_threshold': self.ssim_threshold
        }
    
    def reset(self):
        """Reset saved diagrams list"""
        self.saved_diagrams = []
