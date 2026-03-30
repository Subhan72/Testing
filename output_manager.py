"""
Output manager module
Handles saving extracted diagrams and metadata
"""
import cv2
import json
import os
from typing import Dict, List, Optional
from datetime import datetime
import config


class OutputManager:
    """Manages output of extracted diagrams"""
    
    def __init__(self, output_dir: str = None):
        """
        Initialize output manager
        
        Args:
            output_dir: Output directory path
        """
        self.output_dir = output_dir or config.OUTPUT_DIR
        self.image_format = config.OUTPUT_IMAGE_FORMAT.lower()
        self.image_quality = config.OUTPUT_IMAGE_QUALITY
        self.metadata_filename = config.METADATA_FILENAME
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create subdirectories
        self.diagrams_dir = os.path.join(self.output_dir, 'diagrams')
        self.enhanced_diagrams_dir = os.path.join(self.output_dir, 'enhanced_diagrams')
        os.makedirs(self.diagrams_dir, exist_ok=True)
        os.makedirs(self.enhanced_diagrams_dir, exist_ok=True)
        
        # Metadata storage
        self.metadata: List[Dict] = []
        self.diagram_count = 0
        # Track diagram IDs to file mappings for replacement
        self.diagram_id_to_file: Dict[int, str] = {}
        # Track enhanced diagram paths
        self.enhanced_diagram_paths: Dict[int, str] = {}
    
    def save_diagram(
        self,
        image: any,
        timestamp: float,
        frame_number: int,
        board_type: str,
        completeness_score: float,
        analysis_results: Dict,
        stability_score: float = 0.0
    ) -> str:
        """
        Save diagram image and add to metadata
        
        Args:
            image: Diagram image
            timestamp: Timestamp in video
            frame_number: Frame number
            board_type: Type of board
            completeness_score: Completeness score
            analysis_results: Content analysis results
            stability_score: Stability score
            
        Returns:
            Path to saved image
        """
        # Generate filename
        self.diagram_count += 1
        filename = f"diagram_{self.diagram_count:04d}.{self.image_format}"
        filepath = os.path.join(self.diagrams_dir, filename)
        
        # Save image
        if self.image_format == 'png':
            cv2.imwrite(filepath, image)
        else:  # JPEG
            cv2.imwrite(filepath, image, [cv2.IMWRITE_JPEG_QUALITY, self.image_quality])
        
        # Create metadata entry
        metadata_entry = {
            'diagram_id': self.diagram_count,
            'filename': filename,
            'filepath': filepath,
            'timestamp': timestamp,
            'frame_number': frame_number,
            'board_type': board_type,
            'completeness_score': round(completeness_score, 4),
            'stability_score': round(stability_score, 4),
            'edge_density': round(analysis_results.get('edge_density', 0.0), 4),
            'contour_count': analysis_results.get('contour_count', 0),
            'diagram_score': round(analysis_results.get('diagram_score', 0.0), 4),
            'shape_analysis': analysis_results.get('shape_analysis', {}),
            'geometric_features': analysis_results.get('geometric_features', {}),
            'extraction_time': datetime.now().isoformat()
        }
        
        self.metadata.append(metadata_entry)
        self.diagram_id_to_file[self.diagram_count] = filepath
        
        return filepath
    
    def replace_diagram(
        self,
        diagram_id: int,
        image: any,
        timestamp: float,
        frame_number: int,
        board_type: str,
        completeness_score: float,
        analysis_results: Dict,
        stability_score: float = 0.0
    ) -> str:
        """
        Replace an existing diagram with a new version (progressive building)
        
        Args:
            diagram_id: ID of diagram to replace
            image: New diagram image
            timestamp: Timestamp in video
            frame_number: Frame number
            board_type: Type of board
            completeness_score: Completeness score
            analysis_results: Content analysis results
            stability_score: Stability score
            
        Returns:
            Path to saved image
        """
        # Get existing filepath
        if diagram_id in self.diagram_id_to_file:
            filepath = self.diagram_id_to_file[diagram_id]
            filename = os.path.basename(filepath)
        else:
            # Fallback: generate new filename
            filename = f"diagram_{diagram_id:04d}.{self.image_format}"
            filepath = os.path.join(self.output_dir, filename)
        
        # Overwrite image
        if self.image_format == 'png':
            cv2.imwrite(filepath, image)
        else:  # JPEG
            cv2.imwrite(filepath, image, [cv2.IMWRITE_JPEG_QUALITY, self.image_quality])
        
        # Update metadata entry
        for entry in self.metadata:
            if entry['diagram_id'] == diagram_id:
                entry.update({
                    'timestamp': timestamp,
                    'frame_number': frame_number,
                    'board_type': board_type,
                    'completeness_score': round(completeness_score, 4),
                    'stability_score': round(stability_score, 4),
                    'edge_density': round(analysis_results.get('edge_density', 0.0), 4),
                    'contour_count': analysis_results.get('contour_count', 0),
                    'diagram_score': round(analysis_results.get('diagram_score', 0.0), 4),
                    'shape_analysis': analysis_results.get('shape_analysis', {}),
                    'geometric_features': analysis_results.get('geometric_features', {}),
                    'extraction_time': datetime.now().isoformat(),
                    'replaced': True  # Mark as replaced
                })
                break
        
        return filepath
    
    def save_metadata(self):
        """Save metadata to JSON file"""
        metadata_filepath = os.path.join(self.output_dir, self.metadata_filename)
        
        metadata_dict = {
            'total_diagrams': len(self.metadata),
            'extraction_date': datetime.now().isoformat(),
            'configuration': {
                'frame_extraction_interval': config.FRAME_EXTRACTION_INTERVAL,
                'stability_window_duration': config.STABILITY_WINDOW_DURATION,
                'min_completeness_score': config.MIN_COMPLETENESS_SCORE
            },
            'diagrams': self.metadata
        }
        
        with open(metadata_filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
        
        return metadata_filepath
    
    def get_statistics(self) -> Dict:
        """
        Get extraction statistics
        
        Returns:
            Dictionary with statistics
        """
        if not self.metadata:
            return {
                'total_diagrams': 0,
                'board_types': {},
                'avg_completeness_score': 0.0
            }
        
        board_types = {}
        completeness_scores = []
        
        for entry in self.metadata:
            board_type = entry.get('board_type', 'unknown')
            board_types[board_type] = board_types.get(board_type, 0) + 1
            completeness_scores.append(entry.get('completeness_score', 0.0))
        
        return {
            'total_diagrams': len(self.metadata),
            'board_types': board_types,
            'avg_completeness_score': sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0.0,
            'min_completeness_score': min(completeness_scores) if completeness_scores else 0.0,
            'max_completeness_score': max(completeness_scores) if completeness_scores else 0.0
        }
    
    def remove_diagram(self, diagram_id: int) -> None:
        """
        Remove a diagram from the pipeline: delete its file and remove from metadata.
        Used when classifier marks the image as NOT_DIAGRAM.
        """
        filepath = self.diagram_id_to_file.pop(diagram_id, None)
        if filepath and os.path.isfile(filepath):
            try:
                os.remove(filepath)
            except OSError:
                pass
        self.enhanced_diagram_paths.pop(diagram_id, None)
        self.metadata[:] = [e for e in self.metadata if e.get("diagram_id") != diagram_id]
    
    def save_enhanced_diagram(
        self,
        diagram_id: int,
        enhanced_image_bytes: bytes
    ) -> str:
        """
        Save enhanced diagram image
        
        Args:
            diagram_id: ID of the diagram
            enhanced_image_bytes: Enhanced diagram image bytes
            
        Returns:
            Path to saved enhanced diagram
        """
        filename = f"enhanced_diagram_{diagram_id:04d}.png"
        filepath = os.path.join(self.enhanced_diagrams_dir, filename)
        
        with open(filepath, 'wb') as f:
            f.write(enhanced_image_bytes)
        
        self.enhanced_diagram_paths[diagram_id] = filepath
        
        if config.VERBOSE:
            print(f"Enhanced diagram saved: {filepath}")
        
        return filepath
    
    def save_enhanced_diagram_from_array(
        self, diagram_id: int, image_array
    ) -> str:
        """
        Save an image array (e.g. cropped digital diagram) as the enhanced diagram for diagram_id.
        Used for digital images: crop only, no Gemini enhancement.
        """
        filename = f"enhanced_diagram_{diagram_id:04d}.png"
        filepath = os.path.join(self.enhanced_diagrams_dir, filename)
        cv2.imwrite(filepath, image_array)
        self.enhanced_diagram_paths[diagram_id] = filepath
        if config.VERBOSE:
            print(f"Enhanced (cropped) diagram saved: {filepath}")
        return filepath
    
    def get_enhanced_diagram_path(self, diagram_id: int) -> Optional[str]:
        """
        Get path to enhanced diagram if it exists
        
        Args:
            diagram_id: ID of the diagram
            
        Returns:
            Path to enhanced diagram or None if not found
        """
        return self.enhanced_diagram_paths.get(diagram_id)
    
    def save_transcript(self, transcript_data: Dict, filename: str = "transcript.json"):
        """
        Save transcript data to JSON file
        
        Args:
            transcript_data: Transcript data dictionary
            filename: Filename for transcript (default: transcript.json)
            
        Returns:
            Path to saved transcript file
        """
        import json
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(transcript_data, f, indent=2, ensure_ascii=False)
        
        if config.VERBOSE:
            print(f"Transcript saved: {filepath}")
        
        return filepath
    
    def save_summary(self, summary_text: str, filename: str = "summary.txt"):
        """
        Save summary text to file
        
        Args:
            summary_text: Summary text
            filename: Filename for summary (default: summary.txt)
            
        Returns:
            Path to saved summary file
        """
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        
        if config.VERBOSE:
            print(f"Summary saved: {filepath}")
        
        return filepath
    
    def load_metadata(self):
        """
        Load existing metadata from JSON file
        
        Returns:
            True if metadata was loaded successfully, False otherwise
        """
        metadata_filepath = os.path.join(self.output_dir, self.metadata_filename)
        
        if not os.path.exists(metadata_filepath):
            return False
        
        try:
            with open(metadata_filepath, 'r', encoding='utf-8') as f:
                metadata_dict = json.load(f)
            
            self.metadata = metadata_dict.get('diagrams', [])
            self.diagram_count = len(self.metadata)
            
            # Rebuild diagram_id_to_file mapping
            for entry in self.metadata:
                diagram_id = entry.get('diagram_id')
                filepath = entry.get('filepath')
                if diagram_id and filepath:
                    self.diagram_id_to_file[diagram_id] = filepath
            
            if config.VERBOSE:
                print(f"Loaded {self.diagram_count} diagrams from existing metadata")
            
            return True
        except Exception as e:
            if config.VERBOSE:
                print(f"Error loading metadata: {str(e)}")
            return False