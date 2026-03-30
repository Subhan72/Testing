"""
Diagram completeness checker
Validates if a detected diagram is complete
"""
from typing import Dict, Tuple
import config


class CompletenessChecker:
    """Checks if a diagram is complete"""
    
    def __init__(self):
        """Initialize completeness checker"""
        self.min_completeness_score = config.MIN_COMPLETENESS_SCORE
        self.require_stability = config.REQUIRE_STABILITY
        self.require_structure = config.REQUIRE_STRUCTURE
        self.min_content_coverage = config.MIN_CONTENT_COVERAGE
    
    def check_completeness(
        self,
        stability_score: float,
        is_stable: bool,
        content_analysis: Dict,
        board_region: any,
        board_type: str
    ) -> Tuple[bool, float, Dict]:
        """
        Check if diagram is complete
        
        Args:
            stability_score: Temporal stability score
            is_stable: Whether content is stable
            content_analysis: Content analysis results
            board_region: Board region (for coverage calculation)
            board_type: Type of board
            
        Returns:
            Tuple of (is_complete, completeness_score, reasons)
        """
        reasons = {}
        score_components = []
        
        # Stability check
        if self.require_stability:
            if is_stable:
                stability_component = stability_score
                reasons['stability'] = f"Stable (score: {stability_score:.3f})"
            else:
                stability_component = 0.0
                reasons['stability'] = "Not stable"
            score_components.append(stability_component * 0.4)
        else:
            stability_component = stability_score
            score_components.append(stability_component * 0.2)
        
        # Structure check
        if self.require_structure:
            if content_analysis.get('is_diagram_like', False):
                structure_component = content_analysis.get('diagram_score', 0.0)
                reasons['structure'] = f"Diagram-like structure (score: {structure_component:.3f})"
            else:
                structure_component = 0.0
                reasons['structure'] = "Not diagram-like"
            score_components.append(structure_component * 0.4)
        else:
            structure_component = content_analysis.get('diagram_score', 0.0)
            score_components.append(structure_component * 0.3)
        
        # Content coverage check
        coverage_score = self._check_content_coverage(board_region, content_analysis)
        reasons['coverage'] = f"Content coverage: {coverage_score:.3f}"
        score_components.append(coverage_score * 0.2)
        
        # Calculate overall completeness score
        completeness_score = sum(score_components)
        
        # Determine if complete
        is_complete = completeness_score >= self.min_completeness_score
        
        # Additional validation: must have minimum structure even if not required
        if content_analysis.get('diagram_score', 0.0) < 0.3:
            is_complete = False
            reasons['validation'] = "Diagram score too low"
        
        reasons['overall_score'] = completeness_score
        reasons['is_complete'] = is_complete
        
        return is_complete, completeness_score, reasons
    
    def _check_content_coverage(self, board_region: any, content_analysis: Dict) -> float:
        """
        Check content coverage of board area
        
        Args:
            board_region: Board region
            content_analysis: Content analysis results
            
        Returns:
            Coverage score (0-1)
        """
        # Use edge density as proxy for content coverage
        edge_density = content_analysis.get('edge_density', 0.0)
        
        # Normalize to coverage score
        coverage = min(edge_density / self.min_content_coverage, 1.0) if self.min_content_coverage > 0 else 1.0
        
        return coverage
    
    def validate_diagram(self, analysis_results: Dict) -> bool:
        """
        Final validation of diagram
        
        Args:
            analysis_results: Combined analysis results
            
        Returns:
            True if valid diagram
        """
        # Must have minimum edge density
        if analysis_results.get('edge_density', 0.0) < config.MIN_EDGE_DENSITY:
            return False
        
        # Must have minimum contours
        if analysis_results.get('contour_count', 0) < config.MIN_CONTOUR_COUNT:
            return False
        
        # Must have diagram-like structure
        if not analysis_results.get('is_diagram_like', False):
            return False
        
        return True
