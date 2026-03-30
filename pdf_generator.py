"""
PDF generation service
Combines transcriptions and enhanced diagrams into a PDF document
"""
import os
import json
from typing import Dict, List, Optional
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, HRFlowable
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from io import BytesIO
import config


class PDFGenerator:
    """Service for generating PDF documents with transcriptions and diagrams"""
    
    def __init__(self):
        """Initialize PDF generator"""
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        # Title style
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor='black',
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        # Heading style
        self.heading_style = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor='black',
            spaceAfter=12,
            spaceBefore=12,
            alignment=TA_LEFT,
            fontName='Helvetica-Bold'
        )
        
        # Timestamp style
        self.timestamp_style = ParagraphStyle(
            'Timestamp',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor='gray',
            spaceAfter=6,
            alignment=TA_LEFT,
            fontName='Helvetica-Oblique'
        )
        
        # Normal text style
        self.normal_style = ParagraphStyle(
            'CustomNormal',
            parent=self.styles['Normal'],
            fontSize=11,
            textColor='black',
            spaceAfter=12,
            alignment=TA_JUSTIFY,
            fontName='Helvetica'
        )
        
        # Small caption/footnote style
        self.caption_style = ParagraphStyle(
            'Caption',
            parent=self.styles['Normal'],
            fontSize=9,
            textColor='gray',
            spaceAfter=8,
            alignment=TA_CENTER,
            fontName='Helvetica-Oblique'
        )
        
        # Subheading style for summary sections
        self.subheading_style = ParagraphStyle(
            'Subheading',
            parent=self.styles['Heading3'],
            fontSize=14,
            textColor='black',
            spaceAfter=10,
            spaceBefore=16,
            alignment=TA_LEFT,
            fontName='Helvetica-Bold'
        )
    
    def _add_page_number(self, canv: canvas.Canvas, doc):
        """Draw page number in footer on each page."""
        page_num = canv.getPageNumber()
        text = f"Page {page_num}"
        canv.setFont("Helvetica", 9)
        canv.setFillColorRGB(0.5, 0.5, 0.5)
        width, height = A4
        canv.drawRightString(width - 0.75 * inch, 0.5 * inch, text)
    
    def format_timestamp(self, seconds: float) -> str:
        """
        Format timestamp in seconds to readable format
        
        Args:
            seconds: Timestamp in seconds
            
        Returns:
            Formatted timestamp string (MM:SS or HH:MM:SS)
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"
    
    def _parse_markdown_summary(self, summary_text: str) -> List:
        """
        Parse markdown-formatted summary text and convert to PDF elements
        
        Args:
            summary_text: Summary text with markdown formatting
            
        Returns:
            List of PDF elements (Paragraphs, Spacers, HRFlowables)
        """
        import re
        elements = []
        lines = summary_text.split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines
            if not line:
                i += 1
                continue
            
            # Remove leading "Here is a comprehensive summary..." if present
            if i == 0 and ("Here is a comprehensive summary" in line or "Here is" in line.lower()):
                # Skip this line
                i += 1
                continue
            
            # Handle horizontal rules (---)
            if re.match(r'^---+$', line):
                elements.append(HRFlowable(width="100%", thickness=0.5, color="#CCCCCC", spaceBefore=8, spaceAfter=8))
                i += 1
                continue
            
            # Handle headings (### Title or ## Title)
            heading_match = re.match(r'^#{1,3}\s+(.+)$', line)
            if heading_match:
                heading_text = heading_match.group(1).strip()
                # Remove bold markers if present
                heading_text = re.sub(r'\*\*(.+?)\*\*', r'\1', heading_text)
                # Escape for ReportLab
                heading_text = heading_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                elements.append(Paragraph(heading_text, self.subheading_style))
                i += 1
                continue
            
            # Handle regular text - convert bold (**text**) to <b>text</b>
            # and clean up any remaining markdown
            text = line
            # Convert **bold** to <b>bold</b>
            text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
            # Remove any remaining markdown markers
            text = re.sub(r'^#+\s*', '', text)  # Remove leading # markers
            text = text.replace('---', '')  # Remove horizontal rule markers
            
            # Escape special characters for ReportLab
            text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            # But restore <b> tags
            text = text.replace('&lt;b&gt;', '<b>').replace('&lt;/b&gt;', '</b>')
            
            if text.strip():
                elements.append(Paragraph(text, self.normal_style))
            
            i += 1
        
        return elements
    
    def generate_pdf(
        self,
        output_path: str,
        transcript_data: Dict,
        diagram_metadata: List[Dict],
        enhanced_diagram_paths: Dict[int, str],
        summary: Optional[str] = None,
        video_name: Optional[str] = None
    ):
        """
        Generate PDF with transcriptions and enhanced diagrams
        
        Args:
            output_path: Path to save PDF file
            transcript_data: Transcript data dictionary with segments
            diagram_metadata: List of diagram metadata dictionaries
            enhanced_diagram_paths: Dictionary mapping diagram_id to enhanced diagram path
            summary: Optional summary text
            video_name: Optional video name for title
        """
        if config.VERBOSE:
            print(f"Generating PDF: {output_path}")
        
        # Create PDF document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch
        )
        
        # Build story (content)
        story = []
        
        # Title page
        title = video_name if video_name else "Lecture Transcript"
        story.append(Paragraph(title, self.title_style))
        story.append(Spacer(1, 0.5*inch))
        story.append(Paragraph("Transcript with Diagrams", self.styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph(
            "Full transcript with diagrams placed at the timestamps where they appear in the lecture.",
            self.normal_style,
        ))
        story.append(PageBreak())
        
        if summary:
            story.append(Paragraph("Summary", self.heading_style))
            story.append(Spacer(1, 0.15*inch))
            # Parse markdown and add formatted elements
            summary_elements = self._parse_markdown_summary(summary)
            story.extend(summary_elements)
            story.append(PageBreak())
        
        # Create a mapping of timestamp to diagram for quick lookup
        diagram_by_timestamp = {}
        for diagram in diagram_metadata:
            timestamp = diagram.get('timestamp', 0)
            diagram_id = diagram.get('diagram_id')
            if diagram_id and diagram_id in enhanced_diagram_paths:
                if timestamp not in diagram_by_timestamp:
                    diagram_by_timestamp[timestamp] = []
                diagram_by_timestamp[timestamp].append({
                    'diagram_id': diagram_id,
                    'path': enhanced_diagram_paths[diagram_id],
                    'metadata': diagram
                })
        
        # Process transcript segments
        segments = transcript_data.get('segments', [])
        current_diagram_index = 0
        
        story.append(Paragraph("Transcript", self.heading_style))
        story.append(Spacer(1, 0.15*inch))
        
        for segment in segments:
            seg_start = segment.get('start', 0)
            seg_end = segment.get('end', 0)
            seg_text = segment.get('text', '').strip()
            
            if not seg_text:
                continue
            
            # Check if there's a diagram at this timestamp
            # Check if any diagram timestamp is within this segment
            diagram_to_insert = None
            for diagram_timestamp, diagrams in diagram_by_timestamp.items():
                if seg_start <= diagram_timestamp <= seg_end:
                    # Use the first diagram found
                    if diagrams:
                        diagram_to_insert = diagrams[0]
                        # Remove from dict to avoid duplicate insertion
                        diagram_by_timestamp.pop(diagram_timestamp, None)
                        break
            
            # Add timestamp
            timestamp_str = self.format_timestamp(seg_start)
            story.append(Paragraph(f"[{timestamp_str}]", self.timestamp_style))
            
            # Add transcript text
            # Escape special characters for ReportLab
            safe_text = seg_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            story.append(Paragraph(safe_text, self.normal_style))
            
            # Insert diagram if available
            if diagram_to_insert:
                diagram_path = diagram_to_insert['path']
                if os.path.exists(diagram_path):
                    try:
                        # Validate image before adding
                        from PIL import Image as PILImage
                        try:
                            pil_img = PILImage.open(diagram_path)
                            pil_img.verify()
                            # Reopen after verify (verify closes the file)
                            pil_img = PILImage.open(diagram_path)
                            img_width, img_height = pil_img.size
                            
                            # Check if image is valid
                            if img_width > 0 and img_height > 0:
                                aspect_ratio = img_height / img_width if img_width > 0 else 1
                                
                                # Set max width and calculate height
                                max_width = 5*inch
                                calculated_height = max_width * aspect_ratio
                                max_height = 4*inch
                                
                                if calculated_height > max_height:
                                    calculated_height = max_height
                                    max_width = calculated_height / aspect_ratio if aspect_ratio > 0 else max_width
                                
                                img = Image(diagram_path, width=max_width, height=calculated_height)
                                story.append(Spacer(1, 0.1*inch))
                                story.append(img)
                                
                                # Add diagram caption
                                diagram_meta = diagram_to_insert['metadata']
                                caption = f"Diagram at {self.format_timestamp(diagram_meta.get('timestamp', 0))}"
                                story.append(Paragraph(caption, self.caption_style))
                                story.append(Spacer(1, 0.25*inch))
                            else:
                                if config.VERBOSE:
                                    print(f"Warning: Diagram {diagram_to_insert['diagram_id']} has invalid dimensions")
                        except Exception as img_error:
                            if config.VERBOSE:
                                print(f"Warning: Could not validate diagram {diagram_to_insert['diagram_id']}: {str(img_error)}")
                            # Try to use original diagram as fallback
                            diagram_meta = diagram_to_insert['metadata']
                            original_path = diagram_meta.get('filepath')
                            if original_path and os.path.exists(original_path):
                                try:
                                    pil_img = PILImage.open(original_path)
                                    pil_img.verify()
                                    pil_img = PILImage.open(original_path)
                                    img_width, img_height = pil_img.size
                                    if img_width > 0 and img_height > 0:
                                        aspect_ratio = img_height / img_width
                                        max_width = 5*inch
                                        calculated_height = max_width * aspect_ratio
                                        max_height = 4*inch
                                        if calculated_height > max_height:
                                            calculated_height = max_height
                                            max_width = calculated_height / aspect_ratio if aspect_ratio > 0 else max_width
                                        img = Image(original_path, width=max_width, height=calculated_height)
                                        story.append(Spacer(1, 0.1*inch))
                                        story.append(img)
                                        caption = f"Diagram at {self.format_timestamp(diagram_meta.get('timestamp', 0))} (original)"
                                        story.append(Paragraph(caption, self.caption_style))
                                        story.append(Spacer(1, 0.25*inch))
                                except:
                                    if config.VERBOSE:
                                        print(f"Warning: Could not use original diagram {diagram_to_insert['diagram_id']} either")
                    except Exception as e:
                        if config.VERBOSE:
                            print(f"Warning: Could not insert diagram {diagram_to_insert['diagram_id']}: {str(e)}")
            
            # Add subtle divider between segments for visual clarity
            story.append(Spacer(1, 0.08*inch))
            story.append(HRFlowable(width="100%", thickness=0.3, color="#DDDDDD", spaceBefore=4, spaceAfter=6))
        
        # Insert any remaining diagrams that weren't matched to segments
        for diagram_timestamp, diagrams in diagram_by_timestamp.items():
            for diagram_info in diagrams:
                diagram_path = diagram_info['path']
                if os.path.exists(diagram_path):
                    try:
                        story.append(Spacer(1, 0.3*inch))
                        story.append(Paragraph("Additional Diagram", self.heading_style))
                        img = Image(diagram_path, width=5*inch, height=3.75*inch)
                        story.append(Spacer(1, 0.1*inch))
                        story.append(img)
                        
                        diagram_meta = diagram_info['metadata']
                        caption = f"Diagram at {self.format_timestamp(diagram_meta.get('timestamp', 0))}"
                        story.append(Paragraph(caption, self.timestamp_style))
                    except Exception as e:
                        if config.VERBOSE:
                            print(f"Warning: Could not insert remaining diagram {diagram_info['diagram_id']}: {str(e)}")
        
        # Build PDF
        try:
            doc.build(
                story,
                onFirstPage=self._add_page_number,
                onLaterPages=self._add_page_number,
            )
            if config.VERBOSE:
                print(f"PDF generated successfully: {output_path}")
        except Exception as e:
            raise RuntimeError(f"PDF generation failed: {str(e)}")

    def generate_explanation_pdf(
        self,
        output_path: str,
        explanation_text: str,
        diagram_metadata: List[Dict],
        enhanced_diagram_paths: Dict[int, str],
        video_name: Optional[str] = None
    ):
        """
        Generate PDF from detailed explanation text. Replaces [DIAGRAM:N] markers
        with the corresponding diagram image (N = 1-based index by order of appearance).
        """
        import re
        if config.VERBOSE:
            print(f"Generating explanation PDF: {output_path}")

        # Order diagrams by timestamp so index 1 = first diagram, etc.
        ordered = sorted(diagram_metadata, key=lambda d: d.get("timestamp", 0))
        diagram_index_to_info = {}
        for i, d in enumerate(ordered, start=1):
            did = d.get("diagram_id")
            if did and did in enhanced_diagram_paths:
                diagram_index_to_info[i] = {
                    "path": enhanced_diagram_paths[did],
                    "metadata": d,
                }

        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch,
        )
        story = []

        # Title
        title = (video_name if video_name else "Lecture") + " — Detailed Explanation"
        story.append(Paragraph(title, self.title_style))
        story.append(Spacer(1, 0.3 * inch))
        story.append(Paragraph(
            "A detailed explanation of the lecture with diagrams placed where they are referenced.",
            self.normal_style,
        ))
        story.append(PageBreak())

        # Split by [DIAGRAM:N] and build (text, diagram_index or None)
        pattern = re.compile(r"\[DIAGRAM:(\d+)\]")
        parts = pattern.split(explanation_text)
        # parts: [text_before, N1, text_after_N1, N2, ...] so odd indices are diagram numbers
        segs = []
        i = 0
        while i < len(parts):
            if i % 2 == 0 and parts[i].strip():
                segs.append(("text", parts[i]))
            elif i % 2 == 1:
                try:
                    n = int(parts[i])
                    segs.append(("diagram", n))
                except ValueError:
                    pass
            i += 1

        for kind, value in segs:
            if kind == "text":
                # Render markdown-like text (headings, paragraphs)
                for el in self._parse_markdown_summary(value):
                    story.append(el)
            else:
                # Insert diagram
                info = diagram_index_to_info.get(value)
                if not info or not os.path.exists(info["path"]):
                    continue
                path = info["path"]
                try:
                    from PIL import Image as PILImage
                    pil_img = PILImage.open(path)
                    pil_img.verify()
                    pil_img = PILImage.open(path)
                    w, h = pil_img.size
                    if w <= 0 or h <= 0:
                        continue
                    aspect = h / w if w else 1
                    max_w = 5 * inch
                    max_h = 4 * inch
                    calc_h = max_w * aspect
                    if calc_h > max_h:
                        calc_h = max_h
                        max_w = calc_h / aspect if aspect else max_w
                    img = Image(path, width=max_w, height=calc_h)
                    story.append(Spacer(1, 0.15 * inch))
                    story.append(img)
                    ts = info["metadata"].get("timestamp", 0)
                    story.append(Paragraph(
                        f"Figure at {self.format_timestamp(ts)}",
                        self.caption_style,
                    ))
                    story.append(Spacer(1, 0.2 * inch))
                except Exception as e:
                    if config.VERBOSE:
                        print(f"Warning: Could not insert diagram {value}: {e}")

        try:
            doc.build(
                story,
                onFirstPage=self._add_page_number,
                onLaterPages=self._add_page_number,
            )
            if config.VERBOSE:
                print(f"Explanation PDF generated: {output_path}")
        except Exception as e:
            raise RuntimeError(f"Explanation PDF generation failed: {str(e)}")
    
    def generate_pdf_from_files(
        self,
        output_path: str,
        transcript_json_path: str,
        diagram_metadata_json_path: str,
        enhanced_diagrams_dir: str,
        summary_path: Optional[str] = None,
        video_name: Optional[str] = None
    ):
        """
        Generate PDF from saved files
        
        Args:
            output_path: Path to save PDF file
            transcript_json_path: Path to transcript JSON file
            diagram_metadata_json_path: Path to diagram metadata JSON file
            enhanced_diagrams_dir: Directory containing enhanced diagrams
            summary_path: Optional path to summary text file
            video_name: Optional video name for title
        """
        # Load transcript
        with open(transcript_json_path, 'r', encoding='utf-8') as f:
            transcript_data = json.load(f)
        
        # Load diagram metadata
        with open(diagram_metadata_json_path, 'r', encoding='utf-8') as f:
            metadata_dict = json.load(f)
            diagram_metadata = metadata_dict.get('diagrams', [])
        
        # Build enhanced diagram paths dictionary
        # First try enhanced diagrams, fallback to original diagrams
        from PIL import Image
        enhanced_diagram_paths = {}
        for diagram in diagram_metadata:
            diagram_id = diagram.get('diagram_id')
            if diagram_id:
                # Look for enhanced diagram file first
                enhanced_filename = f"enhanced_diagram_{diagram_id:04d}.png"
                enhanced_path = os.path.join(enhanced_diagrams_dir, enhanced_filename)
                if os.path.exists(enhanced_path):
                    # Validate the enhanced diagram
                    try:
                        with Image.open(enhanced_path) as img:
                            img.verify()
                        # Reopen after verify
                        with Image.open(enhanced_path) as img:
                            if img.size[0] > 0 and img.size[1] > 0:
                                enhanced_diagram_paths[diagram_id] = enhanced_path
                            else:
                                # Invalid dimensions, use original
                                original_path = diagram.get('filepath')
                                if original_path and os.path.exists(original_path):
                                    enhanced_diagram_paths[diagram_id] = original_path
                    except Exception:
                        # Enhanced diagram is corrupted, use original
                        original_path = diagram.get('filepath')
                        if original_path and os.path.exists(original_path):
                            enhanced_diagram_paths[diagram_id] = original_path
                else:
                    # Fallback to original diagram
                    original_path = diagram.get('filepath')
                    if original_path and os.path.exists(original_path):
                        enhanced_diagram_paths[diagram_id] = original_path
        
        # Load summary if provided
        summary = None
        if summary_path and os.path.exists(summary_path):
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary = f.read()
        
        # Generate PDF
        self.generate_pdf(
            output_path,
            transcript_data,
            diagram_metadata,
            enhanced_diagram_paths,
            summary,
            video_name
        )
