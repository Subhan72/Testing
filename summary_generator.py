"""
Summary generation service using Google Gemini API
Generates summaries from transcriptions
"""
import os
import time
from typing import Dict, Optional, List
try:
    from google.genai import Client as _GenaiClient
except ImportError as e:
    raise ImportError(
        "Failed to import Google GenAI SDK. Install the new SDK: pip install google-genai\n"
        "If you have the old SDK, uninstall it first: pip uninstall google-generativeai"
    ) from e
from api_config import APIConfig
from transcription_service import TranscriptionService
import config


class SummaryGenerator:
    """Service for generating summaries from transcriptions using Gemini API"""
    
    def __init__(self, transcription_service: Optional[TranscriptionService] = None):
        """
        Initialize summary generator
        
        Args:
            transcription_service: Optional TranscriptionService instance for getting transcripts
        """
        self.transcription_service = transcription_service
        self.summaries = {}
        
        # Initialize Gemini client (new google.genai client, consistent with diagram_enhancer)
        try:
            api_key = APIConfig.get_google_api_key()
            self.client = _GenaiClient(api_key=api_key)
            # Use gemini-3-pro-preview for text generation (summaries)
            self.model_name = "gemini-3-pro-preview"
        except ValueError as e:
            if config.VERBOSE:
                print(f"Warning: {e}")
            self.client = None
    
    def generate_summary(
        self,
        transcript_text: Optional[str] = None,
        max_length: Optional[int] = None
    ) -> str:
        """
        Generate overall summary from transcript
        
        Args:
            transcript_text: Transcript text (uses transcription service if None)
            max_length: Maximum summary length in words (None for auto)
            
        Returns:
            Generated summary text
        """
        if self.client is None:
            raise RuntimeError("Gemini client not initialized. Check Google API key.")
        
        # Get transcript if not provided
        if transcript_text is None:
            if not self.transcription_service:
                raise ValueError("No transcript text provided and no transcription service available.")
            transcript_text = self.transcription_service.get_full_transcript(use_english=True)
        
        if not transcript_text or not transcript_text.strip():
            return "No transcript available for summary generation."
        
        if config.VERBOSE:
            print("Generating summary from transcript...")
        
        try:
            # Construct prompt
            length_instruction = ""
            if max_length:
                length_instruction = f" The summary should be approximately {max_length} words."
            
            prompt = f"""You are an expert at summarizing educational lecture content. Please provide a comprehensive, well-structured summary of the following lecture transcript.

**Instructions:**
- Create a clear, well-structured summary
- Highlight key concepts, main topics, and important points
- Organize the summary logically
- Use clear, concise language{length_instruction}
- Focus on educational content and learning outcomes

**Transcript:**
{transcript_text}

**Summary:**"""

            # Generate summary using new google.genai client
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt],
                config={
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 2048,
                },
            )
            
            summary_text = (getattr(response, "text", None) or "").strip()
            
            if config.VERBOSE:
                print("Summary generation complete.")
            
            return summary_text
            
        except Exception as e:
            raise RuntimeError(f"Summary generation failed: {str(e)}")

    def generate_lecture_title(
        self,
        transcript_text: Optional[str] = None,
    ) -> str:
        """
        Generate a short, descriptive title for the lecture from its transcript.
        Suitable for document headers (HTML/DOCX). Returns a single line; falls back to empty string on failure.
        """
        if self.client is None:
            return ""
        if transcript_text is None:
            if not self.transcription_service:
                return ""
            transcript_text = self.transcription_service.get_full_transcript(use_english=True)
        if not transcript_text or not transcript_text.strip():
            return ""
        # Use first ~8k chars to avoid token limits; enough for title
        excerpt = transcript_text[:8000].strip()
        prompt = f"""Based on the following lecture transcript excerpt, produce a single, complete title for this lecture.

**Rules:**
- Output exactly one line: the full title. Do not abbreviate, truncate, or shorten. Capture the complete topic (e.g. "Introduction to Machine Learning: Supervised vs Unsupervised Learning" not "Introduction to Machine Learning" if the lecture covers both).
- Use title case. Be clear and professional. No trailing period, colon, or quotes.
- Do not output "Title:" or any prefix—only the title text itself.
- If the lecture has a clear main topic and subtopic, include both in one line. The title should allow a reader to identify the lecture content at a glance.

**Transcript excerpt:**
{excerpt}

**Title:**"""
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt],
                config={
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "top_k": 40,
                    "max_output_tokens": 256,
                },
            )
            title = (getattr(response, "text", None) or "").strip()
            # Clean common artifacts
            for prefix in ("Title:", "title:", '"', "'"):
                if title.startswith(prefix):
                    title = title[len(prefix):].strip()
            if title.endswith('"') or title.endswith("'"):
                title = title[:-1].strip()
            return title[:200] if title else ""
        except Exception:
            return ""
    
    def generate_section_summaries(
        self,
        section_duration: float = 600.0,
        transcript_data: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Generate summaries for different sections of the transcript
        
        Args:
            section_duration: Duration of each section in seconds (default: 600 = 10 minutes)
            transcript_data: Transcript data dictionary (uses transcription service if None)
            
        Returns:
            List of dictionaries with section summaries, each containing:
            - start_time: Section start time
            - end_time: Section end time
            - summary: Generated summary
        """
        if self.client is None:
            raise RuntimeError("Gemini client not initialized. Check Google API key.")
        
        # Get transcript data
        if transcript_data is None:
            if not self.transcription_service:
                raise ValueError("No transcript data provided and no transcription service available.")
            transcript_data = (
                self.transcription_service.transcript_english 
                if self.transcription_service.transcript_english 
                else self.transcription_service.transcript_data
            )
        
        if not transcript_data:
            return []
        
        total_duration = transcript_data.get('duration', 0)
        sections = []
        
        if config.VERBOSE:
            print(f"Generating section summaries (every {section_duration:.0f} seconds)...")
        
        current_time = 0.0
        section_number = 1
        
        while current_time < total_duration:
            section_start = current_time
            section_end = min(current_time + section_duration, total_duration)
            
            # Get transcript segment for this section
            if self.transcription_service:
                section_text = self.transcription_service.get_transcript_segment(
                    section_start, section_end, use_english=True
                )
            else:
                # Extract from transcript_data directly
                segments = transcript_data.get('segments', [])
                section_text_parts = []
                for segment in segments:
                    seg_start = segment.get('start', 0)
                    seg_end = segment.get('end', 0)
                    if seg_start < section_end and seg_end > section_start:
                        section_text_parts.append(segment.get('text', ''))
                section_text = ' '.join(section_text_parts).strip()
            
            if section_text:
                try:
                    # Format time
                    start_min = int(section_start // 60)
                    start_sec = int(section_start % 60)
                    end_min = int(section_end // 60)
                    end_sec = int(section_end % 60)
                    
                    prompt = f"""You are an expert at summarizing educational lecture content. Please provide a concise summary of the following section from a lecture transcript.

**Section:** {start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}

**Transcript Section:**
{section_text}

**Summary:**"""

                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=[prompt],
                        config={
                            "temperature": 0.7,
                            "top_p": 0.95,
                            "top_k": 40,
                            "max_output_tokens": 512,
                        },
                    )
                    
                    summary_text = (getattr(response, "text", None) or "").strip()
                    
                    sections.append({
                        'section_number': section_number,
                        'start_time': section_start,
                        'end_time': section_end,
                        'start_time_formatted': f"{start_min:02d}:{start_sec:02d}",
                        'end_time_formatted': f"{end_min:02d}:{end_sec:02d}",
                        'summary': summary_text
                    })
                    
                    if config.VERBOSE:
                        print(f"  Section {section_number} summary generated ({start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d})")
                    
                except Exception as e:
                    if config.VERBOSE:
                        print(f"  Warning: Failed to generate summary for section {section_number}: {str(e)}")
            
            current_time += section_duration
            section_number += 1
        
        if config.VERBOSE:
            print(f"Generated {len(sections)} section summaries.")
        
        return sections
    
    def generate_detailed_explanation(
        self,
        transcript_text: Optional[str] = None,
        diagram_metadata: Optional[List[Dict]] = None
    ) -> str:
        """
        Generate a detailed explanation of the lecture. Use [DIAGRAM:N] in the text
        exactly where diagram N (by order of appearance, 1-based) should be inserted.

        Args:
            transcript_text: Full transcript (uses transcription service if None)
            diagram_metadata: List of {diagram_id, timestamp, ...} in order of appearance

        Returns:
            Explanation text with [DIAGRAM:1], [DIAGRAM:2], ... markers.
        """
        if self.client is None:
            raise RuntimeError("Gemini client not initialized. Check Google API key.")

        if transcript_text is None:
            if not self.transcription_service:
                raise ValueError("No transcript text and no transcription service.")
            transcript_text = self.transcription_service.get_full_transcript(use_english=True)

        if not transcript_text or not transcript_text.strip():
            return "No transcript available for explanation."

        # Build diagram list for the prompt (ordered by timestamp)
        diagram_list = []
        if diagram_metadata:
            sorted_diagrams = sorted(diagram_metadata, key=lambda d: d.get("timestamp", 0))
            for i, d in enumerate(sorted_diagrams, start=1):
                ts = d.get("timestamp", 0)
                m = int(ts // 60)
                s = int(ts % 60)
                diagram_list.append(f"  Diagram {i}: at {m:02d}:{s:02d} (use marker [DIAGRAM:{i}] where this figure belongs)")

        diagram_instructions = "\n".join(diagram_list) if diagram_list else "  (No diagrams; do not use [DIAGRAM:N] markers.)"

        prompt = f"""You are an expert educator. Write a detailed, well-structured explanation of the following lecture based ONLY on the transcript below. The output will be used for RAG and study notes. Use clear heading hierarchy (## for main sections, ### for subsections). Start directly with the content—no introductory meta-summary.

**Content rules:**
- Use ONLY information explicitly stated in the transcript. Do not add facts, examples, citations, or details that are not in the transcript. Do not hallucinate or infer beyond what is clearly stated.
- Write in clear, formal English. Structure with ## and ### headings for navigation and RAG chunking.
- If the transcript is unclear or sparse on a topic, summarize only what is said; do not fill in with external knowledge.

**Formulas, equations, and mathematical notation (critical):**
- When the transcript contains equations, formulas, mathematical expressions, chemical equations, or any formal notation, reproduce them EXACTLY as they appear (or as literally as the spoken form allows). Do not paraphrase, simplify, or "correct" them.
- Preserve all notation: variables, subscripts, superscripts, Greek letters (e.g. α, β, Σ, π), operators (×, ÷, ∫, √, ∑, ∏), and structure. Do not substitute symbols or rewrite expressions unless the transcript explicitly corrects itself.
- If the speaker reads an equation verbatim, use that exact wording or standard notation that matches it. When in doubt, keep the transcript's wording rather than inventing or "improving" it.
- This applies to all formula types: math, physics, chemistry, statistics, code snippets, or any technical notation.

**Diagram markers:**
- Where a diagram or figure from the lecture is relevant, insert exactly the marker [DIAGRAM:N] at that position. N is the diagram number (1-based, in order of appearance). Use each diagram marker exactly once where the surrounding text refers to that figure.

**Diagram markers to use (insert at the exact place each figure should appear):**
{diagram_instructions}

**Lecture transcript (your only source of content):**
{transcript_text[:120000]}

**Detailed explanation (preserve all formulas exactly; use [DIAGRAM:N] where each figure belongs; use only transcript content):**"""

        max_retries = 5
        base_delay = 2.0
        last_error = None
        for attempt in range(max_retries):
            try:
                if config.VERBOSE:
                    print("Generating detailed explanation...")
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[prompt],
                    config={
                        "temperature": 0.5,
                        "top_p": 0.95,
                        "top_k": 40,
                        "max_output_tokens": 8192,
                    },
                )
                explanation = (getattr(response, "text", None) or "").strip()
                if config.VERBOSE:
                    print("Detailed explanation generated.")
                return explanation
            except Exception as e:
                last_error = e
                err_str = str(e).upper()
                # Retry on transient errors (503, 429, 500, timeout, unavailable)
                if attempt < max_retries - 1 and (
                    "503" in err_str or "429" in err_str or "500" in err_str
                    or "UNAVAILABLE" in err_str or "RESOURCE_EXHAUSTED" in err_str
                    or "TIMEOUT" in err_str or "OVERLOADED" in err_str
                ):
                    delay = base_delay * (2 ** attempt)
                    if config.VERBOSE:
                        print(f"  Retry {attempt + 1}/{max_retries - 1} after {delay:.1f}s: {e}")
                    time.sleep(delay)
                else:
                    raise RuntimeError(f"Detailed explanation generation failed: {str(e)}") from last_error
        raise RuntimeError(f"Detailed explanation generation failed: {str(last_error)}") from last_error

    def save_summary(self, summary_text: str, output_path: str):
        """
        Save summary to text file
        
        Args:
            summary_text: Summary text to save
            output_path: Path to save summary file
        """
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        
        if config.VERBOSE:
            print(f"Summary saved to: {output_path}")
    
    def save_section_summaries(self, sections: List[Dict], output_path: str):
        """
        Save section summaries to text file
        
        Args:
            sections: List of section summary dictionaries
            output_path: Path to save summaries file
        """
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("Section Summaries\n")
            f.write("=" * 60 + "\n\n")
            
            for section in sections:
                f.write(f"Section {section['section_number']}: ")
                f.write(f"{section['start_time_formatted']} - {section['end_time_formatted']}\n")
                f.write("-" * 60 + "\n")
                f.write(f"{section['summary']}\n\n")
        
        if config.VERBOSE:
            print(f"Section summaries saved to: {output_path}")
