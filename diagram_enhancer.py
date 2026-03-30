"""
Diagram enhancement service using Google Gemini Nano Banana Pro
Enhances extracted diagrams using transcript context
"""
import os
import tempfile
from io import BytesIO
from typing import Dict, Optional, Tuple
try:
    from google.genai import Client as _GenaiClient
    from google.genai import types as _GenaiTypes
except ImportError as e:
    raise ImportError(
        "Failed to import Google GenAI SDK. Install the new SDK: pip install google-genai\n"
        "If you have the old SDK, uninstall it first: pip uninstall google-generativeai"
    ) from e
from PIL import Image
from api_config import APIConfig
from transcription_service import TranscriptionService
import config


class DiagramEnhancer:
    """Service for enhancing diagrams using Gemini API with transcript context"""
    
    def __init__(self, transcription_service: Optional[TranscriptionService] = None):
        """
        Initialize diagram enhancer
        
        Args:
            transcription_service: Optional TranscriptionService instance for getting transcript context
        """
        self.transcription_service = transcription_service
        self.enhanced_diagrams = []
        
        # Initialize Gemini client using new API
        try:
            api_key = APIConfig.get_google_api_key()
            self.client = _GenaiClient(api_key=api_key)
            self.model_name = "gemini-3-pro-image-preview"
        except ValueError as e:
            if config.VERBOSE:
                print(f"Warning: {e}")
            self.client = None
            self.model_name = None
    
    def get_transcript_context(
        self,
        timestamp: float,
        context_window: float = 120.0
    ) -> Tuple[str, str, str]:
        """
        Get transcript context around a specific timestamp
        
        Args:
            timestamp: Diagram timestamp in seconds
            context_window: Context window in seconds (default: 120 = 2 minutes)
            
        Returns:
            Tuple of (prior_context, current_context, next_context)
        """
        if not self.transcription_service:
            return ("", "", "")
        
        # Calculate time ranges (2 minutes before and after)
        prior_start = max(0, timestamp - context_window)
        prior_end = timestamp
        current_start = timestamp
        current_end = timestamp + 1  # Current moment (1 second window)
        next_start = timestamp
        next_end = timestamp + context_window
        
        # Get transcript segments
        prior_context = self.transcription_service.get_transcript_segment(
            prior_start, prior_end, use_english=True
        )
        current_context = self.transcription_service.get_transcript_segment(
            current_start, current_end, use_english=True
        )
        next_context = self.transcription_service.get_transcript_segment(
            next_start, next_end, use_english=True
        )
        
        return (prior_context, current_context, next_context)
    
    def _build_enhancement_prompt(
        self,
        prior_context: str,
        current_context: str,
        next_context: str,
        timestamp_str: str,
        timestamp: float,
    ) -> str:
        """Build prompt for digitalizing hand-drawn diagrams only. Output is fully digital, not on board background."""
        return f"""You are a diagram digitalizer. The input image is a hand-drawn diagram from a lecture (on a whiteboard, blackboard, or similar). Your job is to output exactly one image: a fully digitalized version of this diagram.

**INPUT:** One hand-drawn diagram at {timestamp_str} ({timestamp:.2f}s).

**TRANSCRIPT (for context only—do not add content that is not in the image or transcript):**
Before: {prior_context if prior_context.strip() else "None."}
Now: {current_context if current_context.strip() else "None."}
After: {next_context if next_context.strip() else "None."}

**TASK — Fully digitalize (do not keep the board):**
• Output a fully digital diagram. Remove the whiteboard/blackboard background entirely. The result must look like a digital graphic: clean background (e.g. plain white or light), crisp lines and shapes, as if created in a drawing app—not drawn on a board.
• Redraw only what is present in the input image. Do not add anything new: no extra labels, shapes, arrows, or text unless they are clearly in the original image or explicitly stated in the transcript. Do not hallucinate or invent content.
• Preserve the structure, layout, and meaning of the diagram. Improve clarity: sharpen lines, fix proportions, make text readable. Do not change or add substance.
• Remove any humans, hands, or body parts; output only the diagram on a clean digital background.
• Output exactly one image. No explanation, no extra text outside the diagram.

**RULES:**
• Do not hallucinate: every element in the output must correspond to something in the original image or be explicitly in the transcript. When in doubt, omit rather than invent.
• Fully digital = no board texture or board background; use a plain, digital-style background.
• Return that one image and nothing else."""

    def enhance_diagram(
        self,
        diagram_path: str,
        timestamp: float,
        context_window: float = 120.0
    ) -> Optional[bytes]:
        """
        Enhance diagram using Gemini Nano Banana Pro with transcript context
        
        Args:
            diagram_path: Path to original diagram image file (INPUT)
            timestamp: Timestamp when diagram appears in video
            context_window: Context window in seconds (default: 120 = 2 minutes)
            
        Returns:
            Enhanced diagram image bytes, or None if enhancement failed
        """
        if self.client is None:
            raise RuntimeError("Gemini client not initialized. Check API key.")
        
        if not os.path.exists(diagram_path):
            raise FileNotFoundError(f"Diagram file not found: {diagram_path}")
        
        if config.VERBOSE:
            print(f"Enhancing diagram at timestamp {timestamp:.2f}s...")
        
        try:
            # Get transcript context (2 minutes before, exact moment, 2 minutes after)
            prior_context, current_context, next_context = self.get_transcript_context(
                timestamp, context_window
            )
            
            # Format timestamp
            minutes = int(timestamp // 60)
            seconds = int(timestamp % 60)
            timestamp_str = f"{minutes:02d}:{seconds:02d}"
            prompt = self._build_enhancement_prompt(
                prior_context or "",
                current_context or "",
                next_context or "",
                timestamp_str,
                timestamp,
            )
            
            # Load original diagram image (INPUT)
            if config.VERBOSE:
                print("Loading original diagram as input...")
            
            original_image = Image.open(diagram_path)
            
            # Call Gemini API with original diagram and prompt
            if config.VERBOSE:
                print("Calling Gemini API for diagram enhancement...")
            
            try:
                # Request image output explicitly (reduces text-only or inconsistent responses)
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[prompt, original_image],
                    config=_GenaiTypes.GenerateContentConfig(
                        response_modalities=["IMAGE"],
                    ),
                )
                
                # Extract image from response
                for part in response.parts:
                    if part.text is not None:
                        if config.VERBOSE:
                            print(f"Warning: Gemini returned text: {part.text[:200]}...")
                    elif part.inline_data is not None:
                        # Extract image using as_image() method (as in user's example)
                        try:
                            enhanced_image = part.as_image()
                            
                            # Save to temporary file first (exactly as in user's example)
                            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                            temp_path = temp_file.name
                            temp_file.close()
                            
                            # Save image to file (as in user's example: image.save("generated_image.png"))
                            enhanced_image.save(temp_path)
                            
                            # Read the file back as bytes
                            with open(temp_path, 'rb') as f:
                                image_bytes = f.read()
                            
                            # Clean up temp file
                            try:
                                os.unlink(temp_path)
                            except:
                                pass
                            
                            # Validate the image bytes
                            test_img = Image.open(BytesIO(image_bytes))
                            test_img.verify()
                            
                            if config.VERBOSE:
                                print("✓ Diagram enhancement successful (image generated).")
                            return image_bytes
                            
                        except Exception as img_error:
                            if config.VERBOSE:
                                print(f"Error extracting image: {str(img_error)}")
                                import traceback
                                traceback.print_exc()
                            # Clean up temp file on error
                            try:
                                if 'temp_path' in locals():
                                    os.unlink(temp_path)
                            except:
                                pass
                            return None
                
                # If no image found in response
                if config.VERBOSE:
                    print("Warning: No image data in Gemini response.")
                return None
                
            except Exception as api_error:
                error_str = str(api_error)
                if config.VERBOSE:
                    print(f"API call failed: {error_str}")
                raise RuntimeError(f"Gemini API call failed: {error_str}")
            
        except Exception as e:
            if config.VERBOSE:
                print(f"Error enhancing diagram: {str(e)}")
            raise RuntimeError(f"Diagram enhancement failed: {str(e)}")
    
    def enhance_diagram_from_array(
        self,
        diagram_image,
        timestamp: float,
        context_window: float = 120.0
    ) -> Optional[bytes]:
        """
        Enhance diagram from OpenCV image array
        
        Args:
            diagram_image: OpenCV image (numpy array) - original diagram
            timestamp: Timestamp when diagram appears in video
            context_window: Context window in seconds (default: 120 = 2 minutes)
            
        Returns:
            Enhanced diagram image bytes, or None if enhancement failed
        """
        if self.client is None:
            raise RuntimeError("Gemini client not initialized. Check API key.")
        
        if config.VERBOSE:
            print(f"Enhancing diagram at timestamp {timestamp:.2f}s...")
        
        try:
            # Get transcript context
            prior_context, current_context, next_context = self.get_transcript_context(
                timestamp, context_window
            )
            
            # Convert OpenCV image to PIL Image
            import cv2
            import numpy as np
            
            # Convert BGR to RGB
            if len(diagram_image.shape) == 3:
                rgb_image = cv2.cvtColor(diagram_image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = diagram_image
            
            original_image = Image.fromarray(rgb_image)
            
            # Format timestamp
            minutes = int(timestamp // 60)
            seconds = int(timestamp % 60)
            timestamp_str = f"{minutes:02d}:{seconds:02d}"
            prompt = self._build_enhancement_prompt(
                prior_context or "",
                current_context or "",
                next_context or "",
                timestamp_str,
                timestamp,
            )
            
            # Call Gemini API
            if config.VERBOSE:
                print("Calling Gemini API for diagram enhancement...")
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt, original_image],
                config=_GenaiTypes.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                ),
            )
            
            # Extract image from response (following user's example pattern)
            for part in response.parts:
                if part.text is not None:
                    if config.VERBOSE:
                        print(f"Warning: Gemini returned text: {part.text[:200]}...")
                elif part.inline_data is not None:
                    try:
                        # Use as_image() method exactly as in user's example
                        enhanced_image = part.as_image()
                        
                        # Save to temporary file first (matching user's example: image.save("generated_image.png"))
                        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                        temp_path = temp_file.name
                        temp_file.close()
                        
                        # Save image to file (as in user's example)
                        enhanced_image.save(temp_path)
                        
                        # Read the file back as bytes
                        with open(temp_path, 'rb') as f:
                            image_bytes = f.read()
                        
                        # Clean up temp file
                        try:
                            os.unlink(temp_path)
                        except:
                            pass
                        
                        if config.VERBOSE:
                            print("✓ Diagram enhancement successful.")
                        return image_bytes
                    except Exception as e:
                        if config.VERBOSE:
                            print(f"Error extracting image: {str(e)}")
                            import traceback
                            traceback.print_exc()
                        # Clean up temp file on error
                        try:
                            if 'temp_path' in locals():
                                os.unlink(temp_path)
                        except:
                            pass
                        return None
            
            if config.VERBOSE:
                print("Warning: No image data in Gemini response.")
            return None
            
        except Exception as e:
            if config.VERBOSE:
                print(f"Error enhancing diagram: {str(e)}")
            raise RuntimeError(f"Diagram enhancement failed: {str(e)}")
    
    def save_enhanced_diagram(
        self,
        enhanced_image_bytes: bytes,
        output_path: str
    ) -> str:
        """
        Save enhanced diagram to file
        
        Args:
            enhanced_image_bytes: Enhanced diagram image bytes
            output_path: Path to save enhanced diagram
            
        Returns:
            Path to saved file
        """
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        # Validate image bytes before saving
        try:
            from io import BytesIO
            test_img = Image.open(BytesIO(enhanced_image_bytes))
            test_img.verify()
            # Reopen after verify
            test_img = Image.open(BytesIO(enhanced_image_bytes))
            if test_img.size[0] == 0 or test_img.size[1] == 0:
                raise ValueError("Invalid image dimensions")
        except Exception as e:
            raise ValueError(f"Invalid image data: {str(e)}")
        
        with open(output_path, 'wb') as f:
            f.write(enhanced_image_bytes)
        
        if config.VERBOSE:
            print(f"Enhanced diagram saved to: {output_path}")
        
        return output_path
