"""
Transcription service using OpenAI Whisper API
Handles audio extraction, transcription with timestamps, language detection, and translation
"""
import os
import json
import tempfile
import subprocess
import shutil
from typing import Dict, List, Optional, Tuple
from openai import OpenAI
from api_config import APIConfig
import config


class TranscriptionService:
    """Service for transcribing video audio with timestamps"""
    
    def __init__(self):
        """Initialize transcription service"""
        self.openai_client = None
        self.transcript_data = None
        self.transcript_english = None
        self.audio_file_path = None
        
        # Initialize OpenAI client
        try:
            api_key = APIConfig.get_openai_api_key()
            self.openai_client = OpenAI(api_key=api_key)
        except ValueError as e:
            if config.VERBOSE:
                print(f"Warning: {e}")
    
    def extract_audio(self, video_path: str, output_audio_path: Optional[str] = None) -> str:
        """
        Extract audio from video file
        
        Args:
            video_path: Path to input video file
            output_audio_path: Optional path for output audio file
            
        Returns:
            Path to extracted audio file
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        if output_audio_path is None:
            # Create temporary audio file
            temp_dir = tempfile.gettempdir()
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_audio_path = os.path.join(temp_dir, f"{video_name}_audio.mp3")
        
        if config.VERBOSE:
            print(f"Extracting audio from video...")
        
        # First, try using ffmpeg directly (more reliable)
        ffmpeg_path = shutil.which('ffmpeg')
        if ffmpeg_path:
            if config.VERBOSE:
                print("Attempting audio extraction using ffmpeg directly...")
            try:
                cmd = [
                    ffmpeg_path,
                    '-i', video_path,
                    '-vn',  # No video
                    '-acodec', 'libmp3lame',
                    '-ab', '192k',
                    '-ar', '44100',
                    '-y',  # Overwrite output file
                    output_audio_path
                ]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=300  # 5 minute timeout
                )
                
                # Check if output file was created
                if os.path.exists(output_audio_path) and os.path.getsize(output_audio_path) > 0:
                    self.audio_file_path = output_audio_path
                    if config.VERBOSE:
                        print(f"Audio extracted using ffmpeg to: {output_audio_path}")
                    return output_audio_path
                else:
                    raise RuntimeError("FFmpeg extraction completed but output file is empty or missing")
                    
            except subprocess.TimeoutExpired:
                raise RuntimeError("Audio extraction timed out after 5 minutes")
            except subprocess.CalledProcessError as ffmpeg_error:
                error_output = ffmpeg_error.stderr or ffmpeg_error.stdout or ""
                if "no audio" in error_output.lower() or "does not contain any stream" in error_output.lower():
                    raise RuntimeError("Video has no audio track. Cannot transcribe without audio.")
                if config.VERBOSE:
                    print(f"FFmpeg direct extraction failed, trying MoviePy...")
                # Fall through to MoviePy
            except FileNotFoundError:
                if config.VERBOSE:
                    print("FFmpeg not found, trying MoviePy...")
                # Fall through to MoviePy
        
        # Fallback to MoviePy (lazy import to avoid ffmpeg requirement at import time)
        try:
            if config.VERBOSE:
                print("Attempting audio extraction using MoviePy...")
            # Lazy import MoviePy only when needed
            from moviepy import VideoFileClip
            video = VideoFileClip(video_path)
            
            # Check if video has audio track
            if video.audio is None:
                video.close()
                raise RuntimeError("Video has no audio track. Cannot transcribe without audio.")
            
            # Extract audio
            video.audio.write_audiofile(
                output_audio_path,
                codec='mp3',
                bitrate='192k'
            )
            video.close()
            
            self.audio_file_path = output_audio_path
            
            if config.VERBOSE:
                print(f"Audio extracted to: {output_audio_path}")
            
            return output_audio_path
            
        except Exception as e:
            error_msg = str(e)
            if "no audio" in error_msg.lower() or "has no audio track" in error_msg.lower():
                raise RuntimeError("Video has no audio track. Cannot transcribe without audio.")
            elif "ffmpeg" in error_msg.lower() or "Error passing" in error_msg:
                raise RuntimeError(
                    f"Failed to extract audio. The video file may be corrupted, have no audio track, "
                    f"or ffmpeg/MoviePy cannot parse it.\n"
                    f"Error details: {error_msg}\n"
                    f"Please verify the video file has an audio track and is in a supported format."
                )
            else:
                raise RuntimeError(f"Failed to extract audio: {error_msg}")
    
    def transcribe_with_whisper(
        self,
        audio_path: Optional[str] = None,
        language: Optional[str] = None
    ) -> Dict:
        """
        Transcribe audio using OpenAI Whisper API with timestamps
        
        Args:
            audio_path: Path to audio file (uses extracted audio if None)
            language: Optional language code (e.g., 'en', 'es', 'fr'). Auto-detect if None
            
        Returns:
            Dictionary containing transcript with timestamps
        """
        if self.openai_client is None:
            raise RuntimeError("OpenAI client not initialized. Check API key.")
        
        audio_file = audio_path or self.audio_file_path
        if not audio_file or not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        if config.VERBOSE:
            print(f"Transcribing audio with Whisper API...")
            if language:
                print(f"Using language: {language}")
        
        # Check file size (Whisper API limit is 25MB)
        file_size = os.path.getsize(audio_file)
        file_size_mb = file_size / (1024 * 1024)
        
        # If small enough, do a single request
        if file_size_mb <= 24.0:
            try:
                with open(audio_file, 'rb') as audio:
                    transcript = self.openai_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio,
                        response_format="verbose_json",
                        language=language,  # None = auto-detect
                        timestamp_granularities=["segment", "word"]
                    )
                self.transcript_data = {
                    'text': transcript.text,
                    'language': transcript.language,
                    'duration': transcript.duration,
                    'segments': [
                        {
                            'id': seg.id,
                            'start': seg.start,
                            'end': seg.end,
                            'text': seg.text,
                            'words': [
                                {
                                    'word': word.word,
                                    'start': word.start,
                                    'end': word.end
                                } for word in seg.words
                            ] if hasattr(seg, 'words') and seg.words else []
                        } for seg in transcript.segments
                    ]
                }
                if config.VERBOSE:
                    print(f"Transcription complete. Language detected: {transcript.language}")
                    print(f"Duration: {transcript.duration:.2f} seconds")
                    print(f"Segments: {len(self.transcript_data['segments'])}")
                return self.transcript_data
            except Exception as e:
                raise RuntimeError(f"Transcription failed: {str(e)}")
        
        # Large file: use chunked transcription
        if config.VERBOSE:
            print(f"Audio file is {file_size_mb:.2f}MB – using chunked transcription.")
        return self._transcribe_with_whisper_chunks(audio_file, language)

    def _transcribe_with_whisper_chunks(
        self,
        audio_file: str,
        language: Optional[str] = None,
    ) -> Dict:
        """
        Transcribe large audio by splitting into smaller chunks and stitching results.
        Keeps each Whisper request under the ~25MB limit.
        """
        ffmpeg_path = shutil.which('ffmpeg')
        if not ffmpeg_path:
            # Railway/container environments may not expose ffmpeg on PATH but MoviePy can
            # still work via imageio-ffmpeg. Use a robust fallback instead of failing hard.
            if config.VERBOSE:
                print("ffmpeg not found on PATH. Falling back to MoviePy-based chunking...")
            return self._transcribe_with_whisper_chunks_moviepy(audio_file, language)
        
        # Segment the already-extracted audio file into ~TRANSCRIPTION_CHUNK_SECONDS pieces
        segment_time = getattr(config, "TRANSCRIPTION_CHUNK_SECONDS", 600.0)
        chunk_dir = tempfile.mkdtemp(prefix="whisper_chunks_")
        chunk_pattern = os.path.join(chunk_dir, "chunk_%03d.mp3")
        
        if config.VERBOSE:
            print(f"Chunking audio into ~{segment_time:.0f}s segments for transcription...")
        
        try:
            cmd = [
                ffmpeg_path,
                "-i", audio_file,
                "-f", "segment",
                "-segment_time", str(int(segment_time)),
                "-c", "copy",
                "-reset_timestamps", "1",
                chunk_pattern,
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=600,  # 10 minutes for chunking
            )
        except subprocess.TimeoutExpired:
            shutil.rmtree(chunk_dir, ignore_errors=True)
            raise RuntimeError("Audio chunking timed out.")
        except subprocess.CalledProcessError as e:
            shutil.rmtree(chunk_dir, ignore_errors=True)
            err = e.stderr or e.stdout or ""
            raise RuntimeError(f"Audio chunking failed: {err}")
        except Exception as e:
            shutil.rmtree(chunk_dir, ignore_errors=True)
            raise RuntimeError(f"Audio chunking failed: {e}")
        
        # Collect chunks
        chunk_files = sorted(
            os.path.join(chunk_dir, f) for f in os.listdir(chunk_dir) if f.lower().endswith(".mp3")
        )
        if not chunk_files:
            shutil.rmtree(chunk_dir, ignore_errors=True)
            raise RuntimeError("Audio chunking produced no segments.")
        
        all_segments: List[Dict] = []
        text_parts: List[str] = []
        total_duration = 0.0
        language_detected: Optional[str] = None
        
        try:
            for idx, chunk_path in enumerate(chunk_files):
                if config.VERBOSE:
                    print(f"Transcribing chunk {idx + 1}/{len(chunk_files)}: {chunk_path}")
                with open(chunk_path, "rb") as audio:
                    tr = self.openai_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio,
                        response_format="verbose_json",
                        language=language,
                        timestamp_granularities=["segment", "word"],
                    )
                # Record language from first chunk
                if language_detected is None:
                    language_detected = getattr(tr, "language", None)
                
                duration = getattr(tr, "duration", None) or 0.0
                segments = getattr(tr, "segments", []) or []
                
                for seg in segments:
                    base_start = getattr(seg, "start", 0.0) or 0.0
                    base_end = getattr(seg, "end", 0.0) or 0.0
                    seg_dict = {
                        "id": getattr(seg, "id", len(all_segments)),
                        "start": base_start + total_duration,
                        "end": base_end + total_duration,
                        "text": getattr(seg, "text", ""),
                        "words": [],
                    }
                    words = getattr(seg, "words", None) or []
                    for word in words:
                        seg_dict["words"].append(
                            {
                                "word": getattr(word, "word", ""),
                                "start": (getattr(word, "start", 0.0) or 0.0) + total_duration,
                                "end": (getattr(word, "end", 0.0) or 0.0) + total_duration,
                            }
                        )
                    all_segments.append(seg_dict)
                
                if getattr(tr, "text", None):
                    text_parts.append(tr.text)
                total_duration += duration
        finally:
            # Always clean up chunk directory
            shutil.rmtree(chunk_dir, ignore_errors=True)
        
        full_text = " ".join(text_parts).strip()
        self.transcript_data = {
            "text": full_text,
            "language": language_detected or (config.TRANSCRIPTION_LANGUAGE or "en"),
            "duration": total_duration,
            "segments": all_segments,
        }
        
        if config.VERBOSE:
            print(f"Chunked transcription complete. Duration: {total_duration:.2f}s, segments: {len(all_segments)}")
        
        return self.transcript_data

    def _transcribe_with_whisper_chunks_moviepy(
        self,
        audio_file: str,
        language: Optional[str] = None,
    ) -> Dict:
        """
        Fallback chunked transcription using MoviePy when ffmpeg binary is unavailable on PATH.
        This keeps Whisper requests under size limits without requiring system ffmpeg detection.
        """
        try:
            from moviepy import AudioFileClip
        except Exception as e:
            raise RuntimeError(
                "Audio is too large for a single Whisper request and chunking fallback is unavailable. "
                "Install ffmpeg (or ensure MoviePy/audio backend is available), or trim the video."
            ) from e

        segment_time = float(getattr(config, "TRANSCRIPTION_CHUNK_SECONDS", 600.0) or 600.0)
        if segment_time <= 1:
            segment_time = 600.0

        chunk_dir = tempfile.mkdtemp(prefix="whisper_chunks_mp_")
        all_segments: List[Dict] = []
        text_parts: List[str] = []
        language_detected: Optional[str] = None

        try:
            clip = AudioFileClip(audio_file)
            total_audio_duration = float(getattr(clip, "duration", 0.0) or 0.0)
            if total_audio_duration <= 0:
                raise RuntimeError("Could not determine audio duration for chunking.")

            chunk_count = int(total_audio_duration // segment_time) + (1 if total_audio_duration % segment_time > 0 else 0)
            if config.VERBOSE:
                print(f"Chunking audio via MoviePy into {chunk_count} segment(s) of ~{segment_time:.0f}s...")

            for idx in range(chunk_count):
                start = idx * segment_time
                end = min((idx + 1) * segment_time, total_audio_duration)
                if end <= start:
                    continue

                chunk_path = os.path.join(chunk_dir, f"chunk_{idx:03d}.mp3")
                subclip = None
                try:
                    # MoviePy API compatibility across versions.
                    if hasattr(clip, "subclipped"):
                        subclip = clip.subclipped(start, end)
                    else:
                        subclip = clip.subclip(start, end)
                    subclip.write_audiofile(
                        chunk_path,
                        codec="mp3",
                        bitrate="192k",
                        logger=None,
                    )
                finally:
                    if subclip is not None:
                        try:
                            subclip.close()
                        except Exception:
                            pass

                if not os.path.exists(chunk_path) or os.path.getsize(chunk_path) <= 0:
                    raise RuntimeError(f"MoviePy chunking produced empty chunk: {chunk_path}")

                if config.VERBOSE:
                    print(f"Transcribing chunk {idx + 1}/{chunk_count}: {chunk_path}")

                with open(chunk_path, "rb") as audio:
                    tr = self.openai_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio,
                        response_format="verbose_json",
                        language=language,
                        timestamp_granularities=["segment", "word"],
                    )

                if language_detected is None:
                    language_detected = getattr(tr, "language", None)

                segments = getattr(tr, "segments", []) or []
                for seg in segments:
                    base_start = getattr(seg, "start", 0.0) or 0.0
                    base_end = getattr(seg, "end", 0.0) or 0.0
                    seg_dict = {
                        "id": getattr(seg, "id", len(all_segments)),
                        "start": base_start + start,
                        "end": base_end + start,
                        "text": getattr(seg, "text", ""),
                        "words": [],
                    }
                    words = getattr(seg, "words", None) or []
                    for word in words:
                        seg_dict["words"].append(
                            {
                                "word": getattr(word, "word", ""),
                                "start": (getattr(word, "start", 0.0) or 0.0) + start,
                                "end": (getattr(word, "end", 0.0) or 0.0) + start,
                            }
                        )
                    all_segments.append(seg_dict)

                if getattr(tr, "text", None):
                    text_parts.append(tr.text)

            try:
                clip.close()
            except Exception:
                pass

            self.transcript_data = {
                "text": " ".join(text_parts).strip(),
                "language": language_detected or (config.TRANSCRIPTION_LANGUAGE or "en"),
                "duration": total_audio_duration,
                "segments": all_segments,
            }

            if config.VERBOSE:
                print(
                    f"Chunked transcription (MoviePy fallback) complete. "
                    f"Duration: {total_audio_duration:.2f}s, segments: {len(all_segments)}"
                )

            return self.transcript_data
        finally:
            shutil.rmtree(chunk_dir, ignore_errors=True)
    
    def detect_language(self) -> Optional[str]:
        """
        Detect language from transcript
        
        Returns:
            Language code in ISO-639-1 format (e.g., 'en', 'es', 'fr') or None if not available
        """
        if self.transcript_data:
            lang = self.transcript_data.get('language')
            # Convert full language names to ISO-639-1 codes
            lang_map = {
                'english': 'en',
                'spanish': 'es',
                'french': 'fr',
                'german': 'de',
                'italian': 'it',
                'portuguese': 'pt',
                'chinese': 'zh',
                'japanese': 'ja',
                'korean': 'ko',
                'russian': 'ru',
                'arabic': 'ar',
                'hindi': 'hi'
            }
            if lang:
                lang_lower = lang.lower()
                return lang_map.get(lang_lower, lang_lower[:2] if len(lang_lower) >= 2 else lang)
            return lang
        return None
    
    def translate_to_english(self) -> Dict:
        """
        Translate transcript to English if not already in English
        
        Returns:
            Translated transcript dictionary
        """
        if self.openai_client is None:
            raise RuntimeError("OpenAI client not initialized. Check API key.")
        
        if not self.transcript_data:
            raise ValueError("No transcript available. Run transcribe_with_whisper() first.")
        
        detected_language = self.detect_language()
        
        # If already English, return original (no translation needed)
        if detected_language and detected_language.lower() == 'en':
            if config.VERBOSE:
                print("Transcript is already in English. No translation needed.")
            self.transcript_english = self.transcript_data
            return self.transcript_english
        
        if config.VERBOSE:
            print(f"Translating from {detected_language} to English...")
        
        try:
            # Use Whisper API translations endpoint (automatically translates to English)
            audio_file = self.audio_file_path
            if not audio_file or not os.path.exists(audio_file):
                raise FileNotFoundError("Audio file not found for translation")
            
            with open(audio_file, 'rb') as audio:
                # Use translations endpoint which automatically translates to English.
                # Note: timestamp_granularities is not supported by the translations API.
                transcript = self.openai_client.audio.translations.create(
                    model="whisper-1",
                    file=audio,
                    response_format="verbose_json"
                )
            
            # Create translated transcript structure.
            # Translations API verbose_json returns text, duration, segments (no word-level by default).
            segments = getattr(transcript, 'segments', None) or []
            duration = getattr(transcript, 'duration', None)
            if duration is None and self.transcript_data:
                duration = self.transcript_data.get('duration') or 0.0

            if segments:
                seg_list = [
                    {
                        'id': getattr(seg, 'id', idx),
                        'start': getattr(seg, 'start', 0.0),
                        'end': getattr(seg, 'end', 0.0),
                        'text': getattr(seg, 'text', ''),
                        'words': [
                            {'word': getattr(word, 'word', ''), 'start': getattr(word, 'start', 0), 'end': getattr(word, 'end', 0)}
                            for word in (getattr(seg, 'words', None) or [])
                        ]
                    }
                    for idx, seg in enumerate(segments)
                ]
            else:
                # Fallback: translation returned only full text (no segments); use one segment
                seg_list = [{'id': 0, 'start': 0.0, 'end': float(duration or 0), 'text': (transcript.text or '').strip(), 'words': []}]

            self.transcript_english = {
                'text': transcript.text,
                'language': 'en',
                'duration': duration,
                'segments': seg_list,
                'original_language': detected_language
            }
            
            if config.VERBOSE:
                print("Translation complete.")
            
            return self.transcript_english
            
        except Exception as e:
            raise RuntimeError(f"Translation failed: {str(e)}")
    
    def get_transcript_segment(
        self,
        start_time: float,
        end_time: float,
        use_english: bool = True
    ) -> str:
        """
        Get transcript text for a specific time range
        
        Args:
            start_time: Start timestamp in seconds
            end_time: End timestamp in seconds
            use_english: Use English translation if available
            
        Returns:
            Transcript text for the specified time range
        """
        transcript = self.transcript_english if (use_english and self.transcript_english) else self.transcript_data
        
        if not transcript:
            return ""
        
        segments = transcript.get('segments', [])
        text_parts = []
        
        for segment in segments:
            seg_start = segment.get('start', 0)
            seg_end = segment.get('end', 0)
            
            # Check if segment overlaps with requested time range
            if seg_start <= end_time and seg_end >= start_time:
                text_parts.append(segment.get('text', ''))
        
        return ' '.join(text_parts).strip()
    
    def get_full_transcript(self, use_english: bool = True) -> str:
        """
        Get full transcript text
        
        Args:
            use_english: Use English translation if available
            
        Returns:
            Full transcript text
        """
        transcript = self.transcript_english if (use_english and self.transcript_english) else self.transcript_data
        
        if not transcript:
            return ""
        
        return transcript.get('text', '')
    
    def save_transcript(self, output_path: str, use_english: bool = True):
        """
        Save transcript to JSON file
        
        Args:
            output_path: Path to save transcript JSON
            use_english: Save English translation if available
        """
        transcript = self.transcript_english if (use_english and self.transcript_english) else self.transcript_data
        
        if not transcript:
            raise ValueError("No transcript available to save.")
        
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(transcript, f, indent=2, ensure_ascii=False)
        
        if config.VERBOSE:
            print(f"Transcript saved to: {output_path}")
    
    def cleanup(self):
        """Clean up temporary audio file"""
        if self.audio_file_path and os.path.exists(self.audio_file_path):
            try:
                # Only delete if it's in temp directory
                if tempfile.gettempdir() in self.audio_file_path:
                    os.remove(self.audio_file_path)
                    if config.VERBOSE:
                        print(f"Cleaned up temporary audio file: {self.audio_file_path}")
            except Exception as e:
                if config.VERBOSE:
                    print(f"Warning: Could not delete temporary audio file: {e}")
