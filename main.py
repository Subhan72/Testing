"""
Main pipeline for diagram detection from lecture videos
"""
import argparse
import sys
import os
import cv2
from PIL import Image
import imagehash
from video_processor import VideoProcessor
from stability_detector import StabilityDetector
from board_detector import BoardDetector
from content_analyzer import ContentAnalyzer
from completeness_checker import CompletenessChecker
from deduplicator import Deduplicator
from output_manager import OutputManager
from transcription_service import TranscriptionService
from diagram_enhancer import DiagramEnhancer
from diagram_classifier import DiagramClassifier
from summary_generator import SummaryGenerator
from html_doc_generator import HtmlDocGenerator
import config


def process_video(video_path: str, output_dir: str = None):
    """
    Process video to extract complete diagrams.

    Args:
        video_path: Path to input video file
        output_dir: Output directory for extracted diagrams
    """
    print("=" * 60)
    print("Diagram Detection System for Lecture Videos")
    print("=" * 60)
    print()
    
    # Initialize components
    print("Initializing components...")
    video_processor = VideoProcessor(video_path)
    stability_detector = StabilityDetector()
    board_detector = BoardDetector()
    content_analyzer = ContentAnalyzer()
    completeness_checker = CompletenessChecker()
    deduplicator = Deduplicator()
    output_manager = OutputManager(output_dir)
    
    print("Processing video...")
    print()
    
    # Process frames
    extracted_count = 0
    duplicate_count = 0
    # Two-copy color preservation: keep color ROI before grayscale; replace saved files with color after loop
    original_color_by_diagram_id = {}
    
    try:
        for frame, frame_number, timestamp in video_processor.extract_frames():
            # Detect board region
            board_region, board_type, board_confidence = board_detector.detect_board(frame)
            # Keep color copy before any grayscale (same ROI; used to replace files later)
            original_region = board_region.copy()
            
            # Enhance board contrast (may turn board_region grayscale for blackboard/whiteboard)
            if board_type != "unknown":
                board_region = board_detector.enhance_board_contrast(board_region, board_type)
            
            # Add frame to stability detector
            stability_detector.add_frame(board_region, timestamp, frame_number)
            
            # Check stability
            is_stable, stability_score = stability_detector.is_stable()
            
            # Analyze content
            content_analysis = content_analyzer.analyze(board_region)
            
            # Check completeness
            is_complete, completeness_score, reasons = completeness_checker.check_completeness(
                stability_score,
                is_stable,
                content_analysis,
                board_region,
                board_type
            )
            
            # If complete, check for duplicates/progressive building and save
            if is_complete:
                # Check diagram with content analysis
                action, replace_id, similar_hash, similarity = deduplicator.check_diagram(
                    board_region,
                    timestamp,
                    content_analysis.get('edge_density', 0.0),
                    content_analysis.get('contour_count', 0)
                )
                
                if action == 'save':
                    # New unique diagram
                    filepath = output_manager.save_diagram(
                        board_region,
                        timestamp,
                        frame_number,
                        board_type,
                        completeness_score,
                        content_analysis,
                        stability_score
                    )
                    # Store color copy to replace file after loop
                    original_color_by_diagram_id[output_manager.diagram_count] = original_region.copy()
                    
                    # Add to deduplicator
                    deduplicator.add_diagram(
                        board_region,
                        output_manager.diagram_count,
                        timestamp,
                        content_analysis.get('edge_density', 0.0),
                        content_analysis.get('contour_count', 0)
                    )
                    
                    extracted_count += 1
                    
                    if config.VERBOSE:
                        print(f"✓ Diagram {extracted_count} extracted at {timestamp:.2f}s (frame {frame_number})")
                        print(f"  Completeness: {completeness_score:.3f}, Stability: {stability_score:.3f}")
                        print(f"  Board: {board_type}, Saved: {filepath}")
                        print()
                
                elif action == 'replace':
                    # Progressive building - replace old diagram
                    # Get old hash before replacing
                    old_hash = similar_hash
                    
                    filepath = output_manager.replace_diagram(
                        replace_id,
                        board_region,
                        timestamp,
                        frame_number,
                        board_type,
                        completeness_score,
                        content_analysis,
                        stability_score
                    )
                    
                    # Calculate new hash
                    pil_image = Image.fromarray(cv2.cvtColor(board_region, cv2.COLOR_BGR2RGB))
                    new_hash = str(imagehash.phash(pil_image))
                    
                    # Store color copy to replace file after loop
                    original_color_by_diagram_id[replace_id] = original_region.copy()
                    
                    # Update deduplicator
                    deduplicator.remove_diagram_from_recent(replace_id)
                    if old_hash:
                        deduplicator.replace_diagram_in_saved(old_hash, board_region, new_hash)
                    deduplicator.add_diagram(
                        board_region,
                        replace_id,
                        timestamp,
                        content_analysis.get('edge_density', 0.0),
                        content_analysis.get('contour_count', 0)
                    )
                    
                    if config.VERBOSE:
                        print(f"↻ Diagram {replace_id} replaced (progressive building) at {timestamp:.2f}s (frame {frame_number})")
                        print(f"  Completeness: {completeness_score:.3f}, Stability: {stability_score:.3f}")
                        print(f"  Board: {board_type}, Updated: {filepath}")
                        print()
                
                else:  # action == 'skip'
                    # Duplicate or pointer variation - skip
                    duplicate_count += 1
                    if config.VERBOSE:
                        print(f"✗ Duplicate/pointer variation detected at {timestamp:.2f}s (similarity: {similarity:.3f})")
    
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user.")
    except Exception as e:
        print(f"\n\nError during processing: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Replace diagram files with color copies (two-copy preservation)
    for diagram_id, color_image in original_color_by_diagram_id.items():
        filepath = output_manager.diagram_id_to_file.get(diagram_id)
        if filepath and os.path.exists(filepath):
            if output_manager.image_format == 'png':
                cv2.imwrite(filepath, color_image)
            else:
                cv2.imwrite(filepath, color_image, [cv2.IMWRITE_JPEG_QUALITY, output_manager.image_quality])
    
    # Save metadata
    print("\nSaving metadata...")
    metadata_path = output_manager.save_metadata()
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Extraction Complete!")
    print("=" * 60)
    stats = output_manager.get_statistics()
    print(f"Total diagrams extracted: {stats['total_diagrams']}")
    print(f"Duplicates skipped: {duplicate_count}")
    print(f"Average completeness score: {stats['avg_completeness_score']:.3f}")
    print(f"Board types: {stats['board_types']}")
    print(f"\nOutput directory: {output_manager.output_dir}")
    print(f"Metadata file: {metadata_path}")
    print("=" * 60)
    
    return output_manager, stats


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Extract complete diagrams from lecture videos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py video.mp4
  python main.py video.mp4 --output extracted_diagrams
  python main.py video.mp4 --output results --verbose
        """
    )
    
    parser.add_argument(
        'video_path',
        type=str,
        help='Path to input video file'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output directory for extracted diagrams (default: extracted_diagrams)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to custom configuration file (optional)'
    )
    
    args = parser.parse_args()
    
    # Validate video file
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        sys.exit(1)
    
    # Process video
    try:
        process_video(args.video_path, args.output)
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def process_video_with_transcription(
    video_path: str,
    output_dir: str = None,
    enable_transcription: bool = True,
    enable_enhancement: bool = True,
    enable_summary: bool = True,
    enable_output_docs: bool = True,
    skip_diagram_extraction: bool = False,
    video_title: str | None = None,
):
    """
    Process video with transcription, diagram enhancement, and PDF generation
    
    Args:
        video_path: Path to input video file
        output_dir: Output directory for extracted diagrams
        enable_transcription: Enable transcription
        enable_enhancement: Enable diagram enhancement
        enable_summary: Enable summary generation
        enable_output_docs: Enable HTML/DOC and enhanced images output
        skip_diagram_extraction: Skip Step 1 and load existing diagrams
    """
    print("=" * 60)
    print("Video Processing with Transcription and Enhancement")
    print("=" * 60)
    print()
    
    # Step 1: Extract diagrams (existing functionality) or load existing
    output_manager = OutputManager(output_dir)
    
    if skip_diagram_extraction:
        print("Step 1: Loading existing diagrams from metadata...")
        if output_manager.load_metadata():
            stats = output_manager.get_statistics()
            print(f"✓ Loaded {stats['total_diagrams']} existing diagrams")
        else:
            print("✗ No existing metadata found. Please run diagram extraction first.")
            return
    else:
        print("Step 1: Extracting diagrams from video...")
        output_manager, stats = process_video(video_path, output_dir)
    
    if stats['total_diagrams'] == 0:
        print("\nWarning: No diagrams available. Skipping transcription and enhancement.")
        return {
            "transcript_html": None,
            "explanation_html": None,
            "transcript_doc": None,
            "explanation_doc": None,
            "enhanced_diagrams_dir": getattr(
                output_manager, "enhanced_diagrams_dir", os.path.join(output_manager.output_dir, "enhanced_diagrams")
            ),
        }
    
    # Step 2: Transcription
    transcription_service = None
    if enable_transcription and config.ENABLE_TRANSCRIPTION:
        print("\n" + "=" * 60)
        print("Step 2: Transcribing audio...")
        print("=" * 60)
        try:
            transcription_service = TranscriptionService()
            
            # Extract audio
            audio_path = transcription_service.extract_audio(video_path)
            
            # Transcribe
            transcript_data = transcription_service.transcribe_with_whisper(
                audio_path,
                language=config.TRANSCRIPTION_LANGUAGE
            )
            
            # Translate to English if needed
            if config.TRANSLATE_TO_ENGLISH:
                detected_lang = transcription_service.detect_language()
                if detected_lang and detected_lang.lower() != 'en':
                    transcription_service.translate_to_english()
            
            # Save transcript
            transcript_path = output_manager.save_transcript(
                transcription_service.transcript_english 
                if transcription_service.transcript_english 
                else transcription_service.transcript_data,
                "transcript.json"
            )
            
            if transcription_service.transcript_english:
                output_manager.save_transcript(
                    transcription_service.transcript_english,
                    "transcript_english.json"
                )
            
            print(f"✓ Transcription complete. Transcript saved.")
            
        except Exception as e:
            print(f"✗ Transcription failed: {str(e)}")
            if config.VERBOSE:
                import traceback
                traceback.print_exc()
            transcription_service = None
    
    # Step 2.5: Classify diagrams (hand_drawn / digital / not_diagram); discard not_diagram, crop digital
    if enable_enhancement and config.ENABLE_DIAGRAM_ENHANCEMENT:
        print("\n" + "=" * 60)
        print("Step 2.5: Classifying diagrams (hand_drawn / digital / not_diagram)...")
        print("=" * 60)
        try:
            classifier = DiagramClassifier()
            # Iterate over a copy: we may remove entries
            for diagram in list(output_manager.metadata):
                diagram_id = diagram.get("diagram_id")
                diagram_path = diagram.get("filepath")
                if not diagram_path or not os.path.exists(diagram_path):
                    continue
                try:
                    kind = classifier.classify(diagram_path)
                    if kind == "not_diagram":
                        if config.VERBOSE:
                            print(f"  Diagram {diagram_id}: NOT_DIAGRAM — discarding.")
                        output_manager.remove_diagram(diagram_id)
                        continue
                    if kind == "digital":
                        if config.VERBOSE:
                            print(f"  Diagram {diagram_id}: DIGITAL — cropping until valid...")
                        enhanced_path = os.path.join(
                            output_manager.enhanced_diagrams_dir,
                            f"enhanced_diagram_{diagram_id:04d}.png",
                        )
                        if classifier.crop_digital_until_correct(diagram_path, enhanced_path):
                            output_manager.enhanced_diagram_paths[diagram_id] = enhanced_path
                            if config.VERBOSE:
                                print(f"  Diagram {diagram_id}: cropped and saved to enhanced.")
                        else:
                            import shutil
                            shutil.copy2(diagram_path, enhanced_path)
                            output_manager.enhanced_diagram_paths[diagram_id] = enhanced_path
                            if config.VERBOSE:
                                print(f"  Diagram {diagram_id}: crop loop did not validate; using original.")
                        diagram["content_type"] = "digital"
                        continue
                    # hand_drawn
                    diagram["content_type"] = "hand_drawn"
                    if config.VERBOSE:
                        print(f"  Diagram {diagram_id}: HAND_DRAWN — will enhance.")
                except Exception as e:
                    if config.VERBOSE:
                        print(f"  Diagram {diagram_id} classification failed: {e}")
                    diagram["content_type"] = "hand_drawn"
            print(f"  Classification done. Remaining diagrams: {len(output_manager.metadata)}")
        except Exception as e:
            print(f"✗ Diagram classification failed: {str(e)}")
            if config.VERBOSE:
                import traceback
                traceback.print_exc()
    
    # Step 3: Diagram Enhancement (hand_drawn only; digital already cropped in Step 2.5)
    if enable_enhancement and config.ENABLE_DIAGRAM_ENHANCEMENT and transcription_service:
        print("\n" + "=" * 60)
        print("Step 3: Enhancing hand-drawn diagrams with Gemini...")
        print("=" * 60)
        try:
            enhancer = DiagramEnhancer(transcription_service)
            
            enhanced_count = 0
            diagram_metadata = output_manager.metadata
            
            for diagram in diagram_metadata:
                diagram_id = diagram.get('diagram_id')
                timestamp = diagram.get('timestamp', 0)
                diagram_path = diagram.get('filepath')
                content_type = diagram.get("content_type", "hand_drawn")
                
                if not diagram_path or not os.path.exists(diagram_path):
                    continue
                
                # Digital: already have cropped image in enhanced_diagram_paths; skip enhancement
                if content_type == "digital":
                    if output_manager.get_enhanced_diagram_path(diagram_id):
                        enhanced_count += 1
                    continue
                
                # Check if enhanced diagram already exists and is valid
                existing_enhanced = output_manager.get_enhanced_diagram_path(diagram_id)
                if existing_enhanced and os.path.exists(existing_enhanced):
                    try:
                        from PIL import Image
                        with Image.open(existing_enhanced) as img:
                            img.verify()
                        if config.VERBOSE:
                            print(f"⊘ Diagram {diagram_id} already enhanced, skipping...")
                        enhanced_count += 1
                        continue
                    except Exception:
                        if config.VERBOSE:
                            print(f"⚠ Diagram {diagram_id} enhanced image corrupted, regenerating...")
                        try:
                            os.remove(existing_enhanced)
                        except Exception:
                            pass
                
                try:
                    if config.VERBOSE:
                        print(f"Enhancing diagram {diagram_id} (hand-drawn) at {timestamp:.2f}s...")
                    
                    enhanced_bytes = enhancer.enhance_diagram(
                        diagram_path,
                        timestamp,
                        context_window=config.DIAGRAM_CONTEXT_WINDOW
                    )
                    
                    if enhanced_bytes:
                        from PIL import Image
                        from io import BytesIO
                        try:
                            img = Image.open(BytesIO(enhanced_bytes))
                            img.verify()
                            output_manager.save_enhanced_diagram(diagram_id, enhanced_bytes)
                            enhanced_count += 1
                            if config.VERBOSE:
                                print(f"✓ Diagram {diagram_id} enhanced successfully.")
                        except Exception as e:
                            if config.VERBOSE:
                                print(f"✗ Diagram {diagram_id} enhanced bytes invalid: {str(e)}")
                            if config.VERBOSE:
                                print(f"  Using original diagram for {diagram_id}")
                    else:
                        if config.VERBOSE:
                            print(f"✗ Diagram {diagram_id} enhancement returned no image.")
                
                except Exception as e:
                    if config.VERBOSE:
                        print(f"✗ Failed to enhance diagram {diagram_id}: {str(e)}")
                    continue
            
            print(f"\n✓ Enhanced {enhanced_count} out of {len(diagram_metadata)} diagrams.")
            
        except Exception as e:
            print(f"✗ Diagram enhancement failed: {str(e)}")
            if config.VERBOSE:
                import traceback
                traceback.print_exc()
    
    # Step 4: Summary (optional) and Detailed Explanation
    summary_text = None
    explanation_text = None
    if enable_summary and config.GENERATE_SUMMARY and transcription_service:
        print("\n" + "=" * 60)
        print("Step 4: Generating summary...")
        print("=" * 60)
        try:
            summary_gen = SummaryGenerator(transcription_service)
            summary_text = summary_gen.generate_summary()
            output_manager.save_summary(summary_text)
            print(f"✓ Summary generated and saved.")
        except Exception as e:
            print(f"✗ Summary generation failed: {str(e)}")
            if config.VERBOSE:
                import traceback
                traceback.print_exc()

    # Step 4b: Detailed explanation (for HTML/DOC: explanation with diagrams in place)
    explanation_text = None
    if enable_output_docs and getattr(config, 'GENERATE_OUTPUT_DOCUMENTS', True) and transcription_service:
        print("\n" + "=" * 60)
        print("Step 4b: Generating detailed explanation...")
        print("=" * 60)
        try:
            summary_gen = SummaryGenerator(transcription_service)
            explanation_text = summary_gen.generate_detailed_explanation(
                diagram_metadata=output_manager.metadata
            )
            explanation_path = output_manager.save_summary(
                explanation_text,
                "explanation.txt"
            )
            if config.VERBOSE:
                print(f"✓ Explanation saved to: {explanation_path}")
        except Exception as e:
            print(f"✗ Detailed explanation failed: {str(e)}")
            explanation_text = None
            if config.VERBOSE:
                import traceback
                traceback.print_exc()
    
    # Step 5: Generate 2 HTML, 2 DOC, and enhanced images folder (no PDF)
    transcript_html_path = None
    explanation_html_path = None
    transcript_doc_path = None
    explanation_doc_path = None
    enhanced_diagrams_dir = getattr(output_manager, 'enhanced_diagrams_dir', os.path.join(output_manager.output_dir, 'enhanced_diagrams'))
    if enable_output_docs and getattr(config, 'GENERATE_OUTPUT_DOCUMENTS', True) and transcription_service:
        print("\n" + "=" * 60)
        print("Step 5: Generating HTML, DOC, and enhanced images...")
        print("=" * 60)
        try:
            doc_gen = HtmlDocGenerator()
            transcript_data = (
                transcription_service.transcript_english
                if transcription_service.transcript_english
                else transcription_service.transcript_data
            )
            if not transcript_data:
                print("✗ No transcript data available for output generation.")
            else:
                diagram_metadata = output_manager.metadata
                enhanced_diagram_paths = {}
                for diagram in diagram_metadata:
                    diagram_id = diagram.get('diagram_id')
                    enhanced_path = output_manager.get_enhanced_diagram_path(diagram_id)
                    if enhanced_path and os.path.exists(enhanced_path):
                        enhanced_diagram_paths[diagram_id] = enhanced_path
                    else:
                        original_path = diagram.get('filepath')
                        if original_path and os.path.exists(original_path):
                            enhanced_diagram_paths[diagram_id] = original_path

                video_name = os.path.splitext(os.path.basename(video_path))[0]
                out_dir = output_manager.output_dir

                # Prefer the original video title (YouTube/Drive) when provided;
                # otherwise, fall back to a descriptive title from the transcript,
                # then finally to the raw video filename.
                if video_title and str(video_title).strip():
                    lecture_title = str(video_title).strip()
                    if config.VERBOSE:
                        print(f"Using video title from source: {lecture_title}")
                else:
                    try:
                        summary_gen = SummaryGenerator(transcription_service)
                        lecture_title = summary_gen.generate_lecture_title(
                            transcript_text=transcription_service.get_full_transcript(use_english=True)
                            if transcription_service
                            else None
                        )
                        if not lecture_title or not lecture_title.strip():
                            lecture_title = video_name
                        else:
                            if config.VERBOSE:
                                print(f"Generated lecture title: {lecture_title}")
                    except Exception as e:
                        if config.VERBOSE:
                            print(f"Title generation failed, using video name: {e}")
                        lecture_title = video_name

                # Use a filename-safe version of the title for all output files.
                import re as _re_title
                safe_base = _re_title.sub(r'[<>:\"/\\\\|?*]', "_", lecture_title).strip().strip(". ") or video_name

                # HTML 1: Transcript with images by timestamp
                transcript_html_path = os.path.join(
                    out_dir,
                    f"{safe_base}_transcript.html",
                )
                doc_gen.generate_transcript_html(
                    transcript_html_path,
                    transcript_data,
                    diagram_metadata,
                    enhanced_diagram_paths,
                    video_name=lecture_title,
                    output_dir=out_dir,
                )
                print(f"✓ Transcript HTML: {transcript_html_path}")

                # HTML 2: Explanation with TOC and [DIAGRAM:N] → images
                if explanation_text:
                    explanation_html_path = os.path.join(
                        out_dir,
                        f"{safe_base}_explanation.html",
                    )
                    doc_gen.generate_explanation_html(
                        explanation_html_path,
                        explanation_text,
                        diagram_metadata,
                        enhanced_diagram_paths,
                        video_name=lecture_title,
                        output_dir=out_dir,
                    )
                    print(f"✓ Explanation HTML: {explanation_html_path}")

                # DOC 1: Transcript (own try so failure doesn't skip explanation DOC)
                transcript_doc_path = os.path.join(
                    out_dir,
                    f"{safe_base}_transcript.docx",
                )
                try:
                    doc_gen.generate_transcript_doc(
                        transcript_doc_path,
                        transcript_data,
                        diagram_metadata,
                        enhanced_diagram_paths,
                        video_name=lecture_title,
                        output_dir=out_dir,
                    )
                    print(f"✓ Transcript DOC: {transcript_doc_path}")
                except Exception as doc_e:
                    print(f"✗ Transcript DOC failed: {doc_e}")
                    transcript_doc_path = None
                    if config.VERBOSE:
                        import traceback
                        traceback.print_exc()

                # DOC 2: Explanation
                explanation_doc_path = os.path.join(
                    out_dir,
                    f"{safe_base}_explanation.docx",
                )
                try:
                    doc_gen.generate_explanation_doc(
                        explanation_doc_path,
                        explanation_text or "No explanation generated.",
                        diagram_metadata,
                        enhanced_diagram_paths,
                        video_name=lecture_title,
                        output_dir=out_dir,
                    )
                    print(f"✓ Explanation DOC: {explanation_doc_path}")
                except Exception as doc_e:
                    print(f"✗ Explanation DOC failed: {doc_e}")
                    explanation_doc_path = None
                    if config.VERBOSE:
                        import traceback
                        traceback.print_exc()

                print(f"✓ Enhanced images folder: {enhanced_diagrams_dir}")
        except Exception as e:
            print(f"✗ HTML/DOC generation failed: {str(e)}")
            if config.VERBOSE:
                import traceback
                traceback.print_exc()
    
    if transcription_service:
        transcription_service.cleanup()
    
    print("\n" + "=" * 60)
    print("Processing Complete!")
    print("=" * 60)
    print(f"Output directory: {output_manager.output_dir}")
    if transcript_html_path:
        print(f"Transcript HTML: {transcript_html_path}")
    if explanation_html_path:
        print(f"Explanation HTML: {explanation_html_path}")
    if transcript_doc_path:
        print(f"Transcript DOC: {transcript_doc_path}")
    if explanation_doc_path:
        print(f"Explanation DOC: {explanation_doc_path}")
    print(f"Enhanced images: {enhanced_diagrams_dir}")
    print("=" * 60)

    return {
        "transcript_html": transcript_html_path,
        "explanation_html": explanation_html_path,
        "transcript_doc": transcript_doc_path,
        "explanation_doc": explanation_doc_path,
        "enhanced_diagrams_dir": enhanced_diagrams_dir,
    }


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Extract complete diagrams from lecture videos with transcription and enhancement',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py video.mp4
  python main.py video.mp4 --output extracted_diagrams
  python main.py video.mp4 --transcribe --enhance --output-docs
  python main.py video.mp4 --no-transcribe  # Skip transcription
        """
    )
    
    parser.add_argument(
        'video_path',
        type=str,
        help='Path to input video file'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output directory for extracted diagrams (default: extracted_diagrams)'
    )
    
    parser.add_argument(
        '--transcribe',
        action='store_true',
        default=None,
        help='Enable transcription (overrides config)'
    )
    
    parser.add_argument(
        '--no-transcribe',
        action='store_true',
        help='Disable transcription'
    )
    
    parser.add_argument(
        '--enhance',
        action='store_true',
        default=None,
        help='Enable diagram enhancement (overrides config)'
    )
    
    parser.add_argument(
        '--no-enhance',
        action='store_true',
        help='Disable diagram enhancement'
    )
    
    parser.add_argument(
        '--output-docs',
        action='store_true',
        default=None,
        help='Enable HTML/DOC and enhanced images output (overrides config)'
    )
    
    parser.add_argument(
        '--no-output-docs',
        action='store_true',
        help='Disable HTML/DOC generation'
    )
    
    parser.add_argument(
        '--summary',
        action='store_true',
        default=None,
        help='Enable summary generation (overrides config)'
    )
    
    parser.add_argument(
        '--no-summary',
        action='store_true',
        help='Disable summary generation'
    )
    
    parser.add_argument(
        '--skip-extraction',
        action='store_true',
        help='Skip diagram extraction and use existing diagrams from metadata'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to custom configuration file (optional)'
    )
    
    args = parser.parse_args()
    
    # Validate video file
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        sys.exit(1)
    
    # Determine feature flags
    enable_transcription = config.ENABLE_TRANSCRIPTION
    if args.transcribe:
        enable_transcription = True
    elif args.no_transcribe:
        enable_transcription = False
    
    enable_enhancement = config.ENABLE_DIAGRAM_ENHANCEMENT
    if args.enhance:
        enable_enhancement = True
    elif args.no_enhance:
        enable_enhancement = False
    
    enable_output_docs = getattr(config, 'GENERATE_OUTPUT_DOCUMENTS', True)
    if args.output_docs:
        enable_output_docs = True
    elif args.no_output_docs:
        enable_output_docs = False
    
    enable_summary = config.GENERATE_SUMMARY
    if args.summary:
        enable_summary = True
    elif args.no_summary:
        enable_summary = False
    
    # Process video
    try:
        if enable_transcription or enable_enhancement or enable_output_docs:
            # Use full pipeline with transcription
            process_video_with_transcription(
                args.video_path,
                args.output,
                enable_transcription,
                enable_enhancement,
                enable_summary,
                enable_output_docs,
                skip_diagram_extraction=args.skip_extraction,
                video_title=None,
            )
        else:
            # Use basic diagram extraction only (unless skipping)
            if args.skip_extraction:
                print("Warning: --skip-extraction requires --transcribe, --enhance, or --output-docs")
            else:
                process_video(args.video_path, args.output)
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
