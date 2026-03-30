"""
Configuration parameters for diagram detection system
"""
import os

# Video Processing
FRAME_EXTRACTION_INTERVAL = 0.5  # Extract frame every 0.5 seconds
MIN_FRAME_INTERVAL = 0.1  # Minimum interval between frames to process

# Temporal Stability Detection
STABILITY_WINDOW_DURATION = 2.0  # Seconds of stability required
STABILITY_THRESHOLD_SSIM = 0.95  # SSIM threshold for stability (0-1, higher = more stable)
STABILITY_THRESHOLD_MSE = 100.0  # MSE threshold for stability (lower = more stable)
STABILITY_FRAME_COUNT = 4  # Minimum number of stable frames in window

# Board Detection
BOARD_DETECTION_ENABLED = True
BLACKBOARD_LOWER_HSV = (0, 0, 0)  # Lower HSV for blackboard detection
BLACKBOARD_UPPER_HSV = (180, 255, 50)  # Upper HSV for blackboard detection
WHITEBOARD_LOWER_HSV = (0, 0, 200)  # Lower HSV for whiteboard detection
WHITEBOARD_UPPER_HSV = (180, 30, 255)  # Upper HSV for whiteboard detection
MIN_BOARD_AREA_RATIO = 0.1  # Minimum board area as ratio of frame

# Content Analysis
CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150
MIN_CONTOUR_AREA = 100  # Minimum contour area to consider
MIN_EDGE_DENSITY = 0.01  # Minimum edge density (edges/total_pixels)
MIN_CONTOUR_COUNT = 5  # Minimum number of contours for diagram
MIN_SHAPE_VARIETY = 2  # Minimum number of different shape types

# Diagram Completeness
MIN_COMPLETENESS_SCORE = 0.6  # Minimum score (0-1) for diagram completeness
REQUIRE_STABILITY = True  # Must have temporal stability
REQUIRE_STRUCTURE = True  # Must have diagram-like structure
MIN_CONTENT_COVERAGE = 0.15  # Minimum content coverage of board area

# Deduplication
DEDUPLICATION_ENABLED = True
SIMILARITY_THRESHOLD_PHASH = 5  # Perceptual hash threshold (0-64, lower = more strict)
SIMILARITY_THRESHOLD_SSIM = 0.90  # SSIM threshold for similarity (0-1, higher = more strict)
SIMILARITY_THRESHOLD_HIST = 0.85  # Histogram correlation threshold (0-1)

# Output
OUTPUT_DIR = "extracted_diagrams8"
OUTPUT_IMAGE_FORMAT = "PNG"  # PNG or JPEG
OUTPUT_IMAGE_QUALITY = 98  # For JPEG (1-100); higher preserves cropped diagram quality
METADATA_FILENAME = "metadata.json"
SAVE_ORIGINAL_FRAMES = False  # Save original frames for debugging

# Processing
MAX_FRAMES_TO_PROCESS = None  # None = process all frames
PROCESSING_BATCH_SIZE = 100  # Process frames in batches
VERBOSE = True  # Print progress information

# Transcription
ENABLE_TRANSCRIPTION = True  # Enable/disable transcription
TRANSCRIPTION_LANGUAGE = None  # Auto-detect if None, or specify language code (e.g., 'en', 'es', 'fr')
TRANSLATE_TO_ENGLISH = True  # Auto-translate to English if not already in English
TRANSCRIPTION_CHUNK_SECONDS = 600.0  # When audio > ~25MB, split into ~10min chunks for Whisper

# Diagram Enhancement
ENABLE_DIAGRAM_ENHANCEMENT = True  # Enable/disable diagram enhancement
DIAGRAM_CONTEXT_WINDOW = 120.0  # Context window in seconds (default: 120 = 2 minutes)
# Diagram classification and cropping use an OpenAI vision-capable chat model.
# gpt-4.1 supports vision via the chat.completions API with image_url content.
OPENAI_VISION_MODEL = "gpt-4.1"  # Vision model for classification and crop (e.g. gpt-4.1, gpt-4o)
CROP_MAX_ATTEMPTS = 3  # Max attempts to get a validated crop for digital images
# Optional: folder containing few-shot crop examples (Original.png, Cropped.png, Original2.png, Cropped2.png).
# By default we used "assets" under the project, but your current examples live in the
# Cursor project assets folder, so we point there explicitly.
# If you move the images into this repo's ./assets folder instead, change this back to "assets".
#CROP_FEWSHOT_DIR = r"C:\Users\DELL\.cursor\projects\f-v4-Image-Processing-v2-0\assets"

# Summary Generation (optional; not used when generating two PDFs: transcript + explanation)
GENERATE_SUMMARY = False
SUMMARY_SECTION_DURATION = 600.0  # Duration of each section for section summaries in seconds

# Output documents (HTML + DOC + enhanced images folder; no PDF)
GENERATE_OUTPUT_DOCUMENTS = True  # Generate 2 HTML, 2 DOC, and use enhanced_diagrams folder
HTML_TRANSCRIPT_FILENAME = "lecture_transcript.html"
HTML_EXPLANATION_FILENAME = "lecture_explanation.html"
DOC_TRANSCRIPT_FILENAME = "lecture_transcript.docx"
DOC_EXPLANATION_FILENAME = "lecture_explanation.docx"
ENHANCED_DIAGRAMS_FOLDER_NAME = "enhanced_diagrams"  # Subfolder of output dir
