# Diagram Detection System for Lecture Videos

A robust system to automatically detect and extract complete diagrams from lecture videos (blackboard/whiteboard or directly shown diagrams) using computer vision techniques.

## Features

- **Temporal Stability Detection**: Detects when diagrams are complete by analyzing frame-to-frame changes
- **Board Detection**: Automatically detects and extracts blackboard/whiteboard regions
- **Content Analysis**: Analyzes content structure to identify diagram-like features (edges, contours, geometric shapes)
- **Completeness Validation**: Multi-criteria validation to ensure only complete diagrams are extracted
- **Deduplication**: Removes duplicate diagrams using perceptual hashing and similarity metrics
- **Metadata Tracking**: Saves detailed metadata for each extracted diagram

## Requirements

- Python 3.8 or higher
- OpenCV (via ffmpeg) for video processing
- **FFmpeg** – needed for YouTube downloads when the video uses separate video/audio streams (merge). Install and ensure `ffmpeg` is on your PATH for best compatibility.
- See `requirements.txt` for full list of dependencies

## Installation

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python main.py video.mp4
```

This will:
- Process the video file
- Extract complete diagrams
- Save them to `extracted_diagrams/` directory
- Generate `metadata.json` with extraction details

### Custom Output Directory

```bash
python main.py video.mp4 --output my_diagrams
```

### Supported Video Formats

The system supports all video formats that OpenCV can read (via ffmpeg), including:
- MP4, AVI, MOV, MKV, FLV, and more

## Configuration

Edit `config.py` to adjust detection parameters:

- **Frame Extraction**: `FRAME_EXTRACTION_INTERVAL` - How often to extract frames (seconds)
- **Stability Detection**: `STABILITY_WINDOW_DURATION` - Duration of stability required (seconds)
- **Content Analysis**: Thresholds for edge density, contour count, shape variety
- **Deduplication**: Similarity thresholds for duplicate detection
- **Output**: Image format, quality, and directory settings

## How It Works

1. **Frame Extraction**: Extracts frames from video at regular intervals
2. **Board Detection**: Identifies blackboard/whiteboard regions in each frame
3. **Temporal Stability**: Tracks frame-to-frame changes to detect when content stabilizes
4. **Content Analysis**: Analyzes content structure (edges, contours, geometric shapes) to identify diagrams
5. **Completeness Check**: Validates that diagrams are complete using multiple criteria
6. **Deduplication**: Compares with previously extracted diagrams to avoid duplicates
7. **Output**: Saves unique complete diagrams with metadata

## Output Structure

```
extracted_diagrams/
├── diagram_0001.png
├── diagram_0002.png
├── diagram_0003.png
└── metadata.json
```

### Metadata Format

The `metadata.json` file contains:
- Diagram ID and filename
- Timestamp and frame number in video
- Board type (blackboard/whiteboard)
- Completeness and stability scores
- Content analysis results (edge density, contour count, shape analysis)
- Extraction timestamp

## Algorithm Details

### Temporal Stability Detection
- Uses Structural Similarity Index (SSIM) and Mean Squared Error (MSE)
- Tracks frame differences over a time window
- Detects when content remains stable (no significant changes)

### Content Analysis
- **Edge Detection**: Canny edge detection to find structured content
- **Contour Analysis**: Identifies geometric shapes (circles, rectangles, lines)
- **Geometric Structures**: Uses Hough transforms to detect lines and circles
- **Complexity Scoring**: Calculates diagram complexity based on edge density, contour count, and shape variety

### Completeness Validation
- Requires temporal stability (content hasn't changed recently)
- Validates diagram-like structure (edges, contours, geometric features)
- Checks content coverage of board area
- Multi-criteria scoring system

### Deduplication
- Perceptual hashing (pHash) for fast comparison
- SSIM (Structural Similarity Index) for structural similarity
- Histogram correlation for color distribution similarity
- Configurable similarity thresholds

## Tips for Best Results

1. **Video Quality**: Higher quality videos produce better results
2. **Lighting**: Well-lit videos with clear board visibility work best
3. **Configuration**: Adjust thresholds in `config.py` based on your video characteristics
4. **Stability Window**: Increase `STABILITY_WINDOW_DURATION` for videos where diagrams are drawn slowly
5. **Frame Interval**: Decrease `FRAME_EXTRACTION_INTERVAL` for faster-moving content

## Troubleshooting

### No diagrams extracted
- Check if board detection is working (set `BOARD_DETECTION_ENABLED = False` to use full frame)
- Lower `MIN_COMPLETENESS_SCORE` threshold
- Adjust `MIN_EDGE_DENSITY` and `MIN_CONTOUR_COUNT` thresholds

### Too many false positives
- Increase `MIN_COMPLETENESS_SCORE` threshold
- Increase `STABILITY_WINDOW_DURATION`
- Adjust content analysis thresholds

### Missing diagrams
- Decrease `FRAME_EXTRACTION_INTERVAL` to process more frames
- Decrease `STABILITY_WINDOW_DURATION` for quickly shown diagrams
- Lower `MIN_COMPLETENESS_SCORE` threshold

## n8n workflow and “Call Pipeline API” SSL error

**ngrok / Cloudflare:** If you hit SSL/TLS errors or tunnel timeouts, see `N8N_NGROK_SETUP.md`. For **“Task execution timed out after 60 seconds”** (pipeline can run 1+ hour): use the **callback** flow so n8n never runs long—import **`workflow-callback-trigger.json`** and **`workflow-callback-receive.json`**, set `base_url` and `callback_url`, and the API will POST the PDF to your callback webhook when done. Alternatively, use **`workflow-async.json`** and increase the workflow’s execution timeout to 1–2 hours in n8n Settings.

If you see an error at the **Call Pipeline API** node like:

`error:0A000119:SSL routines:tls_get_more_records:decryption failed or bad record mac`

it usually means n8n is calling your FastAPI over HTTPS and something in the TLS/connection chain is wrong (tunnel, proxy, or certificate).

### What to do

1. **Use the right URL**
   - **n8n on your machine** (self‑hosted): use `http://localhost:8000/process-youtube` so the call stays on HTTP and avoids SSL.
   - **n8n Cloud** (e.g. `*.app.n8n.cloud`): `localhost` is the cloud server, not your PC. Use a **public URL** to your API (tunnel or deployed server), e.g. `https://your-tunnel.ngrok.io/process-youtube` or `https://your-server.com/process-youtube`.

2. **Try “Allow Unauthorized Certificates” (for testing only)**
   - Open the **Call Pipeline API** node.
   - In options, enable **“Allow Unauthorized Certificates”** (or “Ignore SSL issues”).
   - Re-run the workflow.  
   - Use this only for testing; in production use proper HTTPS and valid certs.

3. **If you use a tunnel (ngrok, Cloudflare Tunnel, etc.)**
   - Prefer **HTTPS** and the URL they give you (e.g. `https://abc123.ngrok.io`).
   - If the SSL error persists, try another tunnel or a small VPS with a real certificate.

4. **Large PDF and timeouts**
   - The node timeout is set to 600000 ms (10 minutes). If the PDF is big and the request is slow, ensure your tunnel/server does not cut the connection early.

The `workflow.json` in this repo sets `allowUnauthorizedCerts: true` in the Call Pipeline API node options so that self‑signed or bad certs don’t cause the SSL error during testing. Replace the node’s URL with your real public API URL when using n8n Cloud.

## License

This project is provided as-is for educational and research purposes.

## Contributing

Feel free to submit issues or pull requests for improvements.
