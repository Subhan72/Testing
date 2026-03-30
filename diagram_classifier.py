"""
Diagram classifier and crop advisor using OpenAI (GPT-4o).
Classifies each extracted diagram as: hand_drawn, digital, or not_diagram.
Classification uses GPT-4o only. For digital images, crop region and validation use GPT-4o.
"""
import base64
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import cv2

from api_config import APIConfig
import config


def _find_crop_fewshot_paths() -> Optional[List[Tuple[str, str]]]:
    """Return a list of (original, cropped) example pairs if any exist; otherwise None.

    For your current setup, examples are expected to live next to the project
    files themselves, e.g.:

        F:\\v4\\Image Processing v2.0\\Original.png
        F:\\v4\\Image Processing v2.0\\Cropped.png

    Zero-shot cropping is disabled; if no example pairs are found, the caller
    should treat cropping as unavailable.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Known filename patterns for example pairs.
    pair_candidates = [
        ("Original.png", "Cropped.png"),
        ("Original2.png", "Cropped2.png"),
    ]

    pairs: List[Tuple[str, str]] = []
    for orig_name, crop_name in pair_candidates:
        orig_path = os.path.join(base_dir, orig_name)
        crop_path = os.path.join(base_dir, crop_name)
        if os.path.isfile(orig_path) and os.path.isfile(crop_path):
            pairs.append((orig_path, crop_path))

    if pairs:
        if getattr(config, "VERBOSE", False):
            print("✓ Using crop few-shot examples from project directory:", base_dir)
            for idx, (orig, cropped) in enumerate(pairs, start=1):
                print(f"  Pair {idx}: {os.path.basename(orig)} -> {os.path.basename(cropped)}")
        return pairs
    else:
        if getattr(config, "VERBOSE", False):
            print("✗ No crop few-shot example pairs found in project directory; cropping will fall back to original images.")
        return None


# OpenAI vision model for classification and cropping (gpt-4o)
OPENAI_VISION_MODEL = getattr(config, "OPENAI_VISION_MODEL", "gpt-4o")

# Max attempts to get a valid crop for digital images
CROP_MAX_ATTEMPTS = getattr(config, "CROP_MAX_ATTEMPTS", 3)


CLASSIFY_PROMPT = """You are a strict image classifier for lecture video frames. You will receive ONE image that was extracted from a lecture video.

**Your task:** Classify this image into exactly ONE category. Reply with only one of these three words—no explanation, no punctuation, no other text.

**Definitions (follow precisely):**

1. **HAND_DRAWN** — The image shows an actual diagram or figure that was physically drawn by hand on a board (chalk on blackboard, dry-erase on whiteboard, etc.). You see diagram-like content: shapes, equations, hand-drawn charts, sketches. The surface is clearly a physical board, not a screen.

2. **DIGITAL** — The image shows an actual diagram, chart, figure, or substantive visual content displayed digitally: a slide/screenshot that contains a diagram, graph, flowchart, illustration, schematic, or similar. It must have diagram-like content (structure, shapes, data viz), not just text or a title. It was NOT drawn by hand on a physical board.

3. **NOT_DIAGRAM** — The image does NOT contain a useful educational diagram. Classify as NOT_DIAGRAM for:
- Title screens, title slides, splash screens, intro/outro screens (e.g. "Lecture 1", course title, presenter name only)
- Slides or screens that are only text, a title, or branding with no diagram, chart, figure, or illustration
- Only people or the presenter, an empty board, a random scene, close-up of a face, irrelevant content, or blur/no content
- Any frame where there is no actual diagram, graph, flowchart, or figure—just text/title/layout does not count as a diagram

**Rules:**
- Output exactly one word: HAND_DRAWN, DIGITAL, or NOT_DIAGRAM.
- Nothing else: no period, no "Category:", no explanation.
- Title screen or title-only slide = NOT_DIAGRAM. DIGITAL requires actual diagram/figure/chart content on screen, not just that it looks like a slide.
- When in doubt whether there is a real diagram (vs. just text/title), prefer NOT_DIAGRAM."""


CROP_PROMPT = """You are a crop advisor. You will receive ONE image that may be a slide, screen share, or document with an overlaid UI (e.g. Zoom / Google Meet / Teams).

Your job is to choose a rectangular crop that:
- **Keeps** the full diagram or figure and all text that is logically part of it (titles, axis labels, legends, callout boxes, “Actions Taken” boxes, etc.).
- **Removes** people and their panels (participant videos/avatars/names), no matter whether they appear on the left, right, top, or bottom.
- **Optionally removes** other UI chrome (chat panels, toolbars, OS bars) only when that can be done without cutting into the diagram or its important text.

Do NOT make any assumptions about background color. The slide/diagram might be on a white, dark, or colored background. Instead, reason about content:

**What to KEEP (inside the crop):**
- The entire diagram or figure, including any title and content that clearly belongs to the diagram (legends, labels, captions directly describing the figure).
- The **full slide/diagram title at the top**—every letter must be visible; the top crop edge must not touch or cut any part of it.

**Top edge (y1_pct) — critical:**
- Do **not** crop from the top unless there is clearly a **participant row** (faces/names in a horizontal strip) or **OS/app chrome** (window title bar, meeting toolbar) **above** the slide content. If the top of the image is already the slide (e.g. the diagram title), leave it as-is: use y1_pct = 2% (or at most 3–4%) so the title is never cut.
- When in doubt, use y1_pct = 1. Unnecessary cropping from the top is wrong; it is not required.
- A small amount of margin around the rest of the diagram is acceptable; keep the crop precise on the sides and bottom, but preserve the full top including the title.

**What to REMOVE (outside the crop) when possible:**
- Any panel or strip that mainly contains human faces, avatars, or name labels (participant columns/rows), regardless of which edge they are on.
- UI elements that are clearly separate from the diagram content (chat sidebar, participant list, bottom playback bar, window title bar) when trimming them does not cut any part of the diagram or its key text.
- When trimming away a participant/people strip that sits beside the diagram, place the crop boundary just *outside* the outer edge of the diagram content. Do not move the crop boundary into arrows, icons, or text that belong to the diagram even if they are very close to the participant panel.

**How to locate the panel boundary:**
- Look for the visual boundary (a clear line or change in content) between the diagram region and the participant/people panel. This is often where repeated faces/avatars stop and the diagram or slide content begins.
- Place the crop edge exactly on that boundary: the panel must be completely outside the crop, and the diagram region completely inside.

**Safety rule:**
- If you are uncertain whether some text or area belongs to the diagram, prefer to KEEP it rather than risk cutting off part of the diagram. Around the side where a participant/people panel existed, be especially careful not to trim too far inward—leave a narrow margin between the last diagram element and the crop edge. It is better to keep a narrow strip of extra background than to crop away important content.

**Output format (very important):**
- Reply with exactly one JSON object and nothing else. Do NOT wrap it in markdown or a code block.
- The JSON must have exactly these four integer keys (percentages 0–100 of image width and height):
  - "x1_pct": left edge of the crop region (0 = leftmost)
  - "y1_pct": top edge of the crop region (0 = top)
  - "x2_pct": right edge of the crop region (100 = rightmost)
  - "y2_pct": bottom edge of the crop region (100 = bottom)
- Example (when no strip above the slide): {"x1_pct": 5, "y1_pct": 0, "x2_pct": 95, "y2_pct": 92}"""


# Few-shot crop prompt when example images are provided.
# We may send one or more example pairs (Original, Cropped) followed by the target image.
#CROP_PROMPT_FEWSHOT = """You are a crop advisor. You will see a small sequence of images.

#First, you are shown one or more **example pairs**:
#- In each pair, Image A is the original capture (slide + people / UI / borders) and Image B is the desired crop.
#- Notice what is kept in Image B:
#  * The whole diagram area (for example: the entire slide graphic or schematic), including its title and any important labels or callout boxes.
#  * The “Actions Taken”/legend boxes and similar elements that belong to the diagram.
#- Notice what is removed in Image B:
#  * The right-side column of participants / video feeds.
#  * Bottom and top OS / application chrome (taskbars, window title bars, playback controls) when they are clearly outside the slide/diagram.
#  * Empty black/gray borders around the screen that do not contain diagram content.

#The **last image** is the one you must crop. Your job is to infer a crop region that would transform this last image in the **same way** as in the examples:
#- Treat the main slide/diagram area (including its title and key text) as one block that must remain fully visible.
#- Remove side strips with people/video feeds and obvious UI panels, as in the examples.
#- Remove bottom/top bars only when doing so does **not** cut into the slide/diagram area or its important text.
#- It is better to keep a bit of extra margin or stray text than to cut even a small part of the diagram or its labels.

#**Rules:**
#- Reply with exactly one JSON object and nothing else. No markdown, no explanation.
#- Keys (integer percentages 0–100): "x1_pct", "y1_pct", "x2_pct", "y2_pct".
#- Match the visual pattern shown by the example cropped images: a tight but safe frame around the diagram + its important text, with people and side chrome removed.
#- Example: {"x1_pct": 8, "y1_pct": 5, "x2_pct": 92, "y2_pct": 88}"""


VALIDATE_CROP_PROMPT = """You will be shown a **cropped** image and must decide if this crop is acceptable.

The original image may have been a slide, screen share, or document with overlaid UI (video tiles, chat, toolbars, etc.). The cropping rules are:

**Black stripes (mandatory check):** Look at the **very top edge** and the **very bottom edge** of the image. If there is **any** thin black, dark, or empty stripe (screen-sharing/letterbox bar)—even one or two pixels high—at the top or at the bottom, you **must** answer NO. The topmost and bottommost rows must show slide/diagram content or slide background, not a black or empty band. If you see a black bar above or below the slide, the crop is bad.

**A good crop (answer YES) must:**
1. **Pass the black-stripe check above:** No black or empty stripe at the top; no black or empty stripe at the bottom.
2. Contain the **entire diagram or figure** and all important text that belongs to it:
   - Titles that clearly label the diagram or the slide for that diagram. The topmost title line should be fully visible; there must be a noticeable band of background above it so it is not touching or cut by the crop edge in any place (no partially missing letters).
   - Axis labels, legends, callout boxes, and “Actions Taken” or similar content that explains the figure.
   - None of these elements should be visibly cut off at the edges of the crop.
3. Remove visible **people / participant panels** wherever they might have been (left, right, top, or bottom):
   - No faces, avatars, or participant name tiles should be visible.
4. Optionally remove other UI chrome (chat sidebars, toolbars, OS bars), **but only if** they can be removed without cutting into the diagram or its important text.
5. It is fine if the crop includes some extra margin or small amounts of unrelated text, as long as the diagram and its key text are clearly intact, but prefer a **precise** crop that does not include large empty or irrelevant areas when you can avoid it. Near the edge where a participant/people panel used to be, there should still be a comfortable gap between the crop boundary and the closest diagram elements (no labels or arrows “cut right at the edge”), and the participant/people panel should be completely outside the crop.

**A bad crop (answer NO) includes any of:**
- **Any black or dark empty stripe at the top or bottom** of the image—answer NO. Even a thin line counts.
- People, faces, avatars, or participant tiles still visible in the image.
- Any part of the diagram or its main associated text (titles, legends, labels, key boxes) obviously cut off by the crop, **especially if any characters of the top title line are missing or touch the top crop edge**, or along the side where a participant/people panel was removed or where a strong panel/diagram boundary is visible.
- Large areas of UI chrome or side panels dominating the image relative to the diagram (e.g., a big chat/participant panel or toolbar that was not trimmed when it could be).

Be conservative about cutting: it is better for a crop to be slightly larger but keep all of the diagram and its labels than to be tight and remove important content.

Reply with exactly one word: YES or NO. Nothing else."""


def _extract_crop_json(text: str) -> Optional[Dict[str, int]]:
    """Extract crop region JSON from model output. Tolerates markdown and truncated output."""
    if not text or not text.strip():
        return None
    # Strip markdown code fences and extra whitespace
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```\s*$", "", text)
    text = text.strip()
    # Find first { and then matching } by brace count
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    end = -1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end == -1:
        return None
    json_str = text[start : end + 1]
    try:
        data = json.loads(json_str)
        x1 = max(0, min(100, int(data.get("x1_pct", 0))))
        y1 = max(0, min(100, int(data.get("y1_pct", 0))))
        x2 = max(0, min(100, int(data.get("x2_pct", 100))))
        y2 = max(0, min(100, int(data.get("y2_pct", 100))))
        if x1 >= x2 or y1 >= y2:
            return None
        return {"x1_pct": x1, "y1_pct": y1, "x2_pct": x2, "y2_pct": y2}
    except (json.JSONDecodeError, TypeError, ValueError, KeyError):
        return None


class DiagramClassifier:
    """Classify diagrams (hand_drawn / digital / not_diagram) and get crop region for digital. Uses GPT-4o only."""

    def __init__(self):
        self._openai_client = None
        self._openai_model = OPENAI_VISION_MODEL

    def _get_openai_client(self):
        if self._openai_client is None:
            try:
                from openai import OpenAI
                key = APIConfig.get_openai_api_key()
                self._openai_client = OpenAI(api_key=key)
            except Exception as e:
                if config.VERBOSE:
                    print(f"OpenAI client not available for verification: {e}")
        return self._openai_client

    def _classify_with_openai(self, image_path: str) -> Optional[str]:
        """Classify using OpenAI vision. Returns hand_drawn, digital, not_diagram, or None on failure."""
        client = self._get_openai_client()
        if client is None:
            return None
        try:
            with open(image_path, "rb") as f:
                b64 = base64.standard_b64encode(f.read()).decode("ascii")
            mime = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"
            url = f"data:{mime};base64,{b64}"
            resp = client.chat.completions.create(
                model=self._openai_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": CLASSIFY_PROMPT},
                            {"type": "image_url", "image_url": {"url": url}},
                        ],
                    }
                ],
                max_tokens=32,
                temperature=0.1,
            )
            text = (resp.choices[0].message.content or "").strip().upper()
            for token in text.replace(".", " ").split():
                if token == "HAND_DRAWN":
                    return "hand_drawn"
                if token == "DIGITAL":
                    return "digital"
                if token == "NOT_DIAGRAM":
                    return "not_diagram"
            return "hand_drawn"
        except Exception as e:
            if config.VERBOSE:
                print(f"  OpenAI classification failed: {e}")
            return None

    def classify(self, image_path: str) -> str:
        """
        Classify image as hand_drawn, digital, or not_diagram.
        Uses GPT-4o only.
        """
        if not image_path or not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        result = self._classify_with_openai(image_path)
        if result is not None:
            return result
        if config.VERBOSE:
            print("  GPT classification failed; defaulting to hand_drawn.")
        return "hand_drawn"

    def get_crop_region(self, image_path: str) -> Optional[Dict[str, int]]:
        """
        Get crop region for a digital image as percentages (0–100) using
        a zero-shot prompt with GPT-4o.

        The model receives a single image (the frame being cropped) together
        with the CROP_PROMPT instructions. No few-shot examples are used.
        """
        client = self._get_openai_client()
        if client is None:
            raise RuntimeError("OpenAI client not initialized. Check API key.")
        if not image_path or not os.path.exists(image_path):
            return None

        try:
            with open(image_path, "rb") as f:
                b64 = base64.standard_b64encode(f.read()).decode("ascii")
        except OSError:
            return None
        mime = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"
        url = f"data:{mime};base64,{b64}"

        try:
            resp = client.chat.completions.create(
                model=self._openai_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": CROP_PROMPT},
                            {"type": "image_url", "image_url": {"url": url}},
                        ],
                    }
                ],
                max_tokens=256,
                temperature=0.1,
            )
            text = (resp.choices[0].message.content or "").strip()
            region = _extract_crop_json(text)
            if region is None and config.VERBOSE:
                print(f"  Crop region parse failed: {text[:200]}")
            return region
        except Exception as e:
            if config.VERBOSE:
                print(f"  get_crop_region failed: {e}")
            return None

    def validate_crop(self, cropped_image_path: str) -> bool:
        """Ask GPT-4o if the cropped image is correct (only diagram, no chrome). Returns True if YES.

        If few-shot crop examples are available, we also show the model those
        example cropped images so it can better judge whether the candidate
        crop matches the desired style.
        """
        client = self._get_openai_client()
        if client is None or not os.path.exists(cropped_image_path):
            return False
        try:
            def _image_url(path: str) -> str:
                with open(path, "rb") as f:
                    b64 = base64.standard_b64encode(f.read()).decode("ascii")
                mime_local = "image/png" if path.lower().endswith(".png") else "image/jpeg"
                return f"data:{mime_local};base64,{b64}"

            target_url = _image_url(cropped_image_path)

            pairs = _find_crop_fewshot_paths()
            content: List[Dict] = []
            if pairs:
                # Use only the cropped examples here: show what "good" crops look like.
                content.append({"type": "text", "text": VALIDATE_CROP_PROMPT})
                for _orig_path, crop_path in pairs:
                    try:
                        content.append({"type": "image_url", "image_url": {"url": _image_url(crop_path)}})
                    except OSError:
                        # If any example fails to load, fall back to single-image validation.
                        content = []
                        break
                if content:
                    content.append({"type": "image_url", "image_url": {"url": target_url}})
                    if config.VERBOSE:
                        print("  Using few-shot crop examples for validation.")
                else:
                    content = [
                        {"type": "text", "text": VALIDATE_CROP_PROMPT},
                        {"type": "image_url", "image_url": {"url": target_url}},
                    ]
            else:
                content = [
                    {"type": "text", "text": VALIDATE_CROP_PROMPT},
                    {"type": "image_url", "image_url": {"url": target_url}},
                ]

            resp = client.chat.completions.create(
                model=self._openai_model,
                messages=[{"role": "user", "content": content}],
                max_tokens=16,
                temperature=0.1,
            )
            text = (resp.choices[0].message.content or "").strip().upper()
            return "YES" in text.split()
        except Exception as e:
            if config.VERBOSE:
                print(f"  validate_crop failed: {e}")
            return False

    def crop_digital_until_correct(
        self,
        image_path: str,
        output_path: str,
        max_attempts: int = None,
    ) -> bool:
        """
        For a digital image: get crop region, crop, validate. Loop until validation passes or max_attempts.
        Returns True if a valid crop was saved to output_path.
        """
        max_attempts = max_attempts or CROP_MAX_ATTEMPTS
        had_successful_crop = False
        for attempt in range(max_attempts):
            region = self.get_crop_region(image_path)
            if not region:
                if config.VERBOSE:
                    print(f"  Crop attempt {attempt + 1}: no region returned")
                continue
            if not self.crop_image(image_path, region, output_path):
                continue
            had_successful_crop = True
            if self.validate_crop(output_path):
                if config.VERBOSE:
                    print(f"  Crop validated on attempt {attempt + 1}")
                return True
            if config.VERBOSE:
                print(f"  Crop attempt {attempt + 1}: validation failed, retrying...")
        # Use the last applied crop if we got one, even when validation never said YES
        if had_successful_crop and config.VERBOSE:
            print(f"  Using crop after {max_attempts} attempts (validation did not pass).")
        return had_successful_crop

    @staticmethod
    def crop_image(
        image_path: str,
        region: Dict[str, int],
        output_path: str,
    ) -> bool:
        """
        Crop image by percentage region and save to output_path.
        region: {x1_pct, y1_pct, x2_pct, y2_pct} (0–100).
        Returns True on success.
        """
        if not os.path.exists(image_path):
            return False
        img = cv2.imread(image_path)
        if img is None:
            return False
        h, w = img.shape[:2]
        x1 = int(w * region["x1_pct"] / 100)
        y1 = int(h * region["y1_pct"] / 100)
        x2 = int(w * region["x2_pct"] / 100)
        y2 = int(h * region["y2_pct"] / 100)
        x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
        y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
        if x1 >= x2 or y1 >= y2:
            return False
        cropped = img[y1:y2, x1:x2]
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        if output_path.lower().endswith(".png"):
            # Use no compression to preserve full resolution and avoid artifacts
            cv2.imwrite(output_path, cropped, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        else:
            cv2.imwrite(
                output_path,
                cropped,
                [cv2.IMWRITE_JPEG_QUALITY, getattr(config, "OUTPUT_IMAGE_QUALITY", 98)],
            )
        return os.path.exists(output_path)

