#!/usr/bin/env python3
"""
run4.py — Furniture Segmentation + IMPROVED Color Detection

Fixed color detection issues:
1) Uses LAB color space for perceptually accurate color matching
2) Better handling of wood tones, fabrics, and mixed materials
3) Excludes shadows and highlights from color analysis
4) Weighted color extraction focusing on furniture surface, not edges
5) Improved color naming with furniture-specific vocabulary

Models (all open source):
- GroundingDINO (Apache-2.0) - detection
- SAM2 (Apache-2.0) - segmentation
- BLIP (BSD-3-Clause) - captioning

Usage:
    python run4.py --image room.jpg --out result.png
    python run4.py --image room.jpg --out result.png --fast
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from collections import Counter
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
    Sam2Processor,
    Sam2Model,
    BlipProcessor,
    BlipForConditionalGeneration,
)

# ==================== CONFIGURATION ====================

GROUNDING_DINO_ID = "IDEA-Research/grounding-dino-base"
SAM2_ID = "facebook/sam2-hiera-base-plus"
BLIP_ID = "Salesforce/blip-image-captioning-base"
BLIP_LARGE_ID = "Salesforce/blip-image-captioning-large"

FURNITURE_TERMS = [
    "sofa",
    "couch",
    "armchair",
    "chair",
    "ottoman",
    "bench",
    "coffee table",
    "side table",
    "dining table",
    "desk",
    "tv stand",
    "bookcase",
    "shelf",
    "cabinet",
    "dresser",
    "bed",
    "nightstand",
    "lamp",
    "floor lamp",
    "table lamp",
    "rug",
    "carpet",
    "curtain",
    "mirror",
    "artwork",
    "plant",
    "potted plant",
]

BOX_THRESHOLD = 0.25
MAX_DETECTIONS = 25
MIN_MASK_AREA_RATIO = 0.001

# ==================== IMPROVED COLOR SYSTEM ====================

# Comprehensive color database with LAB values for accurate matching
# Format: (name, RGB, category, descriptive_terms)
COLOR_DATABASE = [
    # === WHITES & OFF-WHITES ===
    ("pure white", (255, 255, 255), "white", ["bright", "clean"]),
    ("off-white", (250, 249, 246), "white", ["soft", "warm"]),
    ("ivory", (255, 255, 240), "white", ["creamy", "elegant"]),
    ("cream", (255, 253, 208), "cream", ["warm", "soft"]),
    ("eggshell", (240, 234, 214), "cream", ["neutral", "subtle"]),
    ("linen", (250, 240, 230), "cream", ["natural", "light"]),
    # === GRAYS ===
    ("white gray", (220, 220, 220), "gray", ["light", "soft"]),
    ("light gray", (192, 192, 192), "gray", ["neutral", "modern"]),
    ("silver", (192, 192, 192), "gray", ["metallic", "cool"]),
    ("medium gray", (150, 150, 150), "gray", ["balanced", "neutral"]),
    ("gray", (128, 128, 128), "gray", ["classic", "neutral"]),
    ("dark gray", (96, 96, 96), "gray", ["sophisticated", "deep"]),
    ("charcoal", (54, 69, 79), "gray", ["dark", "rich"]),
    ("slate gray", (112, 128, 144), "gray", ["cool", "blue-tinted"]),
    ("warm gray", (140, 132, 125), "gray", ["taupe-tinted", "cozy"]),
    # === BLACKS ===
    ("black", (0, 0, 0), "black", ["bold", "dramatic"]),
    ("jet black", (20, 20, 20), "black", ["deep", "solid"]),
    ("soft black", (40, 40, 40), "black", ["muted", "subtle"]),
    # === BROWNS & WOOD TONES ===
    ("light oak", (200, 170, 120), "brown", ["natural", "light wood"]),
    ("honey oak", (185, 145, 90), "brown", ["warm", "golden wood"]),
    ("natural oak", (175, 145, 100), "brown", ["classic", "wood"]),
    ("golden oak", (190, 150, 80), "brown", ["rich", "warm wood"]),
    ("medium oak", (160, 120, 75), "brown", ["traditional", "wood"]),
    ("light walnut", (140, 100, 70), "brown", ["medium", "wood"]),
    ("walnut", (94, 69, 50), "brown", ["rich", "dark wood"]),
    ("dark walnut", (70, 50, 35), "brown", ["deep", "elegant wood"]),
    ("espresso", (60, 40, 30), "brown", ["very dark", "coffee"]),
    ("mahogany", (103, 56, 41), "brown", ["reddish", "luxury wood"]),
    ("cherry wood", (120, 60, 45), "brown", ["red-tinted", "warm wood"]),
    ("teak", (140, 105, 60), "brown", ["golden", "tropical wood"]),
    ("pine", (195, 170, 120), "brown", ["light", "natural wood"]),
    ("birch", (210, 190, 150), "brown", ["pale", "light wood"]),
    # === BEIGES & TANS ===
    ("light beige", (245, 235, 220), "beige", ["soft", "neutral"]),
    ("beige", (230, 215, 195), "beige", ["classic", "warm neutral"]),
    ("warm beige", (225, 200, 170), "beige", ["cozy", "inviting"]),
    ("sand", (210, 190, 160), "beige", ["natural", "earthy"]),
    ("tan", (210, 180, 140), "tan", ["medium", "warm"]),
    ("camel", (193, 154, 107), "tan", ["rich", "classic"]),
    ("khaki", (195, 176, 145), "tan", ["muted", "casual"]),
    ("taupe", (130, 115, 100), "taupe", ["gray-brown", "sophisticated"]),
    ("dark taupe", (100, 85, 70), "taupe", ["deep", "rich"]),
    ("greige", (170, 165, 155), "taupe", ["gray-beige", "modern"]),
    # === BLUES ===
    ("powder blue", (176, 224, 230), "blue", ["soft", "light"]),
    ("sky blue", (135, 206, 235), "blue", ["bright", "airy"]),
    ("light blue", (173, 216, 230), "blue", ["gentle", "calming"]),
    ("baby blue", (137, 207, 240), "blue", ["soft", "delicate"]),
    ("steel blue", (70, 130, 180), "blue", ["cool", "industrial"]),
    ("medium blue", (80, 120, 180), "blue", ["classic", "balanced"]),
    ("denim blue", (100, 130, 170), "blue", ["casual", "relaxed"]),
    ("blue", (70, 130, 180), "blue", ["classic", "versatile"]),
    ("royal blue", (65, 105, 225), "blue", ["bold", "rich"]),
    ("cobalt blue", (0, 71, 171), "blue", ["vivid", "striking"]),
    ("navy blue", (0, 0, 128), "blue", ["dark", "classic"]),
    ("dark navy", (20, 30, 70), "blue", ["deep", "sophisticated"]),
    ("midnight blue", (25, 25, 112), "blue", ["very dark", "elegant"]),
    # === TEALS & TURQUOISE ===
    ("light teal", (150, 200, 200), "teal", ["soft", "refreshing"]),
    ("teal", (0, 128, 128), "teal", ["balanced", "sophisticated"]),
    ("dark teal", (0, 90, 90), "teal", ["deep", "rich"]),
    ("turquoise", (64, 224, 208), "teal", ["vibrant", "tropical"]),
    ("aqua", (127, 255, 212), "teal", ["bright", "fresh"]),
    ("seafoam", (160, 210, 200), "teal", ["soft", "coastal"]),
    # === GREENS ===
    ("mint green", (189, 252, 201), "green", ["fresh", "light"]),
    ("sage green", (176, 196, 164), "green", ["muted", "natural"]),
    ("light sage", (190, 210, 180), "green", ["soft", "calming"]),
    ("olive green", (128, 128, 0), "green", ["earthy", "warm"]),
    ("dark olive", (85, 85, 0), "green", ["deep", "military"]),
    ("moss green", (100, 120, 80), "green", ["natural", "forest"]),
    ("forest green", (34, 139, 34), "green", ["deep", "rich"]),
    ("hunter green", (53, 94, 59), "green", ["dark", "classic"]),
    ("emerald green", (0, 155, 119), "green", ["vibrant", "jewel"]),
    ("kelly green", (76, 187, 23), "green", ["bright", "bold"]),
    # === YELLOWS & GOLDS ===
    ("pale yellow", (255, 255, 200), "yellow", ["soft", "buttery"]),
    ("light yellow", (255, 250, 180), "yellow", ["warm", "sunny"]),
    ("butter yellow", (255, 240, 150), "yellow", ["creamy", "soft"]),
    ("yellow", (255, 230, 100), "yellow", ["bright", "cheerful"]),
    ("golden yellow", (255, 220, 80), "yellow", ["warm", "rich"]),
    ("mustard yellow", (255, 200, 60), "yellow", ["deep", "retro"]),
    ("gold", (212, 175, 55), "gold", ["metallic", "luxurious"]),
    ("antique gold", (180, 150, 70), "gold", ["vintage", "warm"]),
    ("brass", (181, 166, 66), "gold", ["metallic", "warm"]),
    # === ORANGES ===
    ("peach", (255, 218, 185), "orange", ["soft", "warm"]),
    ("apricot", (255, 200, 150), "orange", ["warm", "gentle"]),
    ("coral", (255, 127, 80), "orange", ["vibrant", "warm"]),
    ("salmon", (250, 128, 114), "orange", ["pink-orange", "soft"]),
    ("terra cotta", (204, 120, 92), "orange", ["earthy", "warm"]),
    ("burnt orange", (204, 85, 0), "orange", ["deep", "autumn"]),
    ("orange", (255, 165, 0), "orange", ["bright", "energetic"]),
    ("rust", (183, 65, 14), "orange", ["deep", "earthy"]),
    # === REDS ===
    ("light pink", (255, 220, 220), "pink", ["soft", "delicate"]),
    ("blush pink", (255, 200, 200), "pink", ["warm", "romantic"]),
    ("dusty pink", (210, 170, 170), "pink", ["muted", "vintage"]),
    ("rose", (255, 130, 150), "pink", ["medium", "feminine"]),
    ("pink", (255, 150, 180), "pink", ["classic", "soft"]),
    ("hot pink", (255, 105, 180), "pink", ["bold", "vibrant"]),
    ("light red", (255, 100, 100), "red", ["soft", "warm"]),
    ("coral red", (255, 80, 80), "red", ["vibrant", "warm"]),
    ("red", (220, 60, 60), "red", ["classic", "bold"]),
    ("cherry red", (200, 40, 60), "red", ["deep", "rich"]),
    ("crimson", (170, 30, 50), "red", ["dark", "elegant"]),
    ("burgundy", (128, 0, 32), "red", ["wine", "sophisticated"]),
    ("maroon", (100, 30, 40), "red", ["very dark", "rich"]),
    ("oxblood", (75, 25, 30), "red", ["deep", "vintage"]),
    # === PURPLES ===
    ("lavender", (230, 230, 250), "purple", ["soft", "light"]),
    ("light purple", (210, 200, 230), "purple", ["gentle", "feminine"]),
    ("lilac", (200, 162, 200), "purple", ["soft", "romantic"]),
    ("mauve", (200, 150, 180), "purple", ["dusty", "vintage"]),
    ("purple", (150, 100, 160), "purple", ["classic", "regal"]),
    ("violet", (130, 90, 160), "purple", ["rich", "vibrant"]),
    ("plum", (142, 69, 133), "purple", ["deep", "sophisticated"]),
    ("eggplant", (90, 50, 80), "purple", ["very dark", "elegant"]),
    ("dark purple", (70, 40, 90), "purple", ["deep", "dramatic"]),
]


def rgb_to_lab(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    """Convert RGB to LAB color space for perceptual color matching."""
    # Normalize RGB
    r, g, b = [x / 255.0 for x in rgb]

    # RGB to XYZ
    def pivot(n):
        return ((n + 0.055) / 1.055) ** 2.4 if n > 0.04045 else n / 12.92

    r, g, b = pivot(r), pivot(g), pivot(b)

    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    # XYZ to LAB (D65 illuminant)
    x, y, z = x / 0.95047, y / 1.0, z / 1.08883

    def f(t):
        return t ** (1 / 3) if t > 0.008856 else (7.787 * t) + (16 / 116)

    L = (116 * f(y)) - 16
    a = 500 * (f(x) - f(y))
    b_val = 200 * (f(y) - f(z))

    return (L, a, b_val)


def color_distance_lab(rgb1: Tuple[int, int, int], rgb2: Tuple[int, int, int]) -> float:
    """Calculate perceptual color distance using CIEDE2000-simplified (LAB Euclidean)."""
    lab1 = rgb_to_lab(rgb1)
    lab2 = rgb_to_lab(rgb2)

    # Weight L (lightness) less than a,b (color) for better hue matching
    dL = (lab1[0] - lab2[0]) * 0.8
    da = lab1[1] - lab2[1]
    db = lab1[2] - lab2[2]

    return (dL**2 + da**2 + db**2) ** 0.5


def get_color_name_lab(rgb: Tuple[int, int, int]) -> Tuple[str, str, List[str]]:
    """
    Get color name using LAB color space for accurate perceptual matching.
    Returns: (specific_name, category, descriptive_terms)
    """
    min_dist = float("inf")
    best_match = ("gray", "gray", ["neutral"])

    for name, ref_rgb, category, terms in COLOR_DATABASE:
        dist = color_distance_lab(rgb, ref_rgb)
        if dist < min_dist:
            min_dist = dist
            best_match = (name, category, terms)

    return best_match


def is_shadow_or_highlight(
    rgb: Tuple[int, int, int], threshold_dark: int = 30, threshold_bright: int = 245
) -> bool:
    """Check if pixel is likely a shadow or highlight to exclude from analysis."""
    r, g, b = rgb

    # Very dark (shadow)
    if max(rgb) < threshold_dark:
        return True

    # Very bright (highlight/reflection)
    if min(rgb) > threshold_bright:
        return True

    return False


def get_mask_center_weight(mask: np.ndarray) -> np.ndarray:
    """Create weight map that emphasizes center of mask (furniture surface vs edges)."""
    from scipy import ndimage

    # Calculate distance from mask edge
    distance = ndimage.distance_transform_edt(mask)

    # Normalize to 0-1
    if distance.max() > 0:
        weights = distance / distance.max()
    else:
        weights = np.ones_like(mask, dtype=float)

    # Apply power to emphasize center more
    weights = weights**0.5

    return weights


def extract_dominant_colors_improved(
    img_rgb: np.ndarray,
    mask: np.ndarray,
    n_colors: int = 5,
    use_center_weighting: bool = True,
) -> List[Tuple[Tuple[int, int, int], float]]:
    """
    Extract dominant colors with improved accuracy.
    Returns list of (RGB, percentage) tuples.
    """
    # Get masked pixels
    pixels = img_rgb[mask].copy()

    if len(pixels) == 0:
        return [((128, 128, 128), 100.0)]

    # Filter out shadows and highlights
    valid_mask = np.array([not is_shadow_or_highlight(tuple(p)) for p in pixels])
    pixels = pixels[valid_mask]

    if len(pixels) == 0:
        # Fall back to all pixels if filtering removed everything
        pixels = img_rgb[mask].copy()

    # Get center weights if enabled
    if use_center_weighting:
        try:
            weight_map = get_mask_center_weight(mask)
            pixel_weights = (
                weight_map[mask][valid_mask]
                if valid_mask.sum() > 0
                else weight_map[mask]
            )
        except:
            pixel_weights = np.ones(len(pixels))
    else:
        pixel_weights = np.ones(len(pixels))

    # Subsample for efficiency while respecting weights
    max_samples = 8000
    if len(pixels) > max_samples:
        # Weighted sampling
        weights_norm = pixel_weights / pixel_weights.sum()
        idx = np.random.choice(len(pixels), max_samples, replace=False, p=weights_norm)
        pixels = pixels[idx]
        pixel_weights = pixel_weights[idx]

    # Quantize colors (reduce to ~64 distinct colors)
    # Use finer quantization than before for better accuracy
    quantization = 24  # 256/24 ≈ 10 levels per channel
    quantized = ((pixels // quantization) * quantization + quantization // 2).astype(
        np.uint8
    )

    # Count colors with weights
    color_weights = {}
    for color, weight in zip(quantized, pixel_weights):
        key = tuple(color)
        color_weights[key] = color_weights.get(key, 0) + weight

    # Sort by weight
    sorted_colors = sorted(color_weights.items(), key=lambda x: x[1], reverse=True)

    # Calculate percentages
    total_weight = sum(w for _, w in sorted_colors)
    results = []

    for color, weight in sorted_colors[:n_colors]:
        pct = (weight / total_weight) * 100
        results.append((color, pct))

    return results


def merge_similar_colors(
    colors: List[Tuple[Tuple[int, int, int], float]], threshold: float = 15.0
) -> List[Tuple[Tuple[int, int, int], float, str, str]]:
    """
    Merge similar colors and add names.
    Returns: [(rgb, percentage, name, category), ...]
    """
    if not colors:
        return []

    merged = []
    used = set()

    for i, (rgb1, pct1) in enumerate(colors):
        if i in used:
            continue

        # Find similar colors
        combined_rgb = np.array(rgb1) * pct1
        combined_pct = pct1

        for j, (rgb2, pct2) in enumerate(colors[i + 1 :], i + 1):
            if j in used:
                continue

            dist = color_distance_lab(rgb1, rgb2)
            if dist < threshold:
                combined_rgb += np.array(rgb2) * pct2
                combined_pct += pct2
                used.add(j)

        # Average the RGB
        final_rgb = tuple(int(x) for x in combined_rgb / combined_pct)
        name, category, _ = get_color_name_lab(final_rgb)

        merged.append((final_rgb, combined_pct, name, category))
        used.add(i)

    # Sort by percentage
    merged.sort(key=lambda x: x[1], reverse=True)

    return merged


@dataclass
class ColorAnalysisResult:
    """Detailed color analysis result."""

    primary_color: str
    primary_category: str
    primary_rgb: Tuple[int, int, int]
    primary_percentage: float

    secondary_colors: List[str]
    secondary_rgbs: List[Tuple[int, int, int]]

    full_palette: List[Tuple[str, Tuple[int, int, int], float]]  # (name, rgb, pct)

    color_description: str
    is_multicolored: bool
    dominant_tone: str  # warm, cool, neutral


def analyze_colors_improved(
    img_rgb: np.ndarray, mask: np.ndarray
) -> ColorAnalysisResult:
    """
    Comprehensive color analysis with improved accuracy.
    """
    # Extract dominant colors
    raw_colors = extract_dominant_colors_improved(img_rgb, mask, n_colors=8)

    # Merge similar colors and get names
    merged = merge_similar_colors(raw_colors, threshold=18.0)

    if not merged:
        return ColorAnalysisResult(
            primary_color="gray",
            primary_category="gray",
            primary_rgb=(128, 128, 128),
            primary_percentage=100.0,
            secondary_colors=[],
            secondary_rgbs=[],
            full_palette=[("gray", (128, 128, 128), 100.0)],
            color_description="gray furniture",
            is_multicolored=False,
            dominant_tone="neutral",
        )

    # Primary color
    primary_rgb, primary_pct, primary_name, primary_cat = merged[0]

    # Secondary colors (different from primary)
    secondary = [
        (rgb, pct, name, cat)
        for rgb, pct, name, cat in merged[1:4]
        if cat != primary_cat or pct > 10
    ]

    secondary_colors = [name for _, _, name, _ in secondary]
    secondary_rgbs = [rgb for rgb, _, _, _ in secondary]

    # Full palette
    full_palette = [(name, rgb, pct) for rgb, pct, name, _ in merged[:5]]

    # Determine if multicolored
    is_multicolored = len(secondary) >= 2 or (
        len(secondary) >= 1 and secondary[0][1] > 20
    )

    # Determine dominant tone
    r, g, b = primary_rgb
    _, a, b_val = rgb_to_lab(primary_rgb)

    if abs(a) < 10 and abs(b_val) < 10:
        tone = "neutral"
    elif a > 5 or b_val > 10:
        tone = "warm"
    else:
        tone = "cool"

    # Generate description
    if is_multicolored:
        desc = f"{primary_name} with {', '.join(secondary_colors[:2])} accents"
    elif primary_pct > 85:
        desc = f"solid {primary_name}"
    else:
        desc = f"predominantly {primary_name}"

    return ColorAnalysisResult(
        primary_color=primary_name,
        primary_category=primary_cat,
        primary_rgb=primary_rgb,
        primary_percentage=round(primary_pct, 1),
        secondary_colors=secondary_colors,
        secondary_rgbs=secondary_rgbs,
        full_palette=full_palette,
        color_description=desc,
        is_multicolored=is_multicolored,
        dominant_tone=tone,
    )


# ==================== DATA STRUCTURES ====================


@dataclass
class FurnitureItem:
    id: int
    category: str
    confidence: float
    box: List[float]
    colors: ColorAnalysisResult
    size: str
    position: str
    area_pct: float
    material: str
    style: str
    description: str


# ==================== UTILITIES ====================


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def clamp_box(box: np.ndarray, w: int, h: int) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(float)
    x1, x2 = max(0, min(w - 1, x1)), max(0, min(w - 1, x2))
    y1, y2 = max(0, min(h - 1, y1)), max(0, min(h - 1, y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def safe_font(size: int = 14):
    for path in [
        "Arial.ttf",
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except:
            pass
    return ImageFont.load_default()


def analyze_size(box: np.ndarray, img_w: int, img_h: int) -> Tuple[str, str, float]:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    area_pct = (w * h) / (img_w * img_h) * 100

    ratio = max(w / img_w, h / img_h)
    size = "small" if ratio < 0.2 else ("medium" if ratio < 0.45 else "large")

    cx, cy = (x1 + x2) / 2 / img_w, (y1 + y2) / 2 / img_h
    h_pos = "left" if cx < 0.33 else ("right" if cx > 0.66 else "center")
    v_pos = "top" if cy < 0.33 else ("bottom" if cy > 0.66 else "middle")
    position = (
        f"{v_pos}-{h_pos}" if h_pos != "center" or v_pos != "middle" else "center"
    )

    return size, position, round(area_pct, 1)


# ==================== MODEL LOADING ====================


def load_detector(device):
    print("[1/3] Loading GroundingDINO...")
    proc = AutoProcessor.from_pretrained(GROUNDING_DINO_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_DINO_ID).to(
        device
    )
    model.eval()
    return proc, model


def load_segmenter(device):
    print("[2/3] Loading SAM2...")
    proc = Sam2Processor.from_pretrained(SAM2_ID)
    model = Sam2Model.from_pretrained(SAM2_ID).to(device)
    model.eval()
    return proc, model


def load_captioner(device, use_large: bool = False):
    model_id = BLIP_LARGE_ID if use_large else BLIP_ID
    print(f"[3/3] Loading BLIP ({'large' if use_large else 'base'})...")
    proc = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id).to(device)
    model.eval()
    return proc, model


# ==================== DETECTION & SEGMENTATION ====================


def detect(image, proc, model, device, threshold):
    prompt = ". ".join(FURNITURE_TERMS) + "."
    inputs = proc(images=image, text=prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = proc.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        target_sizes=torch.tensor([[image.height, image.width]], device=device),
        threshold=threshold,
    )[0]

    items = []
    boxes = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    labels = results.get("text_labels", results.get("labels", ["item"] * len(boxes)))

    for box, score, label in zip(boxes, scores, labels):
        items.append(
            {
                "label": str(label),
                "score": float(score),
                "box": clamp_box(np.array(box), image.width, image.height),
                "mask": None,
            }
        )

    items.sort(key=lambda x: x["score"], reverse=True)
    return items[:MAX_DETECTIONS]


def segment(image, items, proc, model, device):
    H, W = image.height, image.width

    for item in items:
        box = item["box"].tolist()
        inputs = proc(images=image, input_boxes=[[box]], return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs, multimask_output=True)

        try:
            masks = proc.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
            )[0]
        except TypeError:
            masks = proc.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu(),
            )[0]

        m = masks.numpy() if isinstance(masks, torch.Tensor) else masks
        m = np.squeeze(m)

        if m.ndim == 4:
            m = m[0]
        if m.ndim != 3:
            continue

        best_idx = 0
        if hasattr(outputs, "iou_scores") and outputs.iou_scores is not None:
            s = np.squeeze(outputs.iou_scores.cpu().numpy())
            if s.ndim == 2:
                s = s[0]
            if len(s) == m.shape[0]:
                best_idx = int(np.argmax(s))

        mask = m[best_idx]
        if mask.shape == (H, W) and mask.sum() / (H * W) > MIN_MASK_AREA_RATIO:
            item["mask"] = mask > 0

    return [it for it in items if it.get("mask") is not None]


# ==================== VLM ANALYSIS ====================


def caption_item(
    crop, category, colors: ColorAnalysisResult, proc, model, device
) -> Dict[str, str]:
    """Generate targeted captions with color context."""
    results = {}

    color_hint = colors.primary_color

    prompts = [
        (f"this {color_hint} {category} is made of", "material"),
        (f"the style of this {category} is", "style"),
        (f"a {color_hint} {category}", "description"),
    ]

    for prompt_text, key in prompts:
        inputs = proc(images=crop, text=prompt_text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=50, num_beams=2, do_sample=False
            )
        text = proc.decode(out[0], skip_special_tokens=True).strip()
        results[key] = text

    return results


def infer_attributes_heuristic(
    label: str, colors: ColorAnalysisResult
) -> Dict[str, str]:
    """Heuristic-based attribute inference using color info."""

    # Material hints based on color
    wood_colors = {
        "oak",
        "walnut",
        "mahogany",
        "cherry",
        "teak",
        "pine",
        "birch",
        "espresso",
        "brown",
    }
    fabric_colors = {"beige", "cream", "gray", "blue", "green", "red", "purple", "pink"}
    leather_colors = {"black", "brown", "tan", "burgundy", "cognac"}
    metal_colors = {"silver", "gold", "brass", "black", "white"}

    primary_lower = colors.primary_color.lower()
    category_lower = colors.primary_category.lower()

    # Determine material
    if any(
        w in primary_lower
        for w in ["oak", "walnut", "mahogany", "cherry", "teak", "pine", "birch"]
    ):
        material = f"{colors.primary_color} wood"
    elif (
        category_lower in ["brown", "tan", "taupe"]
        and "table" in label.lower()
        or "desk" in label.lower()
    ):
        material = "wood"
    elif category_lower in ["gray", "beige", "blue", "green"] and any(
        x in label.lower() for x in ["sofa", "chair", "couch"]
    ):
        material = f"{colors.primary_color} fabric upholstery"
    elif category_lower == "black" and any(
        x in label.lower() for x in ["sofa", "chair"]
    ):
        material = "leather or fabric"
    elif "metal" in primary_lower or "silver" in primary_lower:
        material = "metal"
    else:
        material = f"{colors.primary_color} finish"

    # Style based on color tone
    if colors.dominant_tone == "neutral":
        if colors.primary_category in ["gray", "white", "black"]:
            style = "modern/minimalist"
        else:
            style = "contemporary"
    elif colors.dominant_tone == "warm":
        if "wood" in material:
            style = "traditional/classic"
        else:
            style = "warm contemporary"
    else:
        style = "modern"

    return {
        "material": material,
        "style": style,
        "description": f"{colors.color_description} {label}",
    }


# ==================== VISUALIZATION ====================


def visualize(image, items: List[FurnitureItem], out_path: str):
    img = image.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = safe_font(14)

    for i, item in enumerate(items):
        # Use item's actual primary color for the box
        r, g, b = item.colors.primary_rgb
        # Ensure visibility by adjusting if too light
        if r + g + b > 600:
            r, g, b = max(0, r - 60), max(0, g - 60), max(0, b - 60)

        x1, y1, x2, y2 = item.box

        # Box
        draw.rectangle([x1, y1, x2, y2], outline=(r, g, b, 255), width=3)

        # Label
        label = f"{item.id}. {item.category} ({item.confidence:.0%})"
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        ly = max(0, y1 - th - 6)
        draw.rectangle([x1, ly, x1 + tw + 8, ly + th + 4], fill=(0, 0, 0, 220))
        draw.text((x1 + 4, ly + 2), label, fill=(255, 255, 255), font=font)

        # Color info line
        info = f"Color: {item.colors.primary_color}"
        if item.colors.secondary_colors:
            info += f" + {item.colors.secondary_colors[0]}"
        bbox2 = draw.textbbox((0, 0), info, font=font)
        tw2 = bbox2[2] - bbox2[0]
        draw.rectangle(
            [x1, ly + th + 6, x1 + tw2 + 8, ly + th + 6 + th + 4], fill=(r, g, b, 200)
        )
        # Text color based on background brightness
        text_color = (255, 255, 255) if (r + g + b) < 400 else (0, 0, 0)
        draw.text((x1 + 4, ly + th + 8), info, fill=text_color, font=font)

    out = Image.alpha_composite(img, overlay).convert("RGB")
    out.save(out_path, quality=95)
    print(f"[OK] Saved: {out_path}")


def print_report(items: List[FurnitureItem]):
    print("\n" + "=" * 65)
    print("FURNITURE ANALYSIS REPORT")
    print("=" * 65)
    print(f"Total items: {len(items)}\n")

    for item in items:
        print(
            f"[{item.id}] {item.category.upper()} (confidence: {item.confidence:.0%})"
        )
        print(
            f"    📍 Position: {item.position} | Size: {item.size} | Area: {item.area_pct}%"
        )
        print(
            f"    🎨 Primary Color: {item.colors.primary_color} (RGB: {item.colors.primary_rgb})"
        )
        print(f"       Percentage: {item.colors.primary_percentage:.0f}%")
        if item.colors.secondary_colors:
            sec_info = ", ".join(f"{c}" for c in item.colors.secondary_colors[:2])
            print(f"       Secondary: {sec_info}")
        print(f"       Tone: {item.colors.dominant_tone}")
        print(f"       Description: {item.colors.color_description}")
        print(f"    🪑 Material: {item.material}")
        print(f"    🎭 Style: {item.style}")
        print(f"    📝 Full: {item.description}")
        print()

    print("=" * 65)


# ==================== MAIN ====================


def main():
    parser = argparse.ArgumentParser(
        description="Furniture Analyzer v2 - Improved Color Detection"
    )
    parser.add_argument("--image", default="image.png", help="Input image")
    parser.add_argument("--out", default="output.png", help="Output image")
    parser.add_argument("--json", help="Output JSON file")
    parser.add_argument(
        "--threshold", type=float, default=0.25, help="Detection threshold"
    )
    parser.add_argument(
        "--max", type=int, default=MAX_DETECTIONS, help="Max detections"
    )
    parser.add_argument("--fast", action="store_true", help="Skip VLM (use heuristics)")
    parser.add_argument("--large", action="store_true", help="Use BLIP-large")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"ERROR: Image not found: {args.image}")
        sys.exit(1)

    device = pick_device()
    print(f"Device: {device}")
    print(f"Mode: {'Fast (heuristics)' if args.fast else 'Full (VLM)'}\n")

    # Load image
    img = Image.open(args.image).convert("RGB")
    img_np = np.array(img)
    print(f"Image: {img.width}x{img.height}")

    # Detection
    det_proc, det_model = load_detector(device)
    items = detect(img, det_proc, det_model, device, args.threshold)[: args.max]
    print(f"Detected: {len(items)} items")

    if not items:
        print("No furniture detected!")
        sys.exit(0)

    # Segmentation
    seg_proc, seg_model = load_segmenter(device)
    items = segment(img, items, seg_proc, seg_model, device)
    print(f"Segmented: {len(items)} items")

    # Free memory
    del det_proc, det_model, seg_proc, seg_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Load captioner if needed
    cap_proc, cap_model = None, None
    if not args.fast:
        cap_proc, cap_model = load_captioner(device, use_large=args.large)

    # Analyze
    print("\nAnalyzing items...")
    results = []

    for i, item in enumerate(items, 1):
        # Improved color analysis
        colors = analyze_colors_improved(img_np, item["mask"])

        # Size/position
        size, position, area_pct = analyze_size(item["box"], img.width, img.height)

        # Material/style analysis
        if args.fast or cap_proc is None:
            attrs = infer_attributes_heuristic(item["label"], colors)
        else:
            x1, y1, x2, y2 = item["box"].astype(int)
            crop = img.crop(
                (
                    max(0, x1 - 5),
                    max(0, y1 - 5),
                    min(img.width, x2 + 5),
                    min(img.height, y2 + 5),
                )
            )
            attrs = caption_item(
                crop, item["label"], colors, cap_proc, cap_model, device
            )

        result = FurnitureItem(
            id=i,
            category=item["label"],
            confidence=item["score"],
            box=item["box"].tolist(),
            colors=colors,
            size=size,
            position=position,
            area_pct=area_pct,
            material=attrs.get("material", "unknown"),
            style=attrs.get("style", "unknown"),
            description=attrs.get("description", item["label"]),
        )
        results.append(result)
        print(
            f"  [{i}/{len(items)}] {item['label']}: {colors.primary_color} ({colors.primary_percentage:.0f}%)"
        )

    # Output
    visualize(img, results, args.out)
    print_report(results)

    # JSON
    if args.json:
        data = {
            "image": args.image,
            "size": {"w": img.width, "h": img.height},
            "items": [
                {
                    "id": r.id,
                    "category": r.category,
                    "confidence": r.confidence,
                    "box": r.box,
                    "colors": {
                        "primary": r.colors.primary_color,
                        "primary_category": r.colors.primary_category,
                        "primary_rgb": r.colors.primary_rgb,
                        "primary_percentage": r.colors.primary_percentage,
                        "secondary": r.colors.secondary_colors,
                        "secondary_rgbs": r.colors.secondary_rgbs,
                        "palette": [
                            (n, list(rgb), p) for n, rgb, p in r.colors.full_palette
                        ],
                        "description": r.colors.color_description,
                        "tone": r.colors.dominant_tone,
                        "multicolored": r.colors.is_multicolored,
                    },
                    "size": r.size,
                    "position": r.position,
                    "area_pct": r.area_pct,
                    "material": r.material,
                    "style": r.style,
                    "description": r.description,
                }
                for r in results
            ],
        }
        with open(args.json, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[OK] Saved JSON: {args.json}")

    print("\n✅ Done!")


if __name__ == "__main__":
    main()


# Run analysis
# python run4.py --image test.jpg --out result_run4_test.png

# Fast mode
# python run4.py --image image.png --out result_run4_fast.png --fast
