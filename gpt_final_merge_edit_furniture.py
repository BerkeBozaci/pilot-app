#!/usr/bin/env python3
"""
OmniRenovation AI - Complete Merged Pipeline (with Furniture Replacement Preferences)
==================================================================================
Streamlit App with:
- Google Gemini (image generation) for design generation + masked furniture replacement
- GroundingDINO + SAM2 for furniture detection/segmentation
- Heuristic material/style + LAB color analysis
- SearXNG + enrichment for product links
- NEW: User furniture preferences + per-item replacement via mask-based edits

IMPORTANT NOTE ABOUT "NO FLOORPLAN CHANGE"
------------------------------------------
The ONLY reliable way to keep the room structure unchanged is to do localized edits
using segmentation masks (SAM2) and instruct the model to ONLY modify masked pixels.

This file implements:
- storing per-item masks as base64 PNG (full-size mask)
- UI for per-item overrides
- sequential edit loop: apply overrides one by one (reduces drift)

Gemini masked editing support varies by model/version. This code is defensive:
- If Gemini returns no image, it will keep the previous image (no crash).
"""

import streamlit as st
import requests
import json
import base64
import re
import io
import os
import time
from collections import Counter
from PIL import Image
from urllib.parse import quote_plus, urlparse, urljoin
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import warnings

warnings.filterwarnings("ignore")

# ============== PAGE CONFIG ==============
st.set_page_config(page_title="OmniRenovation AI", page_icon="🏠", layout="wide")

# ============== CONSTANTS ==============

FURNITURE_CATEGORIES = [
    "sofa",
    "couch",
    "sectional",
    "loveseat",
    "armchair",
    "accent chair",
    "lounge chair",
    "recliner",
    "ottoman",
    "coffee table",
    "side table",
    "end table",
    "console table",
    "dining table",
    "desk",
    "nightstand",
    "dresser",
    "bookshelf",
    "shelving unit",
    "cabinet",
    "tv stand",
    "bed",
    "headboard",
    "floor lamp",
    "table lamp",
    "pendant light",
    "chandelier",
    "rug",
    "carpet",
    "area rug",
    "curtains",
    "drapes",
    "mirror",
    "artwork",
    "painting",
    "plant",
    "planter",
    "vase",
]

# Model IDs for local segmentation
GROUNDING_DINO_ID = "IDEA-Research/grounding-dino-base"
SAM2_ID = "facebook/sam2.1-hiera-small"
BLIP_ID = "Salesforce/blip-image-captioning-base"
BLIP_LARGE_ID = "Salesforce/blip-image-captioning-large"

SEARX_DEFAULT_URL = "http://localhost:8080"
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
    "wall art",
    "plant",
    "potted plant",
]

BOX_THRESHOLD = 0.25
MAX_DETECTIONS = 25
MIN_MASK_AREA_RATIO = 0.001

# ==================== COLOR DATABASE ====================
COLOR_DATABASE = [
    ("pure white", (255, 255, 255), "white", ["bright", "clean"]),
    ("off-white", (250, 249, 246), "white", ["soft", "warm"]),
    ("ivory", (255, 255, 240), "white", ["creamy", "elegant"]),
    ("cream", (255, 253, 208), "cream", ["warm", "soft"]),
    ("eggshell", (240, 234, 214), "cream", ["neutral", "subtle"]),
    ("linen", (250, 240, 230), "cream", ["natural", "light"]),
    ("white gray", (220, 220, 220), "gray", ["light", "soft"]),
    ("light gray", (192, 192, 192), "gray", ["neutral", "modern"]),
    ("silver", (192, 192, 192), "gray", ["metallic", "cool"]),
    ("medium gray", (150, 150, 150), "gray", ["balanced", "neutral"]),
    ("gray", (128, 128, 128), "gray", ["classic", "neutral"]),
    ("dark gray", (96, 96, 96), "gray", ["sophisticated", "deep"]),
    ("charcoal", (54, 69, 79), "gray", ["dark", "rich"]),
    ("slate gray", (112, 128, 144), "gray", ["cool", "blue-tinted"]),
    ("warm gray", (140, 132, 125), "gray", ["taupe-tinted", "cozy"]),
    ("greige", (170, 165, 155), "gray", ["gray-beige", "modern"]),
    ("black", (0, 0, 0), "black", ["bold", "dramatic"]),
    ("jet black", (20, 20, 20), "black", ["deep", "solid"]),
    ("soft black", (40, 40, 40), "black", ["muted", "subtle"]),
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
    ("light beige", (245, 235, 220), "beige", ["soft", "neutral"]),
    ("beige", (230, 215, 195), "beige", ["classic", "warm neutral"]),
    ("warm beige", (225, 200, 170), "beige", ["cozy", "inviting"]),
    ("sand", (210, 190, 160), "beige", ["natural", "earthy"]),
    ("tan", (210, 180, 140), "tan", ["medium", "warm"]),
    ("camel", (193, 154, 107), "tan", ["rich", "classic"]),
    ("khaki", (195, 176, 145), "tan", ["muted", "casual"]),
    ("taupe", (130, 115, 100), "taupe", ["gray-brown", "sophisticated"]),
    ("dark taupe", (100, 85, 70), "taupe", ["deep", "rich"]),
    ("powder blue", (176, 224, 230), "blue", ["soft", "light"]),
    ("sky blue", (135, 206, 235), "blue", ["bright", "airy"]),
    ("light blue", (173, 216, 230), "blue", ["gentle", "calming"]),
    ("steel blue", (70, 130, 180), "blue", ["cool", "industrial"]),
    ("medium blue", (80, 120, 180), "blue", ["classic", "balanced"]),
    ("denim blue", (100, 130, 170), "blue", ["casual", "relaxed"]),
    ("royal blue", (65, 105, 225), "blue", ["bold", "rich"]),
    ("navy blue", (0, 0, 128), "blue", ["dark", "classic"]),
    ("dark navy", (20, 30, 70), "blue", ["deep", "sophisticated"]),
    ("light teal", (150, 200, 200), "teal", ["soft", "refreshing"]),
    ("teal", (0, 128, 128), "teal", ["balanced", "sophisticated"]),
    ("dark teal", (0, 90, 90), "teal", ["deep", "rich"]),
    ("turquoise", (64, 224, 208), "teal", ["vibrant", "tropical"]),
    ("seafoam", (160, 210, 200), "teal", ["soft", "coastal"]),
    ("mint green", (189, 252, 201), "green", ["fresh", "light"]),
    ("sage green", (176, 196, 164), "green", ["muted", "natural"]),
    ("light sage", (190, 210, 180), "green", ["soft", "calming"]),
    ("olive green", (128, 128, 0), "green", ["earthy", "warm"]),
    ("moss green", (100, 120, 80), "green", ["natural", "forest"]),
    ("forest green", (34, 139, 34), "green", ["deep", "rich"]),
    ("hunter green", (53, 94, 59), "green", ["dark", "classic"]),
    ("emerald green", (0, 155, 119), "green", ["vibrant", "jewel"]),
    ("pale yellow", (255, 255, 200), "yellow", ["soft", "buttery"]),
    ("light yellow", (255, 250, 180), "yellow", ["warm", "sunny"]),
    ("butter yellow", (255, 240, 150), "yellow", ["creamy", "soft"]),
    ("mustard yellow", (255, 200, 60), "yellow", ["deep", "retro"]),
    ("gold", (212, 175, 55), "gold", ["metallic", "luxurious"]),
    ("antique gold", (180, 150, 70), "gold", ["vintage", "warm"]),
    ("brass", (181, 166, 66), "gold", ["metallic", "warm"]),
    ("peach", (255, 218, 185), "orange", ["soft", "warm"]),
    ("apricot", (255, 200, 150), "orange", ["warm", "gentle"]),
    ("coral", (255, 127, 80), "orange", ["vibrant", "warm"]),
    ("terra cotta", (204, 120, 92), "orange", ["earthy", "warm"]),
    ("burnt orange", (204, 85, 0), "orange", ["deep", "autumn"]),
    ("rust", (183, 65, 14), "orange", ["deep", "earthy"]),
    ("light pink", (255, 220, 220), "pink", ["soft", "delicate"]),
    ("blush pink", (255, 200, 200), "pink", ["warm", "romantic"]),
    ("dusty pink", (210, 170, 170), "pink", ["muted", "vintage"]),
    ("rose", (255, 130, 150), "pink", ["medium", "feminine"]),
    ("red", (220, 60, 60), "red", ["classic", "bold"]),
    ("cherry red", (200, 40, 60), "red", ["deep", "rich"]),
    ("crimson", (170, 30, 50), "red", ["dark", "elegant"]),
    ("burgundy", (128, 0, 32), "red", ["wine", "sophisticated"]),
    ("maroon", (100, 30, 40), "red", ["very dark", "rich"]),
    ("lavender", (230, 230, 250), "purple", ["soft", "light"]),
    ("light purple", (210, 200, 230), "purple", ["gentle", "feminine"]),
    ("lilac", (200, 162, 200), "purple", ["soft", "romantic"]),
    ("mauve", (200, 150, 180), "purple", ["dusty", "vintage"]),
    ("purple", (150, 100, 160), "purple", ["classic", "regal"]),
    ("plum", (142, 69, 133), "purple", ["deep", "sophisticated"]),
    ("eggplant", (90, 50, 80), "purple", ["very dark", "elegant"]),
]

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

RETAILERS = [
    {
        "name": "Amazon",
        "search_url": "https://www.amazon.com/s?k={query}&i=garden",
        "icon": "🛒",
    },
    {
        "name": "Wayfair",
        "search_url": "https://www.wayfair.com/keyword.html?keyword={query}",
        "icon": "🏠",
    },
    {
        "name": "IKEA",
        "search_url": "https://www.ikea.com/us/en/search/?q={query}",
        "icon": "🪑",
    },
    {
        "name": "Target",
        "search_url": "https://www.target.com/s?searchTerm={query}",
        "icon": "🎯",
    },
    {
        "name": "Overstock",
        "search_url": "https://www.overstock.com/Home-Garden/?keywords={query}",
        "icon": "📦",
    },
    {
        "name": "Walmart",
        "search_url": "https://www.walmart.com/search?q={query}",
        "icon": "🏬",
    },
    {
        "name": "West Elm",
        "search_url": "https://www.westelm.com/search/?query={query}",
        "icon": "✨",
    },
    {
        "name": "CB2",
        "search_url": "https://www.cb2.com/search/?query={query}",
        "icon": "🛋️",
    },
    {
        "name": "Crate & Barrel",
        "search_url": "https://www.crateandbarrel.com/search?query={query}",
        "icon": "🧱",
    },
    {
        "name": "Pottery Barn",
        "search_url": "https://www.potterybarn.com/search/results.html?words={query}",
        "icon": "🏺",
    },
    {
        "name": "Article",
        "search_url": "https://www.article.com/search?query={query}",
        "icon": "🪴",
    },
    {
        "name": "Burrow",
        "search_url": "https://burrow.com/search?q={query}",
        "icon": "🧩",
    },
    {
        "name": "Joybird",
        "search_url": "https://joybird.com/search/?q={query}",
        "icon": "🎨",
    },
    {
        "name": "Interior Define",
        "search_url": "https://www.interiordefine.com/search?q={query}",
        "icon": "📐",
    },
    {
        "name": "Lamps Plus",
        "search_url": "https://www.lampsplus.com/products/?q={query}",
        "icon": "💡",
    },
    {
        "name": "YLighting",
        "search_url": "https://www.ylighting.com/search?q={query}",
        "icon": "🔆",
    },
    {
        "name": "Rugs USA",
        "search_url": "https://www.rugsusa.com/search?q={query}",
        "icon": "🧶",
    },
    {
        "name": "Revival Rugs",
        "search_url": "https://www.revivalrugs.com/search?q={query}",
        "icon": "🪡",
    },
    {
        "name": "Made.com",
        "search_url": "https://www.made.com/search?q={query}",
        "icon": "🇪🇺",
    },
    {
        "name": "Zara Home",
        "search_url": "https://www.zarahome.com/search?searchTerm={query}",
        "icon": "🧺",
    },
    {
        "name": "H&M Home",
        "search_url": "https://www2.hm.com/en_us/search-results.html?q={query}",
        "icon": "🧵",
    },
]

BLOCKLIST = [
    "pinterest.com",
    "facebook.com",
    "instagram.com",
    "twitter.com",
    "youtube.com",
    "x.com",
    "tiktok.com",
    "reddit.com",
    "houzz.com/discussions",
    "architecturaldigest.com",
    "elledecor.com",
    "designboom.com",
    "dezeen.com",
    "archdaily.com",
    "domino.com",
    "thecut.com",
    "nytimes.com",
    "theguardian.com",
    "hermanmiller.com",
    "vitra.com",
    "knoll.com",
    "steelcase.com",
    "alibaba.com",
    "aliexpress.com",
    "made-in-china.com",
    "globalindustrial.com",
    "grainger.com",
    "popsugar.com",
    "buzzfeed.com",
    "forbes.com",
    "medium.com",
    "slideshare.net",
    "issuu.com",
]

# ============== SESSION STATE ==============
DEFAULT_STATE = {
    "phase": "upload",
    "images": [],
    "preferences": {},
    "valuation": None,
    "designs": None,
    "design_images": {},
    "selected_design": None,
    "selected_design_image": None,  # base64 jpeg from gemini or original
    "selected_design_image_before_prefs": None,  # base64 jpeg
    "selected_design_image_after_prefs": None,  # base64 jpeg
    "furniture_analysis": None,
    "furniture_items": [],
    "product_matches": {},
    "selected_products": {},
    "furniture_overrides": {},  # NEW
    "bom": None,
    "models_loaded": False,
    "furniture_json": None,
}

if "project_state" not in st.session_state:
    st.session_state.project_state = DEFAULT_STATE.copy()

# ============== API KEY MANAGEMENT ==============


def get_api_key(name: str) -> str:
    key_map = {
        "GEMINI_API_KEY": "gemini_api_key",
        "ANTHROPIC_API_KEY": "anthropic_api_key",
    }
    session_key = key_map.get(name, name.lower())
    return st.session_state.get(session_key, "").strip()


def validate_gemini_key(api_key: str) -> tuple:
    if not api_key:
        return False, "API key is empty"
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return True, "Valid"
        elif response.status_code == 403:
            return False, "Invalid or expired"
        else:
            return False, f"Error: {response.status_code}"
    except Exception as e:
        return False, f"Connection error: {str(e)}"


def validate_anthropic_key(api_key: str) -> tuple:
    if not api_key:
        return False, "API key is empty"
    if not api_key.startswith("sk-ant-"):
        return False, "Invalid format"
    return True, "Format valid"


# ==================== COLOR ANALYSIS ====================


def rgb_to_lab(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    """Convert RGB to LAB color space for perceptual color matching."""
    r, g, b = [x / 255.0 for x in rgb]

    def pivot(n):
        return ((n + 0.055) / 1.055) ** 2.4 if n > 0.04045 else n / 12.92

    r, g, b = pivot(r), pivot(g), pivot(b)

    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    x, y, z = x / 0.95047, y / 1.0, z / 1.08883

    def f(t):
        return t ** (1 / 3) if t > 0.008856 else (7.787 * t) + (16 / 116)

    L = (116 * f(y)) - 16
    a = 500 * (f(x) - f(y))
    b_val = 200 * (f(y) - f(z))

    return (L, a, b_val)


def color_distance_lab(rgb1: Tuple[int, int, int], rgb2: Tuple[int, int, int]) -> float:
    lab1 = rgb_to_lab(rgb1)
    lab2 = rgb_to_lab(rgb2)
    dL = (lab1[0] - lab2[0]) * 0.8
    da = lab1[1] - lab2[1]
    db = lab1[2] - lab2[2]
    return (dL**2 + da**2 + db**2) ** 0.5


def get_color_name_lab(rgb: Tuple[int, int, int]) -> Tuple[str, str, List[str]]:
    min_dist = float("inf")
    best_match = ("gray", "gray", ["neutral"])
    for name, ref_rgb, category, terms in COLOR_DATABASE:
        dist = color_distance_lab(rgb, ref_rgb)
        if dist < min_dist:
            min_dist = dist
            best_match = (name, category, terms)
    return best_match


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def is_shadow_or_highlight(
    rgb: Tuple[int, int, int], threshold_dark: int = 30, threshold_bright: int = 245
) -> bool:
    if max(rgb) < threshold_dark:
        return True
    if min(rgb) > threshold_bright:
        return True
    return False


@dataclass
class ColorAnalysisResult:
    primary_color: str
    primary_category: str
    primary_rgb: Tuple[int, int, int]
    primary_percentage: float
    secondary_colors: List[str]
    secondary_rgbs: List[Tuple[int, int, int]]
    full_palette: List[Tuple[str, Tuple[int, int, int], float]]
    color_description: str
    is_multicolored: bool
    dominant_tone: str


def extract_dominant_colors_improved(
    img_rgb: np.ndarray, mask: np.ndarray, n_colors: int = 5
):
    pixels = img_rgb[mask].copy()
    if len(pixels) == 0:
        return [((128, 128, 128), 100.0)]

    valid_mask = np.array([not is_shadow_or_highlight(tuple(p)) for p in pixels])
    if valid_mask.sum() > 0:
        pixels = pixels[valid_mask]

    if len(pixels) == 0:
        pixels = img_rgb[mask].copy()

    max_samples = 8000
    if len(pixels) > max_samples:
        idx = np.random.choice(len(pixels), max_samples, replace=False)
        pixels = pixels[idx]

    quantization = 24
    quantized = ((pixels // quantization) * quantization + quantization // 2).astype(
        np.uint8
    )

    color_counts = {}
    for color in quantized:
        key = tuple(color)
        color_counts[key] = color_counts.get(key, 0) + 1

    sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
    total = sum(c for _, c in sorted_colors)

    results = []
    for color, count in sorted_colors[:n_colors]:
        pct = (count / total) * 100
        results.append((color, pct))

    return results


def merge_similar_colors(colors, threshold: float = 15.0):
    if not colors:
        return []

    merged = []
    used = set()

    for i, (rgb1, pct1) in enumerate(colors):
        if i in used:
            continue

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

        final_rgb = tuple(int(x) for x in combined_rgb / combined_pct)
        name, category, _ = get_color_name_lab(final_rgb)
        merged.append((final_rgb, combined_pct, name, category))
        used.add(i)

    merged.sort(key=lambda x: x[1], reverse=True)
    return merged


def analyze_colors_improved(
    img_rgb: np.ndarray, mask: np.ndarray
) -> ColorAnalysisResult:
    raw_colors = extract_dominant_colors_improved(img_rgb, mask, n_colors=8)
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

    primary_rgb, primary_pct, primary_name, primary_cat = merged[0]
    secondary = [
        (rgb, pct, name, cat)
        for rgb, pct, name, cat in merged[1:4]
        if cat != primary_cat or pct > 10
    ]
    secondary_colors = [name for _, _, name, _ in secondary]
    secondary_rgbs = [rgb for rgb, _, _, _ in secondary]
    full_palette = [(name, rgb, pct) for rgb, pct, name, _ in merged[:5]]

    is_multicolored = len(secondary) >= 2 or (
        len(secondary) >= 1 and secondary[0][1] > 20
    )

    _, a, b_val = rgb_to_lab(primary_rgb)
    if abs(a) < 10 and abs(b_val) < 10:
        tone = "neutral"
    elif a > 5 or b_val > 10:
        tone = "warm"
    else:
        tone = "cool"

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


# ==================== FURNITURE DETECTION ====================


def pick_device():
    import torch

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


def infer_attributes_heuristic(
    label: str, colors: ColorAnalysisResult
) -> Dict[str, str]:
    primary_lower = colors.primary_color.lower()
    category_lower = colors.primary_category.lower()

    if any(
        w in primary_lower
        for w in ["oak", "walnut", "mahogany", "cherry", "teak", "pine", "birch"]
    ):
        material = f"{colors.primary_color} wood"
    elif category_lower in ["brown", "tan", "taupe"] and any(
        x in label.lower() for x in ["table", "desk", "stand", "dresser"]
    ):
        material = "solid wood"
    elif category_lower in ["gray", "beige", "blue", "green", "cream"] and any(
        x in label.lower() for x in ["sofa", "chair", "couch"]
    ):
        material = f"{colors.primary_color} fabric"
    elif category_lower == "black" and any(
        x in label.lower() for x in ["sofa", "chair", "armchair"]
    ):
        material = f"wood frame; {colors.primary_color} leather upholstery"
    elif "metal" in primary_lower or "silver" in primary_lower:
        material = "metal"
    elif any(x in label.lower() for x in ["rug", "carpet"]):
        material = (
            "natural jute"
            if ("khaki" in primary_lower or "tan" in primary_lower)
            else f"{colors.primary_color} fabric"
        )
    elif any(x in label.lower() for x in ["lamp"]):
        material = f"metal; {colors.primary_color.replace('white ', '')} metal base"
    else:
        material = f"{colors.primary_color} finish"

    style = "simple modern"
    if (
        any(x in label.lower() for x in ["rug", "carpet"])
        and "jute" in material.lower()
    ):
        style = "natural jute"
    elif "leather" in material.lower():
        style = "simple elegant"

    return {
        "material": material,
        "style": style,
        "description": f"a {colors.primary_color} {label}",
    }


@st.cache_resource
def load_detection_models():
    """Load GroundingDINO and SAM2 models (cached)"""
    from transformers import (
        AutoProcessor,
        AutoModelForZeroShotObjectDetection,
        Sam2Processor,
        Sam2Model,
    )

    device = pick_device()

    det_proc = AutoProcessor.from_pretrained(GROUNDING_DINO_ID)
    det_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        GROUNDING_DINO_ID
    ).to(device)
    det_model.eval()

    seg_proc = Sam2Processor.from_pretrained(SAM2_ID)
    seg_model = Sam2Model.from_pretrained(SAM2_ID).to(device)
    seg_model.eval()

    return det_proc, det_model, seg_proc, seg_model, device


def detect_furniture(image, det_proc, det_model, device, threshold=0.25):
    import torch

    prompt = ". ".join(FURNITURE_TERMS) + "."
    inputs = det_proc(images=image, text=prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = det_model(**inputs)

    results = det_proc.post_process_grounded_object_detection(
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


def segment_furniture(image, items, seg_proc, seg_model, device):
    import torch

    H, W = image.height, image.width

    for item in items:
        box = item["box"].tolist()
        inputs = seg_proc(images=image, input_boxes=[[box]], return_tensors="pt").to(
            device
        )

        with torch.no_grad():
            outputs = seg_model(**inputs, multimask_output=True)

        try:
            masks = seg_proc.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
            )[0]
        except TypeError:
            masks = seg_proc.post_process_masks(
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
        if mask.shape == (H, W) and (mask.sum() / (H * W)) > MIN_MASK_AREA_RATIO:
            item["mask"] = mask > 0

    return [it for it in items if it.get("mask") is not None]


def convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(i) for i in obj]
    else:
        return obj


def mask_bool_to_base64_png(mask_bool: np.ndarray) -> str:
    """
    Convert boolean mask (H,W) -> base64 PNG (mode 'L'), full-size.
    White(255)=editable, Black(0)=keep.
    """
    mask_u8 = mask_bool.astype(np.uint8) * 255
    img = Image.fromarray(mask_u8, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def run_furniture_analysis(
    image_pil: Image.Image, progress_callback=None
) -> List[Dict]:
    """Run complete furniture analysis pipeline on PIL image, returning items WITH masks."""
    import torch

    if progress_callback:
        progress_callback(0.1, "Loading detection models...")

    det_proc, det_model, seg_proc, seg_model, device = load_detection_models()

    if progress_callback:
        progress_callback(0.3, "Detecting furniture...")

    items = detect_furniture(image_pil, det_proc, det_model, device)

    if not items:
        return []

    if progress_callback:
        progress_callback(0.5, f"Segmenting {len(items)} items...")

    items = segment_furniture(image_pil, items, seg_proc, seg_model, device)

    if progress_callback:
        progress_callback(0.7, "Analyzing colors and materials...")

    img_np = np.array(image_pil)
    results = []

    for i, item in enumerate(items):
        colors = analyze_colors_improved(img_np, item["mask"])
        size, position, area_pct = analyze_size(
            item["box"], image_pil.width, image_pil.height
        )
        attrs = infer_attributes_heuristic(item["label"], colors)

        primary_rgb = (
            list(colors.primary_rgb) if colors.primary_rgb else [128, 128, 128]
        )
        mask_b64 = mask_bool_to_base64_png(item["mask"])

        result = {
            "id": i + 1,
            "name": str(item["label"]),
            "color": str(colors.primary_color),
            "material": str(attrs.get("material", "unknown")),
            "style": str(attrs.get("style", "simple modern")),
            "confidence": int(float(item["score"]) * 100),
            "position": str(position),
            "size": str(size),
            "area_percent": float(area_pct),
            "full": str(
                attrs.get("description", f"a {colors.primary_color} {item['label']}")
            ),
            "box": [float(x) for x in item["box"].tolist()],
            "mask_png_base64": mask_b64,  # NEW
            "colors_detail": {
                "primary": str(colors.primary_color),
                "primary_rgb": primary_rgb,
                "primary_hex": rgb_to_hex(tuple(primary_rgb)),
                "secondary": [str(c) for c in colors.secondary_colors],
                "tone": str(colors.dominant_tone),
            },
        }
        results.append(result)

    if progress_callback:
        progress_callback(1.0, "Analysis complete!")

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return convert_to_serializable(results)


# ==================== SEARCH FUNCTIONS ====================


def build_search_queries(item: Dict, sites: List[str] = None) -> List[str]:
    name = item.get("name", "").strip()
    color = item.get("color", "").strip()
    material = item.get("material", "").strip()

    material_tokens = []
    m_low = material.lower()

    for keyword, token in [
        ("walnut", "walnut"),
        ("oak", "oak"),
        ("wood", "wood"),
        ("metal", "metal"),
        ("leather", "leather"),
        ("fabric", "fabric"),
        ("jute", "jute"),
        ("glass", "glass"),
    ]:
        if keyword in m_low:
            material_tokens.append(token)

    if "black metal" in m_low:
        material_tokens.append('"black metal frame"')

    mat_str = " ".join(material_tokens[:3])

    queries = []
    queries.append(f"{name} {color} {mat_str}".strip())
    queries.append(f"{name} {color}".strip())
    queries.append(f"{name} {mat_str}".strip())
    queries.append(name)

    seen, out = set(), []
    for q in queries:
        q = re.sub(r"\s+", " ", q.strip())
        if q and q not in seen:
            out.append(q)
            seen.add(q)

    return out


def generate_retailer_links(item: Dict) -> List[Dict]:
    queries = build_search_queries(item)
    primary_query = queries[0] if queries else item.get("name", "furniture")
    encoded = quote_plus(primary_query)

    links = []
    for r in RETAILERS:
        links.append(
            {
                "retailer": r["name"],
                "icon": r.get("icon", "🔗"),
                "url": r["search_url"].replace("{query}", encoded),
                "query": primary_query,
            }
        )

    return links


def searx_search(
    searx_base: str, query: str, max_results: int = 10, lang: str = "en"
) -> List[Dict]:
    url = searx_base.rstrip("/") + "/search"
    params = {"q": query, "format": "json", "language": lang, "safesearch": 0}

    try:
        r = requests.get(url, params=params, timeout=30, headers=DEFAULT_HEADERS)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []

    results = []
    for item in data.get("results", [])[:max_results]:
        u = item.get("url")
        if not u:
            continue
        domain = urlparse(u).netloc.lower()
        if any(b in domain for b in BLOCKLIST):
            continue
        results.append(
            {
                "title": item.get("title"),
                "url": u,
                "snippet": item.get("content") or item.get("snippet"),
                "domain": domain,
                "thumbnail": item.get("thumbnail"),
            }
        )

    return results


# ============== PRODUCT ENRICHMENT ==============


def parse_jsonld(html: str) -> List[Dict]:
    """Parse JSON-LD blocks from HTML"""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return []

    results = []
    try:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
            try:
                raw = tag.string or tag.get_text() or ""
                raw = raw.strip()
                if not raw:
                    continue
                data = json.loads(raw)
                if isinstance(data, list):
                    results.extend([d for d in data if isinstance(d, dict)])
                elif isinstance(data, dict):
                    results.append(data)
            except Exception:
                continue
    except Exception:
        pass
    return results


def extract_price_from_html(
    html: str, url: str = ""
) -> Tuple[Optional[str], Optional[str]]:
    price_patterns = [
        r"\$\s*([\d,]+\.?\d*)",
        r"€\s*([\d,]+\.?\d*)",
        r"£\s*([\d,]+\.?\d*)",
        r"([\d,]+\.?\d*)\s*(?:TL|₺)",
        r'"price":\s*"?([\d.,]+)"?',
        r'"amount":\s*"?([\d.,]+)"?',
    ]

    jsonlds = parse_jsonld(html)
    for data in jsonlds:
        if "@graph" in data:
            for item in data.get("@graph", []):
                if isinstance(item, dict) and item.get("offers"):
                    offers = item["offers"]
                    if isinstance(offers, list):
                        offers = offers[0] if offers else {}
                    if isinstance(offers, dict):
                        price = offers.get("price") or offers.get("lowPrice")
                        currency = offers.get("priceCurrency", "USD")
                        if price:
                            return str(price), currency

        offers = data.get("offers")
        if offers:
            if isinstance(offers, list):
                offers = offers[0] if offers else {}
            if isinstance(offers, dict):
                price = offers.get("price") or offers.get("lowPrice")
                currency = offers.get("priceCurrency", "USD")
                if price:
                    return str(price), currency

    for pattern in price_patterns:
        matches = re.findall(pattern, html[:10000], re.IGNORECASE)
        if matches:
            price = matches[0].replace(",", "")
            try:
                val = float(price)
                if 1 < val < 100000:
                    if ".co.uk" in url:
                        return price, "GBP"
                    elif ".de" in url or ".fr" in url:
                        return price, "EUR"
                    elif ".tr" in url:
                        return price, "TRY"
                    return price, "USD"
            except Exception:
                continue

    return None, None


def extract_image_from_html(html: str, base_url: str) -> Optional[str]:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return None

    try:
        soup = BeautifulSoup(html, "html.parser")

        jsonlds = parse_jsonld(html)
        for data in jsonlds:
            if "@graph" in data:
                for item in data.get("@graph", []):
                    if isinstance(item, dict):
                        img = item.get("image")
                        if isinstance(img, str):
                            return img
                        elif isinstance(img, list) and img:
                            return (
                                img[0] if isinstance(img[0], str) else img[0].get("url")
                            )
                        elif isinstance(img, dict):
                            return img.get("url")

            img = data.get("image")
            if isinstance(img, str):
                return img
            elif isinstance(img, list) and img:
                return img[0] if isinstance(img[0], str) else img[0].get("url")
            elif isinstance(img, dict):
                return img.get("url")

        og = soup.find("meta", property="og:image")
        if og and og.get("content"):
            img_url = og["content"]
            if img_url.startswith("//"):
                return "https:" + img_url
            elif img_url.startswith("/"):
                return urljoin(base_url, img_url)
            return img_url

        selectors = [
            'img[class*="product"]',
            'img[class*="gallery"]',
            'img[class*="main"]',
            '[class*="product-image"] img',
        ]

        for sel in selectors:
            imgs = soup.select(sel)
            for img in imgs[:3]:
                src = img.get("src") or img.get("data-src")
                if src and not any(
                    x in src.lower() for x in ["icon", "logo", "sprite", "1x1"]
                ):
                    if src.startswith("//"):
                        return "https:" + src
                    elif src.startswith("/"):
                        return urljoin(base_url, src)
                    return src
    except Exception:
        pass

    return None


def enrich_product(url: str, timeout: int = 15) -> Dict:
    result = {
        "image": None,
        "price": None,
        "currency": None,
        "title": None,
        "success": False,
    }
    try:
        r = requests.get(
            url, headers=DEFAULT_HEADERS, timeout=timeout, allow_redirects=True
        )
        if r.status_code >= 400:
            return result

        html = r.text
        if len(html) < 500:
            return result

        result["image"] = extract_image_from_html(html, r.url)
        price, currency = extract_price_from_html(html, r.url)
        result["price"] = price
        result["currency"] = currency

        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")
            og_title = soup.find("meta", property="og:title")
            if og_title and og_title.get("content"):
                result["title"] = og_title["content"]
            elif soup.title:
                result["title"] = soup.title.get_text().strip()
        except Exception:
            pass

        result["success"] = bool(result["image"] or result["price"])
    except Exception:
        pass

    return result


def search_with_enrichment(
    searx_base: str,
    query: str,
    max_results: int = 6,
    enrich_top: int = 3,
    lang: str = "en",
) -> List[Dict]:
    results = searx_search(searx_base, query, max_results=max_results * 2, lang=lang)
    if not results:
        return []

    def is_product_page(url: str) -> bool:
        u = url.lower()
        good = ["/p/", "/product/", "/products/", "/item/", "/dp/", "-p-"]
        bad = ["/search", "/s?", "/category", "/collections", "/c/"]
        if any(x in u for x in good):
            return True
        if any(x in u for x in bad):
            return False
        return True

    results.sort(key=lambda x: 0 if is_product_page(x.get("url", "")) else 1)
    results = results[:max_results]

    for i, result in enumerate(results[:enrich_top]):
        url = result.get("url")
        if url:
            enriched = enrich_product(url)
            result["enriched"] = enriched
            if enriched.get("image"):
                result["image"] = enriched["image"]
            if enriched.get("price"):
                result["price"] = enriched["price"]
                result["currency"] = enriched.get("currency", "USD")
            if enriched.get("title"):
                result["title"] = enriched["title"]
        time.sleep(0.5)

    return results


# ============== GEMINI IMAGE GENERATION + MASKED REPLACEMENT ==============


def _gemini_generate_content(
    parts: List[Dict],
    model_variants: List[str],
    api_key: str,
    temperature: float = 0.8,
    timeout: int = 180,
) -> Dict:
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "responseModalities": ["IMAGE", "TEXT"],
            "temperature": float(temperature),
        },
    }

    last_error = None
    for model_name in model_variants:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        for attempt in range(3):
            try:
                resp = requests.post(
                    url, headers=headers, json=payload, timeout=timeout
                )
                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts_out = candidates[0].get("content", {}).get("parts", [])
                        for p in parts_out:
                            if "inlineData" in p and p["inlineData"].get("data"):
                                return {
                                    "success": True,
                                    "image_base64": p["inlineData"]["data"],
                                    "method": model_name,
                                }
                    last_error = "No image in response"
                    break
                elif resp.status_code == 429:
                    time.sleep((2**attempt) * 5)
                    continue
                elif resp.status_code == 404:
                    last_error = f"Model {model_name} not found"
                    break
                else:
                    last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    break
            except Exception as e:
                last_error = str(e)
                continue

    return {"success": False, "error": f"Generation failed: {last_error}"}


def generate_design_with_gemini(
    image_base64: str, style: str, variation: str, room_type: str = "room"
) -> dict:
    api_key = get_api_key("GEMINI_API_KEY")
    if not api_key:
        return {"success": False, "error": "Gemini API key not configured"}

    style_descriptions = {
        "Modern Minimalist": "modern minimalist style with clean lines, neutral colors, minimal furniture, natural light",
        "Scandinavian": "Scandinavian style with light oak wood, white walls, cozy textiles, plants, warm lighting",
        "Industrial": "industrial loft style with exposed brick, black metal, Edison bulbs, leather furniture",
        "Mid-Century Modern": "mid-century modern with walnut furniture, organic shapes, mustard/teal accents",
        "Contemporary": "contemporary style with mixed materials, neutral base, bold accents",
        "Bohemian": "bohemian style with colorful patterns, layered textiles, macrame, plants",
        "Coastal": "coastal style with blue/white palette, rattan furniture, nautical accents",
        "Farmhouse": "modern farmhouse with shiplap walls, rustic wood, neutral tones",
    }

    variation_descriptions = {
        "Light & Airy": "bright atmosphere with natural light, white/cream palette",
        "Warm & Cozy": "warm atmosphere with ambient lighting, earth tones, soft textures",
        "Bold & Dramatic": "dramatic atmosphere with deep colors, high contrast, statement pieces",
    }

    style_desc = style_descriptions.get(style, "modern interior design")
    variation_desc = variation_descriptions.get(variation, "")

    prompt = f"""Transform this {room_type} into a beautifully designed interior space.

STYLE: {style_desc}
MOOD: {variation_desc}

CRITICAL INSTRUCTIONS:
1. KEEP THE EXACT SAME ROOM STRUCTURE - walls, windows, doors, ceiling, floor plan
2. KEEP THE SAME CAMERA ANGLE AND PERSPECTIVE
3. Add furniture, decor, colors, textures, lighting fixtures
4. Make it photorealistic and professionally designed
5. Ensure cohesive magazine-quality design

Generate the redesigned room image."""

    model_variants = [
        "gemini-2.0-flash-exp-image-generation",
        "gemini-2.0-flash-exp",
    ]

    parts = [
        {"text": prompt},
        {"inlineData": {"mimeType": "image/jpeg", "data": image_base64}},
    ]
    return _gemini_generate_content(
        parts, model_variants, api_key, temperature=0.8, timeout=180
    )


def replace_furniture_with_mask_gemini(
    base_image_b64_jpeg: str,
    mask_png_b64: str,
    replace_prompt: str,
    item_name: str,
    temperature: float = 0.65,
) -> Dict:
    """
    Mask-based replacement attempt using Gemini.
    We provide:
      - base image (jpeg)
      - mask image (png) where white=editable, black=keep
    and a strict prompt to edit ONLY masked region.

    If the model doesn't support mask editing, it may ignore mask. We fail safely:
    - If no image returned -> returns success False and caller keeps base image.
    """
    api_key = get_api_key("GEMINI_API_KEY")
    if not api_key:
        return {"success": False, "error": "Gemini API key not configured"}

    strict_prompt = f"""You are editing an interior design photo.

TASK:
Replace ONLY the furniture inside the provided MASK region with:
{replace_prompt}

HARD CONSTRAINTS:
- MODIFY ONLY pixels inside the MASK (white area). DO NOT change anything outside the mask.
- DO NOT change room structure: walls, floor, ceiling, windows, doors, architecture.
- DO NOT change camera angle, perspective, lens, or framing.
- Keep lighting and shadows consistent with the scene.
- Produce a photorealistic result.

The masked object category is: {item_name}.
Return the edited full image."""

    model_variants = [
        "gemini-2.0-flash-exp-image-generation",
        "gemini-2.0-flash-exp",
    ]

    parts = [
        {"text": strict_prompt},
        {"inlineData": {"mimeType": "image/jpeg", "data": base_image_b64_jpeg}},
        {"inlineData": {"mimeType": "image/png", "data": mask_png_b64}},
    ]

    return _gemini_generate_content(
        parts, model_variants, api_key, temperature=temperature, timeout=240
    )


def apply_overrides_sequentially(
    base_design_b64: str,
    furniture_items: List[Dict],
    overrides: Dict[str, Dict],
    selected_products: Dict[str, Dict],
    progress_cb=None,
) -> Dict:
    """
    Applies user overrides one-by-one to reduce drift.
    - base_design_b64: starting JPEG base64
    - overrides: dict item_id -> {enabled, mode, text, use_selected_product}
    - selected_products: dict item_id -> product dict (optional)
    Returns dict with success + final image base64.
    """
    current_b64 = base_design_b64
    editable = [(str(it.get("id")), it) for it in furniture_items]

    total = max(
        1,
        sum(1 for item_id, _ in editable if overrides.get(item_id, {}).get("enabled")),
    )
    done = 0

    for item_id, item in editable:
        ov = overrides.get(item_id, {})
        if not ov.get("enabled"):
            continue

        mask_b64 = item.get("mask_png_base64")
        if not mask_b64:
            continue

        # Build replacement prompt
        mode = ov.get("mode", "text")
        replace_prompt = ""
        if mode == "product":
            prod = selected_products.get(item_id, {})
            title = (
                prod.get("title")
                or prod.get("enriched", {}).get("title")
                or "selected product"
            )
            price = prod.get("price")
            currency = prod.get("currency")
            meta = f"{title}"
            if price:
                meta += f", approx price {currency or ''}{price}"
            replace_prompt = f"Make the furniture look like: {meta}. Match materials, shape, legs, cushions, and style as closely as possible."
            # If user also typed extra notes
            extra = (ov.get("text") or "").strip()
            if extra:
                replace_prompt += f" Extra requirements: {extra}"
        else:
            replace_prompt = (ov.get("text") or "").strip()

        if not replace_prompt:
            # nothing to do
            continue

        if progress_cb:
            progress_cb(
                min(0.98, (done / total)),
                f"Replacing item {item_id}: {item.get('name','item')}...",
            )

        out = replace_furniture_with_mask_gemini(
            base_image_b64_jpeg=current_b64,
            mask_png_b64=mask_b64,
            replace_prompt=replace_prompt,
            item_name=item.get("name", "furniture"),
        )

        if out.get("success") and out.get("image_base64"):
            current_b64 = out["image_base64"]
        # else: keep current_b64 (fail-safe)

        done += 1
        time.sleep(0.2)

    if progress_cb:
        progress_cb(1.0, "✅ Preferences applied!")

    return {"success": True, "image_base64": current_b64}


# ============== CLAUDE FOR ROOM ANALYSIS + DESIGN CONCEPTS ==============


def analyze_room_with_claude(images: list, preferences: dict) -> dict:
    api_key = get_api_key("ANTHROPIC_API_KEY")
    if not api_key:
        return {"error": "Anthropic API key not configured"}

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }

    content = []
    for img in images:
        if img.get("success"):
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": img["data"],
                    },
                }
            )

    content.append(
        {
            "type": "text",
            "text": f"""Analyze this room for renovation. Style: {preferences.get('style', 'Modern')}.

Return ONLY valid JSON:
{{
    "property_assessment": {{
        "room_type": "Living Room/Bedroom/etc",
        "current_condition": "poor/fair/good/excellent",
        "square_footage_estimate": "estimated sq ft",
        "notable_features": ["feature1", "feature2"]
    }},
    "cost_estimate": {{
        "low": 8000,
        "mid": 15000,
        "high": 30000
    }}
}}""",
        }
    )

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": content}],
            },
            timeout=60,
        )

        if response.status_code == 200:
            data = response.json()
            text = data["content"][0]["text"]
            match = re.search(r"\{[\s\S]*\}", text)
            if match:
                return json.loads(match.group())
            return {"error": "Could not parse response"}
        return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def create_design_concepts(valuation: dict, preferences: dict) -> dict:
    api_key = get_api_key("ANTHROPIC_API_KEY")
    if not api_key:
        return {"error": "Anthropic API key not configured"}

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }

    room_type = valuation.get("property_assessment", {}).get("room_type", "room")
    style = preferences.get("style", "Modern Minimalist")

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 2048,
                "messages": [
                    {
                        "role": "user",
                        "content": f"""Create 3 interior design variations for a {room_type} in {style} style.

Return ONLY valid JSON:
{{
    "design_options": [
        {{
            "option_number": 1,
            "name": "{style} - Light & Airy",
            "variation": "Light & Airy",
            "concept": "Bright, open design",
            "color_palette": {{"primary": "#F5F5F5", "secondary": "#E8E8E8", "accent": "#4A90A4"}},
            "key_furniture": ["light grey sofa", "oak coffee table", "brass lamp", "cream rug"],
            "estimated_cost": 12000
        }},
        {{
            "option_number": 2,
            "name": "{style} - Warm & Cozy",
            "variation": "Warm & Cozy",
            "concept": "Warm, inviting atmosphere",
            "color_palette": {{"primary": "#F5E6D3", "secondary": "#D4A574", "accent": "#8B4513"}},
            "key_furniture": ["cognac leather sofa", "walnut table", "ceramic lamps", "jute rug"],
            "estimated_cost": 15000
        }},
        {{
            "option_number": 3,
            "name": "{style} - Bold & Dramatic",
            "variation": "Bold & Dramatic",
            "concept": "Striking sophisticated design",
            "color_palette": {{"primary": "#2C3E50", "secondary": "#34495E", "accent": "#C0392B"}},
            "key_furniture": ["navy velvet sofa", "marble table", "gold pendant", "persian rug"],
            "estimated_cost": 20000
        }}
    ]
}}""",
                    }
                ],
            },
            timeout=60,
        )

        if response.status_code == 200:
            data = response.json()
            text = data["content"][0]["text"]
            match = re.search(r"\{[\s\S]*\}", text)
            if match:
                return json.loads(match.group())
            return {"error": "Could not parse response"}
        return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}


# ============== IMAGE PROCESSING ==============


def process_uploaded_image(uploaded_file) -> dict:
    try:
        uploaded_file.seek(0)
        img = Image.open(uploaded_file)
        if img.mode in ("RGBA", "P", "LA"):
            img = img.convert("RGB")

        max_size = 1024
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.LANCZOS)

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        base64_data = base64.b64encode(buffer.getvalue()).decode()

        return {
            "success": True,
            "data": base64_data,
            "name": getattr(uploaded_file, "name", "upload.jpg"),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============== UI COMPONENTS ==============


def render_header():
    st.title("🏠 OmniRenovation AI")
    st.caption(
        "AI-Powered Interior Design with GroundingDINO + SAM2 Furniture Analysis"
    )

    phases = [
        "Upload",
        "Analysis",
        "Design",
        "Furniture",
        "Products",
        "Preferences",  # NEW
        "BOM",
        "Complete",
    ]
    phase_map = {
        "upload": 0,
        "valuation": 1,
        "design": 2,
        "furniture_analysis": 3,
        "products": 4,
        "preferences": 5,
        "bom": 6,
        "complete": 7,
    }
    current = phase_map.get(st.session_state.project_state["phase"], 0)

    cols = st.columns(len(phases))
    for i, phase in enumerate(phases):
        with cols[i]:
            if i < current:
                st.success(f"✓ {phase}")
            elif i == current:
                st.info(f"→ {phase}")
            else:
                st.write(f"○ {phase}")


def render_upload_phase():
    st.header("📤 Upload Your Room")

    with st.expander("🔑 API Configuration", expanded=True):
        st.info("**Required:** Google Gemini + Anthropic Claude API keys")

        col1, col2 = st.columns(2)

        with col1:
            gemini_key = st.text_input(
                "Google Gemini API Key",
                type="password",
                value=st.session_state.get("gemini_api_key", ""),
                placeholder="Enter Gemini API key",
            )
            if gemini_key:
                st.session_state.gemini_api_key = gemini_key
                is_valid, msg = validate_gemini_key(gemini_key)
                if is_valid:
                    st.success("✅ Valid")
                else:
                    st.error(f"❌ {msg}")

        with col2:
            anthropic_key = st.text_input(
                "Anthropic Claude API Key",
                type="password",
                value=st.session_state.get("anthropic_api_key", ""),
                placeholder="Enter Claude API key (sk-ant-...)",
            )
            if anthropic_key:
                st.session_state.anthropic_api_key = anthropic_key
                is_valid, msg = validate_anthropic_key(anthropic_key)
                if is_valid:
                    st.success("✅ Valid format")
                else:
                    st.error(f"❌ {msg}")

    st.divider()

    uploaded_files = st.file_uploader(
        "📷 Upload Room Photo",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        cols = st.columns(min(4, len(uploaded_files)))
        for i, f in enumerate(uploaded_files):
            with cols[i % 4]:
                f.seek(0)
                st.image(Image.open(f), use_container_width=True)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        budget = st.select_slider(
            "Budget Range",
            options=["<$5K", "$5K-15K", "$15K-30K", "$30K-50K", ">$50K"],
            value="$15K-30K",
        )
        style = st.selectbox(
            "Design Style",
            options=[
                "Modern Minimalist",
                "Scandinavian",
                "Industrial",
                "Mid-Century Modern",
                "Contemporary",
                "Bohemian",
                "Coastal",
                "Farmhouse",
            ],
        )

    with col2:
        room_type = st.selectbox(
            "Room Type",
            options=[
                "Living Room",
                "Bedroom",
                "Kitchen",
                "Bathroom",
                "Home Office",
                "Dining Room",
            ],
        )

    st.divider()

    can_start = (
        bool(st.session_state.get("gemini_api_key"))
        and bool(st.session_state.get("anthropic_api_key"))
        and bool(uploaded_files)
    )

    if st.button("🚀 Analyze Room", type="primary", disabled=not can_start):
        processed = [process_uploaded_image(f) for f in uploaded_files]
        successful = [img for img in processed if img.get("success")]

        if successful:
            st.session_state.project_state = DEFAULT_STATE.copy()
            st.session_state.project_state.update(
                {
                    "images": successful,
                    "preferences": {
                        "budget": budget,
                        "style": style,
                        "room_type": room_type,
                    },
                    "phase": "valuation",
                }
            )
            st.rerun()


def render_valuation_phase():
    st.header("📊 Room Analysis")

    if not st.session_state.project_state["valuation"]:
        with st.spinner("🔍 Analyzing room..."):
            result = analyze_room_with_claude(
                st.session_state.project_state["images"],
                st.session_state.project_state["preferences"],
            )
            st.session_state.project_state["valuation"] = result
            st.rerun()

    valuation = st.session_state.project_state["valuation"]

    if "error" in valuation:
        st.error(f"Analysis failed: {valuation['error']}")
        if st.button("🔄 Retry"):
            st.session_state.project_state["valuation"] = None
            st.rerun()
        return

    assessment = valuation.get("property_assessment", {})
    costs = valuation.get("cost_estimate", {})

    col1, col2, col3 = st.columns(3)
    col1.metric("Room Type", assessment.get("room_type", "N/A"))
    col2.metric("Condition", str(assessment.get("current_condition", "N/A")).title())
    col3.metric("Est. Cost", f"${costs.get('mid', 0):,}")

    if st.session_state.project_state["images"]:
        st.image(
            base64.b64decode(st.session_state.project_state["images"][0]["data"]),
            caption="Your Room",
            use_container_width=True,
        )

    st.divider()

    if st.button("🎨 Generate Designs", type="primary"):
        st.session_state.project_state["phase"] = "design"
        st.rerun()


def render_design_phase():
    st.header("🎨 Design Options")

    if not st.session_state.project_state["designs"]:
        with st.spinner("Creating designs..."):
            result = create_design_concepts(
                st.session_state.project_state["valuation"],
                st.session_state.project_state["preferences"],
            )
            st.session_state.project_state["designs"] = result
            st.rerun()

    designs = st.session_state.project_state["designs"]
    if "error" in designs:
        st.error(f"Failed: {designs['error']}")
        if st.button("🔄 Retry"):
            st.session_state.project_state["designs"] = None
            st.rerun()
        return

    original_image = (
        st.session_state.project_state["images"][0]["data"]
        if st.session_state.project_state["images"]
        else None
    )
    design_images = st.session_state.project_state.get("design_images", {})
    room_type = st.session_state.project_state["preferences"].get("room_type", "room")

    for option in designs.get("design_options", []):
        num = option["option_number"]
        st.subheader(f"Option {num}: {option.get('name', 'Design')}")

        col1, col2 = st.columns([2, 1])

        with col1:
            if num in design_images:
                if design_images[num].get("success"):
                    st.image(
                        base64.b64decode(design_images[num]["image_base64"]),
                        caption="AI Generated Design",
                        use_container_width=True,
                    )
                else:
                    st.error(f"Failed: {design_images[num].get('error')}")
                    if st.button(f"🔄 Retry", key=f"retry_{num}"):
                        del design_images[num]
                        st.session_state.project_state["design_images"] = design_images
                        st.rerun()
            else:
                palette = option.get("color_palette", {})
                st.markdown("**Color Palette:**")
                pcols = st.columns(3)
                for i, (name, color) in enumerate(list(palette.items())[:3]):
                    with pcols[i]:
                        st.color_picker(
                            name.title(), color, disabled=True, key=f"color_{num}_{i}"
                        )

                if original_image and st.button(f"🖼️ Generate Design", key=f"gen_{num}"):
                    with st.spinner("Generating with Gemini..."):
                        result = generate_design_with_gemini(
                            original_image,
                            st.session_state.project_state["preferences"]["style"],
                            option.get("variation", "Light & Airy"),
                            room_type,
                        )
                        design_images[num] = result
                        st.session_state.project_state["design_images"] = design_images
                        st.rerun()

        with col2:
            st.markdown(f"**Concept:** {option.get('concept', 'N/A')}")
            st.metric("Cost", f"${option.get('estimated_cost', 0):,}")

            st.markdown("**Key Items:**")
            for item in option.get("key_furniture", [])[:4]:
                st.write(f"• {item}")

            if st.button(f"✅ Select", key=f"select_{num}", type="primary"):
                st.session_state.project_state["selected_design"] = option
                if num in design_images and design_images[num].get("success"):
                    chosen = design_images[num]["image_base64"]
                else:
                    chosen = original_image

                st.session_state.project_state["selected_design_image"] = chosen
                st.session_state.project_state["selected_design_image_before_prefs"] = (
                    chosen
                )
                st.session_state.project_state["selected_design_image_after_prefs"] = (
                    None
                )
                st.success("Selected!")

        st.divider()

    if st.session_state.project_state.get("selected_design"):
        st.success(
            f"**Selected:** {st.session_state.project_state['selected_design'].get('name')}"
        )

        if st.button("🔍 Analyze Furniture (GroundingDINO + SAM2)", type="primary"):
            st.session_state.project_state["phase"] = "furniture_analysis"
            st.rerun()


def render_furniture_analysis_phase():
    st.header("🔍 Furniture Analysis (GroundingDINO + SAM2)")

    design_image_b64 = st.session_state.project_state.get("selected_design_image")
    if not design_image_b64:
        st.error("No design image selected")
        return

    if not st.session_state.project_state.get("furniture_analysis"):
        st.info(
            "🔬 Running furniture detection with GroundingDINO and segmentation with SAM2..."
        )
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(pct, msg):
            progress_bar.progress(float(pct))
            status_text.text(msg)

        try:
            image_bytes = base64.b64decode(design_image_b64)
            image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            furniture_items = run_furniture_analysis(image_pil, update_progress)

            if furniture_items:
                st.session_state.project_state["furniture_analysis"] = {
                    "items": furniture_items
                }
                st.session_state.project_state["furniture_items"] = furniture_items
                st.session_state.project_state["furniture_json"] = {
                    "items": furniture_items
                }
                st.rerun()
            else:
                st.warning("No furniture detected in the image")

        except ImportError as e:
            st.error(f"Missing required packages: {e}")
            st.info("Please install: pip install torch transformers beautifulsoup4")
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            if st.button("🔄 Retry"):
                st.session_state.project_state["furniture_analysis"] = None
                st.session_state.project_state["furniture_items"] = []
                st.rerun()

    furniture_items = st.session_state.project_state.get("furniture_items", [])
    if not furniture_items:
        st.warning("No furniture items to show yet.")
        if st.button("⬅️ Back to Designs"):
            st.session_state.project_state["phase"] = "design"
            st.rerun()
        return

    st.success(f"Detected {len(furniture_items)} furniture items.")

    # Show items table-ish
    for it in furniture_items:
        with st.expander(
            f"#{it.get('id')} — {it.get('name')} (conf {it.get('confidence')}%)",
            expanded=False,
        ):
            c1, c2 = st.columns(2)
            with c1:
                st.write(
                    f"**Color:** {it.get('color')} ({it.get('colors_detail', {}).get('primary_hex', '')})"
                )
                st.write(f"**Material:** {it.get('material')}")
                st.write(f"**Style:** {it.get('style')}")
                st.write(
                    f"**Position:** {it.get('position')} | **Size:** {it.get('size')} | **Area:** {it.get('area_percent')}%"
                )
                st.write(f"**Description:** {it.get('full')}")
            with c2:
                # Show mask preview
                m_b64 = it.get("mask_png_base64")
                if m_b64:
                    try:
                        mask_img = Image.open(
                            io.BytesIO(base64.b64decode(m_b64))
                        ).convert("L")
                        st.image(
                            mask_img,
                            caption="Mask (white = editable)",
                            use_container_width=True,
                        )
                    except Exception:
                        st.warning("Mask preview failed.")

    st.divider()

    colA, colB = st.columns([1, 1])
    with colA:
        if st.button("🔎 Find Product Links (SearXNG)", type="primary"):
            st.session_state.project_state["phase"] = "products"
            st.rerun()
    with colB:
        if st.button("⬅️ Back to Designs"):
            st.session_state.project_state["phase"] = "design"
            st.rerun()


def render_products_phase():
    st.header("🔗 Product Search (SearXNG)")

    furniture_items = st.session_state.project_state.get("furniture_items", [])
    if not furniture_items:
        st.error("No furniture items available. Go back and run furniture analysis.")
        if st.button("⬅️ Back"):
            st.session_state.project_state["phase"] = "furniture_analysis"
            st.rerun()
        return

    # Config
    with st.expander("⚙️ Search settings", expanded=True):
        searx_url = st.text_input(
            "SearXNG Base URL",
            value=st.session_state.get("searx_url", SEARX_DEFAULT_URL),
        )
        st.session_state.searx_url = searx_url

        lang = st.selectbox("Language", options=["en", "tr", "de", "fr", "it"], index=0)
        max_results = st.slider("Max results per item", 3, 15, 6)
        enrich_top = st.slider("Enrich top N results (price/image)", 0, 6, 3)

    product_matches = st.session_state.project_state.get("product_matches", {}) or {}

    # Search buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🔍 Search All Items", type="primary"):
            progress = st.progress(0)
            status = st.empty()
            total = len(furniture_items)
            for idx, it in enumerate(furniture_items, start=1):
                status.text(
                    f"Searching #{it.get('id')} {it.get('name')} ({idx}/{total})..."
                )
                queries = build_search_queries(it)
                results = []
                for q in queries[:2]:
                    results = search_with_enrichment(
                        searx_base=searx_url,
                        query=q,
                        max_results=max_results,
                        enrich_top=enrich_top,
                        lang=lang,
                    )
                    if results:
                        break
                product_matches[str(it.get("id"))] = {
                    "queries": queries,
                    "results": results,
                }
                progress.progress(idx / total)
                time.sleep(0.2)

            st.session_state.project_state["product_matches"] = product_matches
            st.success("Search complete!")
            st.rerun()

    with col2:
        if st.button("➡️ Next: Preferences (Replace Furniture)", type="primary"):
            st.session_state.project_state["phase"] = "preferences"
            st.rerun()

    st.divider()

    # Per item results
    for it in furniture_items:
        item_id = str(it.get("id"))
        with st.expander(
            f"#{item_id} — {it.get('name')} | {it.get('color')} | {it.get('material')}",
            expanded=False,
        ):
            links = generate_retailer_links(it)
            st.markdown("**Quick retailer searches:**")
            link_cols = st.columns(4)
            for i, l in enumerate(links[:8]):
                with link_cols[i % 4]:
                    st.markdown(f"{l['icon']} [{l['retailer']}]({l['url']})")

            data = product_matches.get(item_id)
            if not data:
                st.info("No SearX results saved for this item yet.")
                if st.button(f"Search this item", key=f"search_one_{item_id}"):
                    queries = build_search_queries(it)
                    results = []
                    for q in queries[:2]:
                        results = search_with_enrichment(
                            searx_base=st.session_state.searx_url,
                            query=q,
                            max_results=max_results,
                            enrich_top=enrich_top,
                            lang=lang,
                        )
                        if results:
                            break
                    product_matches[item_id] = {"queries": queries, "results": results}
                    st.session_state.project_state["product_matches"] = product_matches
                    st.rerun()
                continue

            st.caption("Top results (enriched when possible):")
            results = data.get("results", []) or []
            if not results:
                st.warning("No results.")
                continue

            selected_products = (
                st.session_state.project_state.get("selected_products", {}) or {}
            )
            current_sel = selected_products.get(item_id)

            for ridx, r in enumerate(results[:10]):
                title = r.get("title") or "(no title)"
                url = r.get("url") or ""
                domain = r.get("domain") or ""
                price = r.get("price")
                cur = r.get("currency")
                img = r.get("image")

                card = st.container()
                with card:
                    c1, c2 = st.columns([1, 3])
                    with c1:
                        if img:
                            try:
                                st.image(img, use_container_width=True)
                            except Exception:
                                st.write("🖼️ (image blocked)")
                    with c2:
                        st.markdown(f"**{title}**")
                        st.markdown(f"[Open]({url})  \n`{domain}`")
                        if price:
                            st.write(f"💰 {cur or ''} {price}")
                        if r.get("snippet"):
                            st.caption(r["snippet"][:240])

                        # select product
                        is_selected = (
                            current_sel is not None and current_sel.get("url") == url
                        )
                        if st.button(
                            ("✅ Selected" if is_selected else "Select this"),
                            key=f"sel_{item_id}_{ridx}",
                        ):
                            selected_products[item_id] = {
                                "url": url,
                                "title": title,
                                "domain": domain,
                                "price": price,
                                "currency": cur,
                                "image": img,
                                "raw": r,
                            }
                            st.session_state.project_state["selected_products"] = (
                                selected_products
                            )
                            st.rerun()

    st.divider()

    if st.button("⬅️ Back to Furniture Analysis"):
        st.session_state.project_state["phase"] = "furniture_analysis"
        st.rerun()


def render_preferences_phase():
    st.header("🧩 Preferences: Replace Furniture (per item)")

    base_design_b64 = st.session_state.project_state.get(
        "selected_design_image_before_prefs"
    ) or st.session_state.project_state.get("selected_design_image")
    if not base_design_b64:
        st.error("No base design image found. Please select a design first.")
        if st.button("⬅️ Back"):
            st.session_state.project_state["phase"] = "design"
            st.rerun()
        return

    furniture_items = st.session_state.project_state.get("furniture_items", [])
    if not furniture_items:
        st.error("No furniture items detected. Run furniture analysis first.")
        if st.button("⬅️ Back"):
            st.session_state.project_state["phase"] = "furniture_analysis"
            st.rerun()
        return

    overrides = st.session_state.project_state.get("furniture_overrides", {}) or {}
    selected_products = (
        st.session_state.project_state.get("selected_products", {}) or {}
    )

    st.caption(
        "Tip: Best results happen when you **select a product link** for the item and apply the edit with its mask."
    )

    colA, colB = st.columns([2, 1])
    with colA:
        st.image(
            base64.b64decode(base_design_b64),
            caption="Base image (before preferences)",
            use_container_width=True,
        )
    with colB:
        if st.session_state.project_state.get("selected_design_image_after_prefs"):
            st.image(
                base64.b64decode(
                    st.session_state.project_state["selected_design_image_after_prefs"]
                ),
                caption="After preferences",
                use_container_width=True,
            )
        else:
            st.info("No preference-applied image yet.")

    st.divider()

    # Per item override UI
    for it in furniture_items:
        item_id = str(it.get("id"))
        existing = overrides.get(item_id, {})

        with st.expander(
            f"#{item_id} — {it.get('name')} ({it.get('color')}, {it.get('material')})",
            expanded=False,
        ):
            enabled = st.checkbox(
                "Enable replacement for this item",
                value=bool(existing.get("enabled", False)),
                key=f"en_{item_id}",
            )

            mode = st.radio(
                "Replacement source",
                options=["Use selected product", "Use custom text prompt"],
                index=0 if existing.get("mode", "product") == "product" else 1,
                key=f"mode_{item_id}",
                horizontal=True,
            )

            extra = st.text_area(
                "Extra requirements (optional)",
                value=str(existing.get("text", "")),
                placeholder="e.g. 'low-profile, bouclé fabric, black metal legs' or 'replace with a walnut mid-century coffee table'",
                key=f"txt_{item_id}",
            )

            # Show selected product if any
            sel = selected_products.get(item_id)
            if mode == "Use selected product":
                if sel:
                    st.success(f"Selected product: {sel.get('title')}")
                    st.markdown(f"[Open]({sel.get('url')})")
                    if sel.get("image"):
                        try:
                            st.image(
                                sel["image"],
                                caption="Selected product image",
                                use_container_width=True,
                            )
                        except Exception:
                            pass
                else:
                    st.warning(
                        "No product selected for this item. Go to Products tab and select one."
                    )
            else:
                st.info(
                    "Custom prompt will be used as the replacement description (mask-limited)."
                )

            # Save back into session dict
            overrides[item_id] = {
                "enabled": bool(enabled),
                "mode": "product" if mode == "Use selected product" else "text",
                "text": extra.strip(),
            }

    st.session_state.project_state["furniture_overrides"] = overrides

    st.divider()

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("🧪 Apply Preferences (Replace Items)", type="primary"):
            progress = st.progress(0)
            status = st.empty()

            def p_cb(p, msg):
                progress.progress(float(p))
                status.text(msg)

            out = apply_overrides_sequentially(
                base_design_b64=base_design_b64,
                furniture_items=furniture_items,
                overrides=overrides,
                selected_products=selected_products,
                progress_cb=p_cb,
            )
            if out.get("success") and out.get("image_base64"):
                st.session_state.project_state["selected_design_image_after_prefs"] = (
                    out["image_base64"]
                )
                st.session_state.project_state["selected_design_image"] = out[
                    "image_base64"
                ]
                st.success("Preferences applied! (Mask-based edits)")
                st.rerun()
            else:
                st.error(
                    out.get(
                        "error", "Preference application failed (no image returned)."
                    )
                )

    with col2:
        if st.button("↩️ Reset to Base Image"):
            st.session_state.project_state["selected_design_image_after_prefs"] = None
            st.session_state.project_state["selected_design_image"] = base_design_b64
            st.success("Reset done.")
            st.rerun()

    with col3:
        if st.button("➡️ Next: BOM", type="primary"):
            st.session_state.project_state["phase"] = "bom"
            st.rerun()

    st.divider()
    if st.button("⬅️ Back to Products"):
        st.session_state.project_state["phase"] = "products"
        st.rerun()


def render_bom_phase():
    st.header("🧾 Bill of Materials (BOM)")

    furniture_items = st.session_state.project_state.get("furniture_items", [])
    selected_products = (
        st.session_state.project_state.get("selected_products", {}) or {}
    )
    overrides = st.session_state.project_state.get("furniture_overrides", {}) or {}

    if not furniture_items:
        st.error("No furniture items.")
        if st.button("⬅️ Back"):
            st.session_state.project_state["phase"] = "furniture_analysis"
            st.rerun()
        return

    bom_rows = []
    for it in furniture_items:
        item_id = str(it.get("id"))
        sel = selected_products.get(item_id)
        ov = overrides.get(item_id, {})
        enabled = bool(ov.get("enabled", False))
        mode = ov.get("mode", "")

        row = {
            "id": item_id,
            "name": it.get("name"),
            "color": it.get("color"),
            "material": it.get("material"),
            "style": it.get("style"),
            "replacement_enabled": enabled,
            "replacement_mode": mode,
            "selected_product_title": sel.get("title") if sel else None,
            "selected_product_url": sel.get("url") if sel else None,
            "price": sel.get("price") if sel else None,
            "currency": sel.get("currency") if sel else None,
        }
        bom_rows.append(row)

    st.session_state.project_state["bom"] = bom_rows

    # Display as simple markdown table-like via Streamlit
    st.caption("BOM is built from detected furniture + any selected products.")
    for row in bom_rows:
        with st.expander(f"#{row['id']} — {row['name']}", expanded=False):
            st.write(
                f"**Detected:** {row['color']} | {row['material']} | {row['style']}"
            )
            st.write(
                f"**Replacement enabled:** {row['replacement_enabled']} ({row['replacement_mode']})"
            )
            if row["selected_product_url"]:
                st.markdown(
                    f"**Selected product:** [{row['selected_product_title']}]({row['selected_product_url']})"
                )
                if row["price"]:
                    st.write(f"💰 {row.get('currency') or ''} {row.get('price')}")

    st.divider()
    if st.button("✅ Finish", type="primary"):
        st.session_state.project_state["phase"] = "complete"
        st.rerun()

    if st.button("⬅️ Back to Preferences"):
        st.session_state.project_state["phase"] = "preferences"
        st.rerun()


def render_complete_phase():
    st.header("✅ Complete")

    final_img_b64 = st.session_state.project_state.get("selected_design_image")
    if final_img_b64:
        st.image(
            base64.b64decode(final_img_b64),
            caption="Final Image",
            use_container_width=True,
        )

    bom = st.session_state.project_state.get("bom")
    if bom:
        st.subheader("BOM Export")
        bom_json = json.dumps(bom, indent=2)
        st.download_button(
            "⬇️ Download BOM (JSON)",
            data=bom_json,
            file_name="bom.json",
            mime="application/json",
        )

    furniture_json = st.session_state.project_state.get("furniture_json")
    if furniture_json:
        fj = json.dumps(furniture_json, indent=2)
        st.download_button(
            "⬇️ Download Furniture Analysis (JSON)",
            data=fj,
            file_name="furniture.json",
            mime="application/json",
        )

    st.divider()
    if st.button("🔄 Start New Project"):
        st.session_state.project_state = DEFAULT_STATE.copy()
        st.rerun()


# ============== MAIN ROUTER ==============


def render_sidebar_nav():
    st.sidebar.header("Navigation")

    phase = st.session_state.project_state.get("phase", "upload")
    phase_labels = {
        "upload": "Upload",
        "valuation": "Analysis",
        "design": "Design",
        "furniture_analysis": "Furniture",
        "products": "Products",
        "preferences": "Preferences",
        "bom": "BOM",
        "complete": "Complete",
    }

    st.sidebar.write(f"**Current:** {phase_labels.get(phase, phase)}")

    # Allow jumping backwards safely
    if st.sidebar.button("🏠 Upload"):
        st.session_state.project_state["phase"] = "upload"
        st.rerun()
    if st.sidebar.button("📊 Analysis"):
        st.session_state.project_state["phase"] = "valuation"
        st.rerun()
    if st.sidebar.button("🎨 Design"):
        st.session_state.project_state["phase"] = "design"
        st.rerun()
    if st.sidebar.button("🔍 Furniture"):
        st.session_state.project_state["phase"] = "furniture_analysis"
        st.rerun()
    if st.sidebar.button("🔗 Products"):
        st.session_state.project_state["phase"] = "products"
        st.rerun()
    if st.sidebar.button("🧩 Preferences"):
        st.session_state.project_state["phase"] = "preferences"
        st.rerun()
    if st.sidebar.button("🧾 BOM"):
        st.session_state.project_state["phase"] = "bom"
        st.rerun()
    if st.sidebar.button("✅ Complete"):
        st.session_state.project_state["phase"] = "complete"
        st.rerun()

    st.sidebar.divider()
    st.sidebar.caption(
        "Tip: If Gemini image quota is 0, furniture replacement edits will not work until quota is enabled."
    )


def main():
    render_sidebar_nav()
    render_header()

    phase = st.session_state.project_state.get("phase", "upload")

    if phase == "upload":
        render_upload_phase()
    elif phase == "valuation":
        render_valuation_phase()
    elif phase == "design":
        render_design_phase()
    elif phase == "furniture_analysis":
        render_furniture_analysis_phase()
    elif phase == "products":
        render_products_phase()
    elif phase == "preferences":
        render_preferences_phase()
    elif phase == "bom":
        render_bom_phase()
    elif phase == "complete":
        render_complete_phase()
    else:
        st.session_state.project_state["phase"] = "upload"
        st.rerun()


if __name__ == "__main__":
    main()
