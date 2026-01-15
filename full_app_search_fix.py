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
import tempfile

try:
    from google import genai as ggenai
    from google.genai.types import (
        Image as GenAIImage,
        RawReferenceImage,
        MaskReferenceImage,
        MaskReferenceConfig,
        EditImageConfig,
    )

    VERTEX_GENAI_AVAILABLE = True
except Exception:
    VERTEX_GENAI_AVAILABLE = False

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
IMAGEN_EDIT_MODEL = "imagen-3.0-capability-001"

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
    {
        "name": "Vivense",
        "search_url": "https://www.vivense.com/arama?q={query}",
        "icon": "🇹🇷",
    },
    {
        "name": "Trendyol",
        "search_url": "https://www.trendyol.com/sr?q={query}",
        "icon": "🧡",
    },
    {
        "name": "IKEA Türkiye",
        "search_url": "https://www.ikea.com.tr/arama?q={query}",
        "icon": "🟦",
    },
    {
        "name": "Hepsiburada",
        "search_url": "https://www.hepsiburada.com/ara?q={query}",
        "icon": "🛒",
    },
    {
        "name": "Evidea",
        "search_url": "https://www.evidea.com/arama?q={query}",
        "icon": "🏠",
    },
    {
        "name": "Enza Home",
        "search_url": "https://www.enzahome.com.tr/arama?q={query}",
        "icon": "🛋️",
    },
    {
        "name": "Doğtaş",
        "search_url": "https://www.dogtas.com/search?q={query}",
        "icon": "🪵",
    },
    {
        "name": "Mudo Concept",
        "search_url": "https://www.mudo.com.tr/search?q={query}",
        "icon": "✨",
    },
    {
        "name": "Kelebek Mobilya",
        "search_url": "https://www.kelebek.com/arama?q={query}",
        "icon": "🦋",
    },
    {
        "name": "İstikbal",
        "search_url": "https://www.istikbal.com.tr/arama?q={query}",
        "icon": "🏷️",
    },
    {
        "name": "Bellona",
        "search_url": "https://www.bellona.com.tr/arama?q={query}",
        "icon": "🏷️",
    },
    {
        "name": "Koçtaş",
        "search_url": "https://www.koctas.com.tr/arama?q={query}",
        "icon": "🧰",
    },
    {
        "name": "Modalife",
        "search_url": "https://www.modalife.com/arama?q={query}",
        "icon": "🛋️",
    },
    {
        "name": "Adore Mobilya",
        "search_url": "https://www.adoremobilya.com/arama?q={query}",
        "icon": "🪑",
    },
    {
        "name": "Lazzoni",
        "search_url": "https://www.lazzoni.com.tr/search?q={query}",
        "icon": "🖤",
    },
    {
        "name": "Yataş Bedding",
        "search_url": "https://www.yatasbedding.com.tr/search?q={query}",
        "icon": "🛏️",
    },
    {
        "name": "Alfemo",
        "search_url": "https://www.alfemo.com.tr/arama?q={query}",
        "icon": "🛋️",
    },
    {
        "name": "Rani Mobilya",
        "search_url": "https://www.rani.com.tr/search?q={query}",
        "icon": "🪑",
    },
    {
        "name": "Minar Mobilya",
        "search_url": "https://www.minarmobilya.com/search?q={query}",
        "icon": "🪵",
    },
    {
        "name": "İnegöl Mobilya Vadi",
        "search_url": "https://www.inegolmobilyavadi.com/arama?q={query}",
        "icon": "🏭",
    },
]


def _normalize_domain(netloc: str) -> str:
    d = (netloc or "").lower().strip()
    d = d.split(":")[0]
    if d.startswith("www."):
        d = d[4:]
    return d


def get_allowed_retailer_domains() -> List[str]:
    """
    Build an allowlist of retailer domains from the RETAILERS search URLs.
    Example: https://www.ikea.com/us/en/search/?q=... -> ikea.com
    """
    domains = set()
    for r in RETAILERS:
        try:
            u = r.get("search_url", "")
            netloc = urlparse(u).netloc
            d = _normalize_domain(netloc)
            if d:
                domains.add(d)
        except Exception:
            continue

    # Return stable ordering (nice for debugging)
    return sorted(domains)


def is_allowed_retailer_domain(domain: str, allowed_domains: List[str]) -> bool:
    """
    True if domain is exactly or is a subdomain of an allowed retailer domain.
    Example: 'www.ikea.com' matches 'ikea.com'
    """
    d = _normalize_domain(domain)
    if not d:
        return False
    for ad in allowed_domains:
        if d == ad or d.endswith("." + ad):
            return True
    return False


def build_site_restricted_query(query: str, allowed_domains: List[str]) -> str:
    """
    SearXNG typically supports search syntax like:
      (site:amazon.com OR site:ikea.com) <your query>
    """
    q = re.sub(r"\s+", " ", (query or "").strip())
    if not q or not allowed_domains:
        return q
    sites = " OR ".join([f"site:{d}" for d in allowed_domains])
    return f"({sites}) {q}".strip()


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


st.markdown("### Vertex AI (Imagen) settings (optional)")
if not VERTEX_GENAI_AVAILABLE:
    st.warning("google-genai not installed. Run: pip install google-genai google-auth")

use_vertex = st.checkbox(
    "Use Vertex AI Imagen for mask edits (instead of Gemini API key)",
    value=bool(st.session_state.get("use_vertex_ai", False)),
)
st.session_state.use_vertex_ai = use_vertex

v_project = st.text_input(
    "GCP Project ID (Vertex)",
    value=st.session_state.get(
        "vertex_project_id", os.getenv("GOOGLE_CLOUD_PROJECT", "")
    ),
    placeholder="e.g. my-gcp-project",
)
st.session_state.vertex_project_id = v_project

v_location = st.text_input(
    "GCP Location (Vertex)",
    value=st.session_state.get(
        "vertex_location", os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    ),
)
st.session_state.vertex_location = v_location


# ==================== COLOR ANALYSIS ====================


@st.cache_resource
def _load_depth_estimator():
    """
    Monocular depth estimation pipeline.
    pip install transformers torch
    """
    from transformers import pipeline

    # DPT is a common solid option; you can swap to another depth model if you want.
    return pipeline(task="depth-estimation", model="Intel/dpt-large")


def estimate_item_measurements_from_photo(
    image_pil: Image.Image,
    item_mask_bool: np.ndarray,
    item_box_xyxy: List[float],
    reference: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Estimate approximate item measurements from a single photo.

    Inputs:
      - image_pil: full RGB image (PIL)
      - item_mask_bool: (H,W) boolean mask for the furniture item (True=item)
      - item_box_xyxy: [x1,y1,x2,y2] bounding box in image pixels
      - reference (optional): a known real-world reference to scale estimates.
          Example:
            {"type":"door_height_cm", "value_cm":210}
            {"type":"tile_size_cm", "value_cm":60}   # if tiles are visible & roughly square
          NOTE: Without reference, returns only relative / heuristic estimates.

    Output:
      Dict with:
        - pixel_width, pixel_height
        - depth_stats (relative)
        - approx_width_cm, approx_height_cm (only if reference provided)
        - confidence_note
    """
    import numpy as np

    H, W = image_pil.height, image_pil.width
    x1, y1, x2, y2 = [int(v) for v in item_box_xyxy]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W - 1, x2), min(H - 1, y2)

    # Basic pixel dims
    pixel_w = max(1, x2 - x1)
    pixel_h = max(1, y2 - y1)

    # Depth map (relative)
    depth_pipe = _load_depth_estimator()
    depth_out = depth_pipe(image_pil)
    depth = depth_out["depth"]

    # depth can be PIL or numpy depending on transformers version
    if isinstance(depth, Image.Image):
        depth_np = np.array(depth).astype(np.float32)
    else:
        depth_np = np.array(depth).astype(np.float32)

    # Resize depth to image size if needed
    if depth_np.shape[0] != H or depth_np.shape[1] != W:
        depth_np = np.array(
            Image.fromarray(depth_np).resize((W, H), Image.BILINEAR)
        ).astype(np.float32)

    # Mask depth stats
    m = item_mask_bool.astype(bool)
    if m.shape != (H, W):
        # last resort: attempt resize (shouldn't happen with your full-size masks)
        m = (
            np.array(
                Image.fromarray(m.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)
            )
            > 127
        )

    masked_depth = depth_np[m]
    if masked_depth.size < 50:
        return {
            "pixel_width": pixel_w,
            "pixel_height": pixel_h,
            "depth_stats": None,
            "approx_width_cm": None,
            "approx_height_cm": None,
            "confidence_note": "Mask too small for depth-based estimation.",
        }

    d_med = float(np.median(masked_depth))
    d_p10 = float(np.percentile(masked_depth, 10))
    d_p90 = float(np.percentile(masked_depth, 90))

    result = {
        "pixel_width": pixel_w,
        "pixel_height": pixel_h,
        "depth_stats": {"median": d_med, "p10": d_p10, "p90": d_p90},
        "approx_width_cm": None,
        "approx_height_cm": None,
        "confidence_note": "Relative-only (no absolute scale) unless reference is provided.",
    }

    # Optional absolute scaling (very approximate)
    # We use a crude pinhole-like proportionality: real_size ~ pixel_size * scale(depth).
    # Without camera intrinsics this is approximate; reference helps anchor the scale.
    if reference and isinstance(reference, dict) and "value_cm" in reference:
        ref_cm = float(reference["value_cm"])
        ref_type = (reference.get("type") or "").lower()

        # Heuristic: treat the FULL image height as roughly reference if it's a door shot
        # Better: you can pass a reference box/mask and compute its pixel height instead.
        # This function keeps it simple.
        if "door" in ref_type:
            px_ref = float(H)
        else:
            px_ref = float(max(H, W))

        # Base cm-per-pixel
        cm_per_px = ref_cm / max(1.0, px_ref)

        # Depth factor: nearer objects appear larger; use inverse depth normalized
        # Normalize against global median depth (scene-level)
        scene_med = float(np.median(depth_np))
        depth_factor = scene_med / max(1e-6, d_med)  # >1 if item is closer than average

        approx_w_cm = pixel_w * cm_per_px * depth_factor
        approx_h_cm = pixel_h * cm_per_px * depth_factor

        result["approx_width_cm"] = round(float(approx_w_cm), 1)
        result["approx_height_cm"] = round(float(approx_h_cm), 1)
        result["confidence_note"] = (
            "Approximate absolute estimate using monocular depth + a coarse reference. "
            "For better accuracy: provide a reference object box (e.g., door bbox) or known camera intrinsics."
        )

    return result


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


def build_search_queries(item: Dict, lang: str = "en") -> List[str]:
    """
    Build optimized search queries for furniture items.
    If lang == 'tr', also generate Turkish query variants (koltuk/kanepe/...).
    """
    name = (item.get("name", "") or "").strip()
    color = (item.get("color", "") or "").strip()
    material = (item.get("material", "") or "").strip()
    style = (item.get("style", "") or "").strip()

    # --- helpers ---
    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip())

    def first_match_token(text: str, mapping: List[Tuple[str, str]]) -> str:
        t = (text or "").lower()
        for k, v in mapping:
            if k in t:
                return v
        return ""

    # English material tokens (your original behavior)
    material_keywords_en = [
        ("walnut", "walnut wood"),
        ("oak", "oak wood"),
        ("wood", "wooden"),
        ("metal", "metal"),
        ("leather", "leather"),
        ("fabric", "fabric upholstered"),
        ("jute", "jute"),
        ("glass", "glass"),
        ("velvet", "velvet"),
        ("linen", "linen"),
    ]
    mat_en = first_match_token(material, material_keywords_en)

    # Map detected label -> Turkish furniture noun (very important for TR sites)
    furniture_map_tr = [
        ("sectional", "köşe koltuk"),
        ("sofa", "koltuk"),
        ("couch", "koltuk"),
        ("loveseat", "ikili koltuk"),
        ("armchair", "berjer"),
        ("accent chair", "berjer"),
        ("chair", "sandalye"),
        ("ottoman", "puf"),
        ("coffee table", "orta sehpa"),
        ("side table", "yan sehpa"),
        ("end table", "yan sehpa"),
        ("console table", "konsol"),
        ("dining table", "yemek masası"),
        ("desk", "çalışma masası"),
        ("nightstand", "komodin"),
        ("dresser", "şifonyer"),
        ("bookshelf", "kitaplık"),
        ("shelving", "raf"),
        ("cabinet", "dolap"),
        ("tv stand", "tv ünitesi"),
        ("bed", "yatak"),
        ("headboard", "başlık"),
        ("floor lamp", "lambader"),
        ("table lamp", "abajur"),
        ("pendant", "sarkıt lamba"),
        ("chandelier", "avize"),
        ("rug", "halı"),
        ("carpet", "halı"),
        ("curtains", "perde"),
        ("drapes", "perde"),
        ("mirror", "ayna"),
        ("artwork", "tablo"),
        ("painting", "tablo"),
        ("plant", "bitki"),
        ("planter", "saksı"),
        ("vase", "vazo"),
    ]

    # Cheap color mapping for Turkish queries (keeps your detected color but adds common TR adjectives)
    color_map_tr = [
        ("greige", "grej"),
        ("off-white", "kırık beyaz"),
        ("ivory", "fildişi"),
        ("cream", "krem"),
        ("beige", "bej"),
        ("tan", "taba"),
        ("taupe", "vizon"),
        ("charcoal", "antrasit"),
        ("dark gray", "koyu gri"),
        ("light gray", "açık gri"),
        ("gray", "gri"),
        ("black", "siyah"),
        ("white", "beyaz"),
        ("navy", "lacivert"),
        ("blue", "mavi"),
        ("green", "yeşil"),
        ("pink", "pembe"),
        ("red", "kırmızı"),
        ("yellow", "sarı"),
        ("gold", "altın"),
        ("brass", "pirinç"),
        ("walnut", "ceviz"),
        ("oak", "meşe"),
    ]

    # Material hints for TR queries
    material_map_tr = [
        ("walnut", "ceviz"),
        ("oak", "meşe"),
        ("wood", "ahşap"),
        ("metal", "metal"),
        ("leather", "deri"),
        ("fabric", "kumaş"),
        ("velvet", "kadife"),
        ("linen", "keten"),
        ("glass", "cam"),
        ("jute", "jüt"),
        ("marble", "mermer"),
    ]

    name_l = name.lower()
    noun_tr = (
        first_match_token(name_l, furniture_map_tr) or name
    )  # fallback to original
    color_tr = first_match_token(color.lower(), color_map_tr) or color
    mat_tr = first_match_token(material.lower(), material_map_tr)

    # --- Build queries ---
    queries: List[str] = []

    # English queries (original idea)
    if color and mat_en:
        queries.append(f"{color} {mat_en} {name}")
    if color:
        queries.append(f"{color} {name}")
    if mat_en:
        queries.append(f"{mat_en} {name}")
    if style and "modern" in style.lower():
        queries.append(f"modern {name}")
    queries.append(f"buy {name}")
    queries.append(name)

    # Turkish queries (extra variants)
    if lang == "tr":
        # Add Turkish “buy intent” terms (these work better on TR e-commerce)
        buy_terms = [
            "fiyat",
            "satın al",
            "indirim",
            "kampanya",
            "trendyol",
            "hepsiburada",
        ]

        base_tr_parts = [p for p in [color_tr, mat_tr, noun_tr] if p]
        if base_tr_parts:
            queries.append(" ".join(base_tr_parts))

        # Add a few structured variants
        if noun_tr:
            if color_tr:
                queries.append(f"{color_tr} {noun_tr}")
            if mat_tr:
                queries.append(f"{mat_tr} {noun_tr}")
            queries.append(f"{noun_tr} fiyat")

        # Add intent boosters
        if noun_tr:
            for bt in buy_terms[:3]:
                queries.append(f"{noun_tr} {bt}")

    # Deduplicate while preserving order
    seen, out = set(), []
    for q in queries:
        q = norm(q)
        if q and q.lower() not in seen:
            out.append(q)
            seen.add(q.lower())

    return out[:8]  # allow a few more because TR variants help a lot


def generate_retailer_links(item: Dict, lang: str = "en") -> List[Dict]:
    queries = build_search_queries(item, lang=lang)
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


def is_blocked_domain(domain: str, blocklist: List[str]) -> bool:
    d = _normalize_domain(domain)
    for b in blocklist:
        b = b.strip().lower()
        if not b:
            continue
        # allow entries like "houzz.com/discussions" to behave like "houzz.com"
        b = b.split("/")[0]
        b = _normalize_domain(b)
        if d == b or d.endswith("." + b):
            return True
    return False


def searx_search(
    searx_base: str,
    query: str,
    max_results: int = 10,
    lang: str = "en",
    restrict_to_retailers: bool = True,
    allowed_domains: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Search using SearXNG with improved error handling and fallback strategies.
    """
    url = searx_base.rstrip("/") + "/search"

    if allowed_domains is None:
        allowed_domains = get_allowed_retailer_domains()

    query_variants = []

    if restrict_to_retailers and allowed_domains:
        sites_str = " OR ".join([f"site:{d}" for d in allowed_domains[:10]])
        query_variants.append(f"({sites_str}) {query}")
        query_variants.append(f"satın al {query}" if lang == "tr" else f"buy {query}")
        query_variants.append(f"{query} fiyat" if lang == "tr" else f"{query} price")

    query_variants.append(query)

    all_results = []

    for effective_query in query_variants:
        params = {
            "q": effective_query,
            "format": "json",
            "language": lang,
            "safesearch": 0,
            # KEY CHANGE: shopping works better than general for products
            "categories": "shopping" if lang in ("tr", "en") else "general",
        }

        try:
            headers = {
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json",
            }
            r = requests.get(url, params=params, timeout=30, headers=headers)
            if r.status_code != 200:
                continue

            data = r.json()

            for item in data.get("results", []):
                u = item.get("url")
                if not u:
                    continue

                domain = _normalize_domain(urlparse(u).netloc)

                if is_blocked_domain(domain, BLOCKLIST):
                    continue

                if restrict_to_retailers and allowed_domains:
                    if not is_allowed_retailer_domain(domain, allowed_domains):
                        continue

                result = {
                    "title": item.get("title", ""),
                    "url": u,
                    "snippet": item.get("content") or item.get("snippet", ""),
                    "domain": domain,
                    "thumbnail": item.get("thumbnail"),
                }

                if not any(r0["url"] == result["url"] for r0 in all_results):
                    all_results.append(result)

            if len(all_results) >= max_results:
                break

        except requests.exceptions.Timeout:
            st.warning(f"Search timeout for query: {effective_query[:50]}...")
            continue
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to SearXNG. Please check if it's running.")
            break
        except Exception as e:
            st.warning(f"Search error: {str(e)[:120]}")
            continue

    return all_results[:max_results]


# ============== PRODUCT ENRICHMENT ==============
def download_image_to_base64(
    url: str, timeout: int = 15, max_bytes: int = 6_000_000
) -> Optional[str]:
    if not url:
        return None
    try:
        r = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout, stream=True)
        r.raise_for_status()
        content = r.content
        if not content or len(content) > max_bytes:
            return None

        # Try to normalize to JPEG (safer for models)
        img = Image.open(io.BytesIO(content)).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return None


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
        result["final_url"] = r.url
        result["status_code"] = r.status_code
        if r.status_code in (403, 429):
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


@st.cache_data(ttl=60 * 60)  # 1 hour cache
def cached_searx_search(*args, **kwargs):
    return searx_search(*args, **kwargs)


@st.cache_data(ttl=24 * 60 * 60)  # 24h cache
def cached_enrich_product(url: str, timeout: int = 15) -> Dict:
    return enrich_product(url, timeout=timeout)


def generate_fallback_results(query: str, domains: List[str]) -> List[Dict]:
    """
    Generate direct retailer search links as fallback when SearXNG fails.
    """
    encoded_query = quote_plus(query)
    results = []

    # Map domains to their search URL patterns
    domain_search_patterns = {
        "amazon.com": f"https://www.amazon.com/s?k={encoded_query}&i=garden",
        "wayfair.com": f"https://www.wayfair.com/keyword.html?keyword={encoded_query}",
        "ikea.com": f"https://www.ikea.com/us/en/search/?q={encoded_query}",
        "target.com": f"https://www.target.com/s?searchTerm={encoded_query}",
        "walmart.com": f"https://www.walmart.com/search?q={encoded_query}",
        "overstock.com": f"https://www.overstock.com/Home-Garden/?keywords={encoded_query}",
        "westelm.com": f"https://www.westelm.com/search/?query={encoded_query}",
        "cb2.com": f"https://www.cb2.com/search/?query={encoded_query}",
        "article.com": f"https://www.article.com/search?query={encoded_query}",
        "potterybarn.com": f"https://www.potterybarn.com/search/results.html?words={encoded_query}",
    }

    for domain in domains:
        if domain in domain_search_patterns:
            results.append(
                {
                    "title": f"Search '{query}' on {domain}",
                    "url": domain_search_patterns[domain],
                    "snippet": f"Click to search for {query} on {domain}",
                    "domain": domain,
                    "is_fallback": True,
                }
            )

    return results[:6]


def search_with_enrichment(
    searx_base: str,
    query: str,
    max_results: int = 6,
    enrich_top: int = 2,
    lang: str = "en",
    restrict_to_retailers: bool = True,
    allowed_domains: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Search and enrich results with price/image data.
    """
    if allowed_domains is None:
        allowed_domains = get_allowed_retailer_domains()

    # Get search results
    results = searx_search(
        searx_base=searx_base,
        query=query,
        max_results=max_results * 2,  # Get extra to filter
        lang=lang,
        restrict_to_retailers=restrict_to_retailers,
        allowed_domains=allowed_domains,
    )

    if not results:
        # Fallback: Generate direct retailer links
        return generate_fallback_results(query, allowed_domains[:5])

    # Score and sort results to prioritize product pages
    def product_url_score(url: str) -> int:
        u = (url or "").lower()
        score = 0

        # Positive signals for product pages
        good_patterns = [
            "/p/",
            "/product/",
            "/products/",
            "/item/",
            "/dp/",
            "/urun/",
            "/pd/",
            "sku=",
            "productid=",
            "pid=",
            "/buy/",
            "/shop/",
        ]

        # Negative signals for non-product pages
        bad_patterns = [
            "/search",
            "/s?",
            "/category",
            "/collections",
            "/c/",
            "/katalog",
            "/liste",
            "/list",
            "/browse",
            "?q=",
            "&q=",
            "sort=",
            "filter=",
            "/blog",
            "/article",
        ]

        for g in good_patterns:
            if g in u:
                score += 3

        for b in bad_patterns:
            if b in u:
                score -= 5

        return score

    results.sort(key=lambda x: product_url_score(x.get("url", "")), reverse=True)
    results = results[:max_results]

    # Enrich top results
    for i, result in enumerate(results[:enrich_top]):
        url = result.get("url")
        if url:
            try:
                enriched = enrich_product(url, timeout=10)
                result["enriched"] = enriched

                if enriched.get("image"):
                    result["image"] = enriched["image"]
                if enriched.get("price"):
                    result["price"] = enriched["price"]
                    result["currency"] = enriched.get("currency", "USD")
                if enriched.get("title") and len(enriched["title"]) > len(
                    result.get("title", "")
                ):
                    result["title"] = enriched["title"]

            except Exception as e:
                result["enriched"] = {"error": str(e)}

            # Small delay to be polite
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


def _genai_image_from_bytes(data: bytes, mime_type: str):
    """
    Robust constructor for google.genai.types.Image across SDK versions.

    Newer SDKs: Image.from_bytes(...)
    Older SDKs: Image(image_bytes=..., mime_type=...)
    Fallback: write temp file + Image.from_file(...)
    """
    from google.genai import types

    # 1) Best case: official helper exists
    if hasattr(types.Image, "from_bytes"):
        return types.Image.from_bytes(data=data, mime_type=mime_type)

    # 2) Most common pydantic schema (what your error indicates you need)
    for kwargs in (
        {"image_bytes": data, "mime_type": mime_type},
        {"image_bytes": data},
        {"bytes": data, "mime_type": mime_type},  # some variants
        {"data": data, "mime_type": mime_type},  # last resort (some very old variants)
    ):
        try:
            return types.Image(**kwargs)
        except Exception:
            # IMPORTANT: catch ValidationError too (not only TypeError)
            continue

    # 3) Last fallback: save temp file and use Image.from_file (documented in Google samples)
    if hasattr(types.Image, "from_file"):
        ext = ".png" if "png" in (mime_type or "").lower() else ".jpg"
        import tempfile, os

        fd, path = tempfile.mkstemp(suffix=ext)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            return types.Image.from_file(location=path)
        finally:
            try:
                os.remove(path)
            except Exception:
                pass

    raise RuntimeError(
        "Could not construct google.genai.types.Image from bytes. "
        "Your google-genai SDK schema does not match expected constructors."
    )


def generate_design_with_gemini(
    image_base64: str, style: str, variation: str, room_type: str = "room"
) -> dict:
    """Generate a redesigned room image using Gemini."""
    api_key = get_api_key("GEMINI_API_KEY")
    if not api_key:
        return {"success": False, "error": "Gemini API key not configured"}

    style_descriptions = {
        "Modern Minimalist": "modern minimalist style with clean lines, neutral colors, minimal furniture, abundant natural light, uncluttered spaces",
        "Scandinavian": "Scandinavian style with light oak wood, white walls, cozy textiles, indoor plants, warm ambient lighting, hygge atmosphere",
        "Industrial": "industrial loft style with exposed brick walls, black metal frames, Edison bulbs, leather furniture, raw materials",
        "Mid-Century Modern": "mid-century modern with walnut furniture, organic curved shapes, mustard and teal accents, iconic design pieces",
        "Contemporary": "contemporary style with mixed materials, neutral base colors, bold accent pieces, artistic elements",
        "Bohemian": "bohemian style with colorful patterns, layered textiles, macrame hangings, abundant plants, eclectic mix",
        "Coastal": "coastal style with blue and white palette, rattan furniture, nautical accents, light and airy feel",
        "Farmhouse": "modern farmhouse with shiplap walls, rustic reclaimed wood, neutral tones, vintage accents",
    }

    variation_descriptions = {
        "Light & Airy": "bright atmosphere with maximum natural light, white and cream palette, open and spacious feel",
        "Warm & Cozy": "warm inviting atmosphere with ambient lighting, earth tones, soft textures, intimate feel",
        "Bold & Dramatic": "dramatic atmosphere with deep rich colors, high contrast, statement pieces, luxurious feel",
    }

    style_desc = style_descriptions.get(style, "beautifully designed interior")
    variation_desc = variation_descriptions.get(
        variation, "harmonious balanced atmosphere"
    )

    prompt = f"""Transform this {room_type} into a stunning, professionally designed interior space.

DESIGN STYLE: {style_desc}
ATMOSPHERE: {variation_desc}

CRITICAL REQUIREMENTS:
1. PRESERVE the exact room structure - all walls, windows, doors, ceiling height, and floor plan must remain identical
2. MAINTAIN the same camera angle, perspective, and viewpoint
3. ADD appropriate furniture, decor, artwork, lighting fixtures, rugs, and plants
4. ENSURE photorealistic quality suitable for a high-end interior design magazine
5. CREATE a cohesive, harmonious design where all elements complement each other
6. APPLY appropriate colors, textures, and materials consistent with the style

Generate the beautifully redesigned room image now."""

    model_variants = [
        "gemini-2.0-flash-exp-image-generation",
        "gemini-2.0-flash-exp",
    ]

    parts = [
        {"text": prompt},
        {"inlineData": {"mimeType": "image/jpeg", "data": image_base64}},
    ]

    return _gemini_generate_content(
        parts, model_variants, api_key, temperature=0.85, timeout=180
    )


def replace_furniture_with_gemini_inpaint(
    base_image_b64: str,
    mask_b64: str,
    item_name: str,
    replacement_description: str,
    product_image_b64: Optional[str] = None,
) -> Dict:
    """
    Replace furniture using Gemini with mask guidance.
    Since Gemini doesn't have true inpainting, we use a multi-image prompt approach.
    """
    api_key = get_api_key("GEMINI_API_KEY")
    if not api_key:
        return {"success": False, "error": "Gemini API key not configured"}

    # Build a detailed prompt for targeted editing
    if product_image_b64:
        prompt = f"""You are an expert interior designer and photo editor.

TASK: Edit the room image to replace the {item_name} with the furniture shown in the reference product image.

INSTRUCTIONS:
1. Look at the MASK image - the WHITE area shows exactly where the {item_name} is located
2. REPLACE ONLY the furniture in the white masked area with the product shown in the reference image
3. Match the style, color, and design of the reference product as closely as possible
4. Keep EVERYTHING outside the white mask area EXACTLY the same - do not change walls, floor, other furniture, lighting, or any other elements
5. Ensure the new furniture fits naturally in the space with correct perspective and lighting
6. The result should look like a professional interior photo

REPLACEMENT: {replacement_description}

Generate the edited room image with the new furniture in place."""

        parts = [
            {"text": prompt},
            {"text": "Room image to edit:"},
            {"inlineData": {"mimeType": "image/jpeg", "data": base_image_b64}},
            {"text": "Mask showing furniture location (white = area to replace):"},
            {"inlineData": {"mimeType": "image/png", "data": mask_b64}},
            {"text": "Reference product to use as replacement:"},
            {"inlineData": {"mimeType": "image/jpeg", "data": product_image_b64}},
        ]
    else:
        prompt = f"""You are an expert interior designer and photo editor.

TASK: Edit the room image to replace the {item_name} based on the description.

INSTRUCTIONS:
1. Look at the MASK image - the WHITE area shows exactly where the {item_name} is located
2. REPLACE ONLY the furniture in the white masked area with: {replacement_description}
3. Keep EVERYTHING outside the white mask area EXACTLY the same
4. Ensure the new furniture fits naturally with correct perspective, lighting, and shadows
5. The result should look like a professional interior photo

Generate the edited room image."""

        parts = [
            {"text": prompt},
            {"text": "Room image to edit:"},
            {"inlineData": {"mimeType": "image/jpeg", "data": base_image_b64}},
            {"text": "Mask showing furniture location (white = area to replace):"},
            {"inlineData": {"mimeType": "image/png", "data": mask_b64}},
        ]

    model_variants = [
        "gemini-2.0-flash-exp-image-generation",
        "gemini-2.0-flash-exp",
    ]

    return _gemini_generate_content(
        parts, model_variants, api_key, temperature=0.7, timeout=240
    )


def replace_furniture_composite_approach(
    base_image_b64: str,
    furniture_items: List[Dict],
    overrides: Dict[str, Dict],
    selected_products: Dict[str, Dict],
) -> Dict:
    """
    Alternative approach: Generate a completely new design that incorporates the selected products.
    This is more reliable than mask-based editing when the API doesn't support true inpainting.
    """
    api_key = get_api_key("GEMINI_API_KEY")
    if not api_key:
        return {"success": False, "error": "Gemini API key not configured"}

    # Build a description of all furniture replacements
    replacements = []
    product_images = []

    for item in furniture_items:
        item_id = str(item.get("id"))
        override = overrides.get(item_id, {})

        if not override.get("enabled"):
            continue

        item_name = item.get("name", "furniture")
        position = item.get("position", "")

        if override.get("mode") == "product":
            product = selected_products.get(item_id, {})
            if product:
                title = product.get("title", "selected product")
                replacements.append(
                    f"- Replace the {item_name} at {position} with: {title}"
                )

                # Add product image if available
                prod_img = product.get("image_b64")
                if prod_img:
                    product_images.append(
                        {
                            "item": item_name,
                            "image": prod_img,
                        }
                    )
        else:
            custom_text = override.get("text", "")
            if custom_text:
                replacements.append(
                    f"- Replace the {item_name} at {position} with: {custom_text}"
                )

    if not replacements:
        return {
            "success": True,
            "image_base64": base_image_b64,
            "message": "No changes requested",
        }

    replacement_text = "\n".join(replacements)

    prompt = f"""You are redesigning this room by replacing specific furniture items.

FURNITURE REPLACEMENTS TO MAKE:
{replacement_text}

CRITICAL INSTRUCTIONS:
1. Keep the exact same room structure - walls, windows, doors, floor, ceiling
2. Keep the same camera angle and perspective
3. Only replace the specific furniture items mentioned above
4. Keep all other furniture and decor exactly as they are
5. Ensure replacements fit naturally with correct scale, perspective, and lighting
6. Produce a photorealistic result

Generate the room with the furniture replacements applied."""

    parts = [
        {"text": prompt},
        {"text": "Original room:"},
        {"inlineData": {"mimeType": "image/jpeg", "data": base_image_b64}},
    ]

    # Add product reference images
    for i, prod in enumerate(product_images[:3]):  # Limit to 3 products
        parts.append({"text": f"Reference for {prod['item']}:"})
        parts.append({"inlineData": {"mimeType": "image/jpeg", "data": prod["image"]}})

    model_variants = [
        "gemini-2.0-flash-exp-image-generation",
        "gemini-2.0-flash-exp",
    ]

    return _gemini_generate_content(
        parts, model_variants, api_key, temperature=0.75, timeout=240
    )


def _first_existing_attr(obj, names):
    for n in names:
        if hasattr(obj, n):
            v = getattr(obj, n)
            if v is not None:
                return v
    return None


def _extract_image_bytes_from_vertex_response(resp) -> Optional[bytes]:
    """
    google-genai SDK responses vary by version.
    Try hard to pull bytes out of common shapes.
    """
    # 1) Most recent: resp.generated_images -> [{image: Image}]
    imgs = _first_existing_attr(
        resp, ["generated_images", "images", "output", "outputs"]
    )
    if imgs:
        try:
            first = imgs[0]
            img_obj = (
                _first_existing_attr(first, ["image", "generated_image", "img"])
                or first
            )

            # Image bytes might be in image_bytes / data / bytes
            b = _first_existing_attr(img_obj, ["image_bytes", "data", "bytes"])
            if isinstance(b, (bytes, bytearray)) and len(b) > 1000:
                return bytes(b)

            # Sometimes nested dict-like
            if isinstance(img_obj, dict):
                for k in ["image_bytes", "data", "bytes"]:
                    v = img_obj.get(k)
                    if isinstance(v, (bytes, bytearray)) and len(v) > 1000:
                        return bytes(v)
        except Exception:
            pass

    # 2) Some versions: resp.candidates[0].content.parts[...].inline_data
    cands = _first_existing_attr(resp, ["candidates", "candidate", "responses"])
    if cands:
        try:
            c0 = cands[0]
            content = _first_existing_attr(c0, ["content", "output", "message"])
            parts = _first_existing_attr(content, ["parts", "part"])
            if parts:
                for p in parts:
                    inline = _first_existing_attr(p, ["inline_data", "inlineData"])
                    if inline:
                        data = _first_existing_attr(inline, ["data", "bytes"])
                        if isinstance(data, (bytes, bytearray)) and len(data) > 1000:
                            return bytes(data)
                        if isinstance(data, str) and len(data) > 1000:
                            # sometimes base64 string
                            try:
                                return base64.b64decode(data)
                            except Exception:
                                pass
        except Exception:
            pass

    return None


def replace_furniture_with_mask_vertex_imagen(
    base_image_b64_jpeg: str,
    mask_png_b64: str,
    replace_prompt: str,
    item_name: str,
    product_image_b64_jpeg: Optional[str] = None,
) -> Dict:
    if not VERTEX_GENAI_AVAILABLE:
        return {
            "success": False,
            "error": "google-genai not installed (pip install google-genai google-auth)",
        }

    project_id = (
        st.session_state.get("vertex_project_id")
        or os.getenv("GOOGLE_CLOUD_PROJECT")
        or ""
    ).strip()
    location = (
        st.session_state.get("vertex_location")
        or os.getenv("GOOGLE_CLOUD_LOCATION")
        or "us-central1"
    ).strip()

    if not project_id:
        return {
            "success": False,
            "error": "Vertex project_id missing. Set GOOGLE_CLOUD_PROJECT or enter it in UI.",
        }

    client = ggenai.Client(vertexai=True, project=project_id, location=location)

    strict_prompt = f"""You are editing a photorealistic interior design photo.

TASK:
Replace ONLY the furniture inside the provided MASK region with:
{replace_prompt}

HARD CONSTRAINTS:
- MODIFY ONLY pixels inside the MASK (white area). DO NOT change anything outside the mask.
- DO NOT change room structure: walls, floor, ceiling, windows, doors, architecture.
- DO NOT change camera angle, perspective, lens, or framing.
- Keep lighting and shadows consistent with the scene.
- Produce a photorealistic result.

Masked object category: {item_name}.
Return the edited full image.
"""

    try:
        from google.genai import types
        from google.genai.types import RawReferenceImage, MaskReferenceImage
    except Exception as e:
        return {"success": False, "error": f"google-genai import error: {e}"}

    # decode bytes
    base_bytes = base64.b64decode(base_image_b64_jpeg)
    mask_bytes = base64.b64decode(mask_png_b64)
    prod_bytes = (
        base64.b64decode(product_image_b64_jpeg) if product_image_b64_jpeg else None
    )

    try:
        base_img = _genai_image_from_bytes(base_bytes, "image/jpeg")
        mask_img = _genai_image_from_bytes(mask_bytes, "image/png")
        prod_img = (
            _genai_image_from_bytes(prod_bytes, "image/jpeg") if prod_bytes else None
        )
    except Exception as e:
        return {"success": False, "error": f"Image construction failed: {e}"}

    # Use enums if present, otherwise fallback to strings
    try:
        mask_mode = types.MaskReferenceConfig.MaskMode.MASK_MODE_USER_PROVIDED
    except Exception:
        mask_mode = "MASK_MODE_USER_PROVIDED"

    try:
        edit_mode = types.EditImageConfig.EditMode.EDIT_MODE_INPAINT_INSERTION
    except Exception:
        edit_mode = "EDIT_MODE_INPAINT_INSERTION"

    reference_images = [
        RawReferenceImage(reference_id=1, reference_image=base_img),
        MaskReferenceImage(
            reference_id=2,
            reference_image=mask_img,
            config=types.MaskReferenceConfig(
                mask_mode=mask_mode,
                mask_dilation=3,
            ),
        ),
    ]

    if prod_img is not None:
        reference_images.append(
            RawReferenceImage(reference_id=3, reference_image=prod_img)
        )

    try:
        resp = client.models.edit_image(
            model=IMAGEN_EDIT_MODEL,
            prompt=strict_prompt,
            reference_images=reference_images,
            config=types.EditImageConfig(
                edit_mode=edit_mode,
                number_of_images=1,
                output_mime_type="image/jpeg",
            ),
        )
    except Exception as e:
        return {"success": False, "error": f"Vertex edit_image call failed: {e}"}

    img_bytes = _extract_image_bytes_from_vertex_response(resp)
    if not img_bytes:
        # last resort: attempt stringification debug (short)
        return {
            "success": False,
            "error": "Vertex returned a response but no image bytes could be extracted.",
        }

    return {
        "success": True,
        "image_base64": base64.b64encode(img_bytes).decode("utf-8"),
        "method": IMAGEN_EDIT_MODEL,
    }


def replace_furniture_with_mask_gemini(
    base_image_b64_jpeg: str,
    mask_png_b64: str,
    replace_prompt: str,
    item_name: str,
    product_image_b64_jpeg: Optional[str] = None,  # NEW
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

    strict_prompt += "\nIf a reference product image is provided, match it as closely as possible for the object inside the mask."

    parts = [
        {"text": strict_prompt},
        {"inlineData": {"mimeType": "image/jpeg", "data": base_image_b64_jpeg}},
        {"inlineData": {"mimeType": "image/png", "data": mask_png_b64}},
    ]

    if product_image_b64_jpeg:
        parts.append(
            {"inlineData": {"mimeType": "image/jpeg", "data": product_image_b64_jpeg}}
        )

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
    Apply user furniture replacement preferences.
    Uses a hybrid approach: tries mask-based first, falls back to composite.
    """
    # Check if any overrides are enabled
    enabled_overrides = [
        (str(it.get("id")), it)
        for it in furniture_items
        if overrides.get(str(it.get("id")), {}).get("enabled")
    ]

    if not enabled_overrides:
        return {
            "success": True,
            "image_base64": base_design_b64,
            "message": "No changes requested",
        }

    use_vertex = bool(st.session_state.get("use_vertex_ai", False))

    # Try sequential mask-based approach first
    current_b64 = base_design_b64
    any_success = False

    total = len(enabled_overrides)

    for idx, (item_id, item) in enumerate(enabled_overrides):
        if progress_cb:
            progress_cb(idx / total, f"Processing {item.get('name', 'item')}...")

        override = overrides.get(item_id, {})
        mask_b64 = item.get("mask_png_base64")

        if not mask_b64:
            continue

        # Build replacement description
        mode = override.get("mode", "text")
        product = selected_products.get(item_id, {}) if mode == "product" else {}
        custom_text = override.get("text", "").strip()

        if mode == "product" and product:
            title = product.get("title", "the selected product")
            replacement_desc = f"furniture matching: {title}"
            if custom_text:
                replacement_desc += f". Additional requirements: {custom_text}"
            product_img = product.get("image_b64")
        else:
            replacement_desc = (
                custom_text if custom_text else "a modern stylish replacement"
            )
            product_img = None

        # Try mask-based replacement
        if use_vertex:
            result = replace_furniture_with_mask_vertex_imagen(
                base_image_b64_jpeg=current_b64,
                mask_png_b64=mask_b64,
                replace_prompt=replacement_desc,
                item_name=item.get("name", "furniture"),
                product_image_b64_jpeg=product_img,
            )
        else:
            result = replace_furniture_with_mask_gemini(
                base_image_b64_jpeg=current_b64,
                mask_png_b64=mask_b64,
                replace_prompt=replacement_desc,
                item_name=item.get("name", "furniture"),
                product_image_b64_jpeg=product_img,
                temperature=0.65,
            )

        if result.get("success") and result.get("image_base64"):
            current_b64 = result["image_base64"]
            any_success = True

        time.sleep(1)  # Rate limiting

    # If mask-based approach didn't work, try composite approach
    if not any_success:
        if progress_cb:
            progress_cb(0.5, "Trying alternative approach...")

        result = replace_furniture_composite_approach(
            base_image_b64=base_design_b64,
            furniture_items=furniture_items,
            overrides=overrides,
            selected_products=selected_products,
        )

        if result.get("success") and result.get("image_base64"):
            current_b64 = result["image_base64"]

    if progress_cb:
        progress_cb(1.0, "✅ Complete!")

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

    # Run analysis once
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

    furniture_items = st.session_state.project_state.get("furniture_items", []) or []
    if not furniture_items:
        st.warning("No furniture items to show yet.")
        if st.button("⬅️ Back to Designs"):
            st.session_state.project_state["phase"] = "design"
            st.rerun()
        return

    st.success(f"Detected {len(furniture_items)} furniture items.")

    # Decode the design image once (used by measurement estimation)
    try:
        design_bytes = base64.b64decode(design_image_b64)
        design_pil = Image.open(io.BytesIO(design_bytes)).convert("RGB")
    except Exception:
        design_pil = None

    # --- Display each item ---
    for idx, item in enumerate(furniture_items):
        item_id = str(item.get("id", idx + 1))
        item_name = item.get("name", "Unknown")
        conf = item.get("confidence", "N/A")

        with st.expander(f"#{item_id} — {item_name} (conf {conf}%)", expanded=False):
            c1, c2 = st.columns(2)

            with c1:
                st.write(
                    f"**Color:** {item.get('color')} ({item.get('colors_detail', {}).get('primary_hex', '')})"
                )
                st.write(f"**Material:** {item.get('material')}")
                st.write(f"**Style:** {item.get('style')}")
                st.write(
                    f"**Position:** {item.get('position')} | "
                    f"**Size:** {item.get('size')} | "
                    f"**Area:** {item.get('area_percent')}%"
                )
                st.write(f"**Description:** {item.get('full')}")

                # ✅ SAFE measurement access (never uses undefined `it`)
                meas = item.get("measurements") or {}
                if meas:
                    st.markdown("**📏 Measurements (estimated):**")
                    st.write(
                        f"- Pixel W×H: {meas.get('pixel_width')} × {meas.get('pixel_height')}\n"
                        f"- Approx W/H (cm): {meas.get('approx_width_cm')} / {meas.get('approx_height_cm')}\n"
                        f"- Note: {meas.get('confidence_note')}"
                    )
                else:
                    st.caption("No measurements yet.")

                # Optional: estimate measurements button (per item)
                if st.button(
                    "📏 Estimate measurements for this item", key=f"meas_btn_{item_id}"
                ):
                    if design_pil is None:
                        st.error(
                            "Could not read the design image for measurement estimation."
                        )
                    else:
                        mask_b64 = item.get("mask_png_base64")
                        box = item.get("box")

                        if not mask_b64 or not box:
                            st.warning(
                                "Missing mask or box for this item; cannot estimate."
                            )
                        else:
                            try:
                                mask_img = Image.open(
                                    io.BytesIO(base64.b64decode(mask_b64))
                                ).convert("L")
                                mask_bool = np.array(mask_img) > 127

                                # Optional reference (leave None to keep relative-only)
                                reference = None
                                # Example if you want: reference = {"type": "door_height_cm", "value_cm": 210}

                                meas_out = estimate_item_measurements_from_photo(
                                    image_pil=design_pil,
                                    item_mask_bool=mask_bool,
                                    item_box_xyxy=box,
                                    reference=reference,
                                )

                                # Write back into session state list
                                st.session_state.project_state["furniture_items"][idx][
                                    "measurements"
                                ] = meas_out
                                st.success("Measurements stored on this item.")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Measurement estimation failed: {e}")

            with c2:
                # Mask preview
                m_b64 = item.get("mask_png_base64")
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
                else:
                    st.info("No mask available for this item.")

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
    """Render the product search phase with improved error handling and fallbacks."""
    st.header("🔗 Product Search")

    furniture_items = st.session_state.project_state.get("furniture_items", [])
    if not furniture_items:
        st.error("No furniture items available. Please run furniture analysis first.")
        if st.button("⬅️ Back to Furniture Analysis"):
            st.session_state.project_state["phase"] = "furniture_analysis"
            st.rerun()
        return

    # Search configuration
    with st.expander("⚙️ Search Settings", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            searx_url = st.text_input(
                "SearXNG Base URL",
                value=st.session_state.get("searx_url", SEARX_DEFAULT_URL),
                help="URL of your local SearXNG instance",
            )
            st.session_state.searx_url = searx_url

            # Test connection button
            if st.button("🔌 Test Connection"):
                try:
                    r = requests.get(f"{searx_url}/healthz", timeout=5)
                    if r.status_code == 200:
                        st.success("✅ SearXNG is reachable!")
                    else:
                        st.warning(f"⚠️ SearXNG responded with status {r.status_code}")
                except:
                    st.error(
                        "❌ Cannot connect to SearXNG. Using fallback retailer links."
                    )

        with col2:
            lang = st.selectbox("Language", options=["en", "tr", "de", "fr"], index=0)
            max_results = st.slider("Max results per item", 3, 12, 6)
            enrich_top = st.slider("Enrich top N results", 0, 4, 2)

        restrict_to_retailers = st.checkbox(
            "Restrict to known retailers only",
            value=True,
            help="Only show results from major furniture retailers",
        )

    product_matches = st.session_state.project_state.get("product_matches", {}) or {}
    selected_products = (
        st.session_state.project_state.get("selected_products", {}) or {}
    )

    # Search controls
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("🔍 Search All Items", type="primary"):
            progress = st.progress(0)
            status = st.empty()

            allowed_domains = (
                get_allowed_retailer_domains() if restrict_to_retailers else None
            )
            total = len(furniture_items)

            for idx, item in enumerate(furniture_items):
                item_id = str(item.get("id"))
                item_name = item.get("name", "item")

                status.text(f"Searching for {item_name}... ({idx + 1}/{total})")

                queries = build_search_queries(item, lang=lang)
                results = []

                # Try each query until we get results
                for query in queries[:3]:
                    results = search_with_enrichment(
                        searx_base=searx_url,
                        query=query,
                        max_results=max_results,
                        enrich_top=enrich_top,
                        lang=lang,
                        restrict_to_retailers=restrict_to_retailers,
                        allowed_domains=allowed_domains,
                    )

                    if results:
                        break
                    time.sleep(0.3)

                product_matches[item_id] = {
                    "queries": queries,
                    "results": results,
                    "search_successful": len(results) > 0,
                }

                progress.progress((idx + 1) / total)
                time.sleep(0.2)

            st.session_state.project_state["product_matches"] = product_matches
            status.text("✅ Search complete!")
            st.rerun()

    with col2:
        if st.button("🔄 Clear All Searches"):
            st.session_state.project_state["product_matches"] = {}
            st.session_state.project_state["selected_products"] = {}
            st.rerun()

    with col3:
        if st.button("➡️ Continue to Preferences", type="primary"):
            st.session_state.project_state["phase"] = "preferences"
            st.rerun()

    st.divider()

    # Display results for each furniture item
    for item in furniture_items:
        item_id = str(item.get("id"))
        item_name = item.get("name", "Unknown")
        item_color = item.get("color", "")
        item_material = item.get("material", "")

        # Check if this item has a selected product
        is_selected = item_id in selected_products
        selection_indicator = "✅ " if is_selected else ""

        with st.expander(
            f"{selection_indicator}#{item_id} — {item_name} | {item_color} | {item_material}",
            expanded=not is_selected,
        ):
            # Quick retailer links
            st.markdown("**🛒 Quick Search Links:**")
            retailer_links = generate_retailer_links(item, lang=lang)

            link_cols = st.columns(5)
            for i, link in enumerate(retailer_links[:10]):
                with link_cols[i % 5]:
                    st.markdown(f"[{link['icon']} {link['retailer']}]({link['url']})")

            st.divider()

            # Search results
            match_data = product_matches.get(item_id, {})
            results = match_data.get("results", [])

            if not match_data:
                st.info("Click 'Search All Items' or search individually below.")

                if st.button(f"🔍 Search this item", key=f"search_{item_id}"):
                    with st.spinner(f"Searching for {item_name}..."):
                        queries = build_search_queries(item, lang=lang)
                        results = []

                        for query in queries[:3]:
                            results = search_with_enrichment(
                                searx_base=st.session_state.searx_url,
                                query=query,
                                max_results=max_results,
                                enrich_top=enrich_top,
                                lang=lang,
                                restrict_to_retailers=restrict_to_retailers,
                            )
                            if results:
                                break

                        product_matches[item_id] = {
                            "queries": queries,
                            "results": results,
                            "search_successful": len(results) > 0,
                        }
                        st.session_state.project_state["product_matches"] = (
                            product_matches
                        )
                        st.rerun()

            elif not results:
                st.warning("No products found. Try the retailer links above.")

            else:
                st.markdown(f"**Found {len(results)} products:**")

                current_selection = selected_products.get(item_id)

                for ridx, result in enumerate(results):
                    title = result.get("title", "Unknown Product")[:80]
                    url = result.get("url", "")
                    domain = result.get("domain", "")
                    price = result.get("price")
                    currency = result.get("currency", "")
                    image_url = result.get("image")
                    is_fallback = result.get("is_fallback", False)

                    is_this_selected = (
                        current_selection is not None
                        and current_selection.get("url") == url
                    )

                    with st.container():
                        cols = st.columns([1, 3, 1])

                        with cols[0]:
                            if image_url and not is_fallback:
                                try:
                                    st.image(image_url, width=100)
                                except:
                                    st.write("🖼️")
                            else:
                                st.write("🔗")

                        with cols[1]:
                            st.markdown(f"**{title}**")
                            st.markdown(f"[{domain}]({url})")
                            if price:
                                st.write(f"💰 {currency} {price}")
                            if result.get("snippet"):
                                st.caption(result["snippet"][:150] + "...")

                        with cols[2]:
                            btn_label = "✅ Selected" if is_this_selected else "Select"
                            btn_type = "primary" if is_this_selected else "secondary"

                            if st.button(
                                btn_label, key=f"sel_{item_id}_{ridx}", type=btn_type
                            ):
                                # Download product image for later use
                                img_b64 = None
                                if image_url:
                                    img_b64 = download_image_to_base64(image_url)

                                selected_products[item_id] = {
                                    "url": url,
                                    "title": title,
                                    "domain": domain,
                                    "price": price,
                                    "currency": currency,
                                    "image": image_url,
                                    "image_b64": img_b64,
                                }
                                st.session_state.project_state["selected_products"] = (
                                    selected_products
                                )
                                st.rerun()

                        st.divider()

    # Navigation
    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        if st.button("⬅️ Back to Furniture Analysis"):
            st.session_state.project_state["phase"] = "furniture_analysis"
            st.rerun()

    with col2:
        selected_count = len(selected_products)
        st.info(f"📦 {selected_count} products selected")


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

    furniture_items = st.session_state.project_state.get("furniture_items", []) or []
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
        after_b64 = st.session_state.project_state.get(
            "selected_design_image_after_prefs"
        )
        if after_b64:
            st.image(
                base64.b64decode(after_b64),
                caption="After preferences",
                use_container_width=True,
            )
        else:
            st.info("No preference-applied image yet.")

    st.divider()

    # Per-item override UI
    for item in furniture_items:
        item_id = str(item.get("id"))
        existing = overrides.get(item_id, {})

        with st.expander(
            f"#{item_id} — {item.get('name')} ({item.get('color')}, {item.get('material')})",
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

            overrides[item_id] = {
                "enabled": bool(enabled),
                "mode": "product" if mode == "Use selected product" else "text",
                "text": (extra or "").strip(),
            }

    st.session_state.project_state["furniture_overrides"] = overrides

    st.divider()

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("⬅️ Back to Products"):
            st.session_state.project_state["phase"] = "products"
            st.rerun()

    with col2:
        if st.button("🧩 Apply Preferences (Replace Selected Items)", type="primary"):
            progress = st.progress(0)
            status = st.empty()

            def prog(p, msg):
                progress.progress(float(p))
                status.text(msg)

            result = apply_overrides_sequentially(
                base_design_b64=base_design_b64,
                furniture_items=furniture_items,
                overrides=overrides,
                selected_products=selected_products,
                progress_cb=prog,
            )

            if result.get("success") and result.get("image_base64"):
                st.session_state.project_state["selected_design_image_after_prefs"] = (
                    result["image_base64"]
                )
                st.success("✅ Preferences applied.")
                st.rerun()
            else:
                st.error(
                    f"Failed to apply preferences: {result.get('error', 'unknown error')}"
                )


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
        sel = selected_products.get(item_id) or {}
        ov = overrides.get(item_id, {}) or {}

        enabled = bool(ov.get("enabled", False))
        mode = ov.get("mode", "")

        row = {
            "id": item_id,
            "name": it.get("name", ""),
            "detected_color": it.get("color", ""),
            "detected_material": it.get("material", ""),
            "detected_style": it.get("style", ""),
            "position": it.get("position", ""),
            "size": it.get("size", ""),
            "area_percent": it.get("area_percent", None),
            "replacement_enabled": enabled,
            "replacement_mode": mode,
            "replacement_extra_requirements": ov.get("text", ""),
            "selected_product_title": sel.get("title", ""),
            "selected_product_url": sel.get("url", ""),
            "selected_product_domain": sel.get("domain", ""),
            "selected_product_price": sel.get("price", ""),
            "selected_product_currency": sel.get("currency", ""),
            "selected_product_image": sel.get("image", ""),
        }
        bom_rows.append(row)

    st.session_state.project_state["bom"] = bom_rows

    st.caption("This BOM lists detected items + your selected replacements (if any).")

    # Display table
    try:
        import pandas as pd

        df = pd.DataFrame(bom_rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
    except Exception:
        # Fallback if pandas not installed
        for r in bom_rows:
            with st.expander(f"#{r['id']} — {r['name']}"):
                st.json(r)

    st.divider()

    # Exports
    colA, colB, colC = st.columns([1, 1, 1])

    with colA:
        json_bytes = json.dumps({"items": bom_rows}, indent=2).encode("utf-8")
        st.download_button(
            "⬇️ Download BOM JSON",
            data=json_bytes,
            file_name="bom.json",
            mime="application/json",
        )

    with colB:
        try:
            import pandas as pd

            csv_bytes = pd.DataFrame(bom_rows).to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download BOM CSV",
                data=csv_bytes,
                file_name="bom.csv",
                mime="text/csv",
            )
        except Exception:
            st.info("Install pandas to export CSV: pip install pandas")

    with colC:
        if st.button("✅ Finish", type="primary"):
            st.session_state.project_state["phase"] = "complete"
            st.rerun()

    st.divider()
    if st.button("⬅️ Back to Preferences"):
        st.session_state.project_state["phase"] = "preferences"
        st.rerun()


def render_complete_phase():
    st.header("✅ Complete")

    final_b64 = st.session_state.project_state.get(
        "selected_design_image_after_prefs"
    ) or st.session_state.project_state.get("selected_design_image")

    if final_b64:
        st.image(
            base64.b64decode(final_b64),
            caption="Final result",
            use_container_width=True,
        )
        st.download_button(
            "⬇️ Download Final Image (JPEG)",
            data=base64.b64decode(final_b64),
            file_name="final.jpg",
            mime="image/jpeg",
        )
    else:
        st.info("No final image found.")

    st.divider()

    bom_rows = st.session_state.project_state.get("bom")
    if bom_rows:
        st.subheader("BOM Summary")
        st.write(f"Items: {len(bom_rows)}")
        replaced = sum(1 for r in bom_rows if r.get("replacement_enabled"))
        st.write(f"Replacements enabled: {replaced}")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔁 Start New Project"):
            st.session_state.project_state = DEFAULT_STATE.copy()
            st.rerun()
    with col2:
        if st.button("⬅️ Back to BOM"):
            st.session_state.project_state["phase"] = "bom"
            st.rerun()


def main():
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
