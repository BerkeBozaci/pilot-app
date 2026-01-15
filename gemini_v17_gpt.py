"""
OmniRenovation AI - Phase 1 Pilot v17
=====================================
- Google Gemini for image generation (ONLY)
- Open-source product search (retailer links you can customize)
- Adds: Per-furniture MASK + detailed identification WITHOUT LLM usage
  - Detection: OWL-ViT (Apache-2.0) via HuggingFace Transformers
  - Segmentation: Meta Segment Anything (SAM v1) (Apache-2.0)
  - Attributes:
      * Color: masked pixel clustering (deterministic)
      * Type/material/style: CLIP zero-shot (MIT)
      * Cushion/pillow counts: best-effort heuristics (OpenCV)
- Claude remains used for room analysis + concept text (as in your current code)
- No backup models for image generation (Gemini only)
"""

import os
import io
import re
import json
import base64
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import requests
import streamlit as st
from PIL import Image
from urllib.parse import quote_plus

# OpenCV + ML helpers
import cv2
from sklearn.cluster import KMeans

# HuggingFace
import torch
from transformers import pipeline

# Segment Anything
try:
    from segment_anything import sam_model_registry, SamPredictor

    SAM_AVAILABLE = True
except Exception:
    SAM_AVAILABLE = False


# ============== PAGE CONFIG ==============
st.set_page_config(page_title="OmniRenovation AI", page_icon="🏠", layout="wide")

# ============== SESSION STATE ==============
DEFAULT_STATE = {
    "phase": "upload",
    "images": [],
    "preferences": {},
    "valuation": None,
    "designs": None,
    "design_images": {},
    "selected_design": None,
    "selected_design_image": None,
    # old
    "furniture_items": [],
    "product_matches": {},
    "selected_products": {},
    "bom": None,
    # NEW: instance-based extraction
    # Each instance: {id, label, score, box_xyxy, mask_png_b64, attrs{}, search_query}
    "furniture_instances": [],
    "furniture_instances_source_hash": None,
}

if "project_state" not in st.session_state:
    st.session_state.project_state = DEFAULT_STATE.copy()


# =========================================================
# ===================== UTILITIES ==========================
# =========================================================


def b64_to_pil(b64_str: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")


def pil_to_b64_jpeg(pil_img: Image.Image, quality: int = 85) -> str:
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def pil_to_b64_png(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def mask_to_png_b64(mask_bool: np.ndarray) -> str:
    # mask_bool: HxW bool
    m = mask_bool.astype(np.uint8) * 255
    pil = Image.fromarray(m, mode="L")
    return pil_to_b64_png(pil)


def png_b64_to_mask(mask_png_b64: str) -> np.ndarray:
    pil = Image.open(io.BytesIO(base64.b64decode(mask_png_b64))).convert("L")
    arr = np.array(pil)
    return arr > 127


def sha1_short(text: str) -> str:
    import hashlib

    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


# =========================================================
# ================== API KEY MANAGEMENT ===================
# =========================================================


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
        elif response.status_code == 400:
            return False, "Invalid API key format"
        elif response.status_code == 403:
            return False, "API key is invalid or expired"
        else:
            return False, f"Error: {response.status_code}"
    except Exception as e:
        return False, f"Connection error: {str(e)}"


def validate_anthropic_key(api_key: str) -> tuple:
    if not api_key:
        return False, "API key is empty"
    if not api_key.startswith("sk-ant-"):
        return False, "Invalid format (should start with sk-ant-)"
    return True, "Format valid"


# =========================================================
# ================ GEMINI IMAGE GENERATION ================
# =========================================================


def generate_design_with_gemini(
    image_base64: str,
    style: str,
    variation: str,
    room_type: str = "room",
    max_retries: int = 3,
) -> dict:
    api_key = get_api_key("GEMINI_API_KEY")
    if not api_key:
        return {
            "error": "Gemini API key not configured. Please add your API key in the settings."
        }

    style_descriptions = {
        "Modern Minimalist": "modern minimalist style with clean lines, neutral colors (white, beige, grey), minimal furniture, lots of natural light, and sleek surfaces",
        "Scandinavian": "Scandinavian style with light oak wood floors, white walls, cozy wool textiles, green plants, functional furniture, and warm ambient lighting",
        "Industrial": "industrial loft style with exposed brick walls, black metal fixtures, Edison bulb lighting, concrete elements, and dark leather furniture",
        "Mid-Century Modern": "mid-century modern style with walnut wood furniture, organic curved shapes, iconic 1960s design pieces, and mustard/teal accent colors",
        "Contemporary": "contemporary style with mixed materials (wood, metal, glass), neutral base with bold accent colors, and clean comfortable furniture",
        "Bohemian": "bohemian style with eclectic colorful patterns, layered textiles and rugs, macrame wall hangings, lots of plants, and artistic collected feel",
        "Coastal": "coastal beach house style with blue and white color palette, natural rattan and wicker furniture, light airy feeling, and nautical accents",
        "Farmhouse": "modern farmhouse style with shiplap white walls, rustic reclaimed wood beams, neutral earth tones, and vintage furniture pieces",
    }

    variation_descriptions = {
        "Light & Airy": "bright and airy atmosphere with abundant natural light, white and cream color palette, and open spacious feeling",
        "Warm & Cozy": "warm and cozy atmosphere with warm ambient lighting, rich earth tones, and layered soft textures",
        "Bold & Dramatic": "bold and dramatic atmosphere with rich deep colors, high contrast, and statement furniture pieces",
    }

    style_desc = style_descriptions.get(style, "modern interior design")
    variation_desc = variation_descriptions.get(variation, "")

    prompt = f"""Transform this {room_type} into a beautifully designed interior space.

STYLE: {style_desc}
MOOD: {variation_desc}

CRITICAL INSTRUCTIONS:
1. KEEP THE EXACT SAME ROOM STRUCTURE - same walls, windows, doors, ceiling, floor plan
2. KEEP THE SAME CAMERA ANGLE AND PERSPECTIVE
3. Only change: furniture, decor, colors, textures, lighting fixtures, and styling
4. Add appropriate furniture for a {room_type}: sofas, tables, chairs, lamps, rugs, artwork, plants
5. Make it look photorealistic and professionally designed
6. Ensure the design is cohesive and magazine-quality

Generate the redesigned room image."""

    model_variants = [
        "gemini-2.5-flash-image",
        "gemini-2.5-flash-image-preview",
        "gemini-2.0-flash-exp",
    ]

    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {"inlineData": {"mimeType": "image/jpeg", "data": image_base64}},
                ]
            }
        ],
        "generationConfig": {
            "responseModalities": ["IMAGE", "TEXT"],
            "temperature": 0.8,
        },
    }

    last_error = None
    for model_name in model_variants:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        for attempt in range(max_retries):
            try:
                st.info(
                    f"🎨 Generating with {model_name} (attempt {attempt + 1}/{max_retries})..."
                )
                response = requests.post(
                    url, headers=headers, json=payload, timeout=120
                )

                if response.status_code == 200:
                    data = response.json()
                    if "candidates" in data and len(data["candidates"]) > 0:
                        parts = (
                            data["candidates"][0].get("content", {}).get("parts", [])
                        )
                        for part in parts:
                            if "inlineData" in part:
                                image_data = part["inlineData"].get("data", "")
                                if image_data:
                                    return {
                                        "success": True,
                                        "image_base64": image_data,
                                        "method": model_name,
                                    }
                        text_parts = [p.get("text", "") for p in parts if "text" in p]
                        if text_parts:
                            last_error = f"Model returned text instead of image: {text_parts[0][:100]}"
                            break
                    last_error = "No image in response"
                    break

                elif response.status_code == 429:
                    wait_time = (2**attempt) * 5
                    st.warning(f"⏳ Rate limited. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue

                elif response.status_code == 404:
                    last_error = f"Model {model_name} not found"
                    break

                elif response.status_code == 400:
                    error_data = response.json()
                    error_msg = error_data.get("error", {}).get(
                        "message", "Bad request"
                    )
                    last_error = f"API Error: {error_msg}"
                    break

                elif response.status_code == 403:
                    last_error = "API key invalid or image generation not enabled"
                    return {"error": last_error}

                else:
                    last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                    break

            except requests.exceptions.Timeout:
                last_error = "Request timed out"
                continue
            except Exception as e:
                last_error = str(e)
                continue

    return {"error": f"All generation attempts failed. Last error: {last_error}"}


# =========================================================
# ================== CLAUDE FOR ANALYSIS ==================
# =========================================================


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
            "text": f"""Analyze this room for renovation planning. Style preference: {preferences.get('style', 'Modern')}.

Return ONLY valid JSON (no markdown, no extra text):
{{
    "property_assessment": {{
        "room_type": "Living Room/Bedroom/Kitchen/etc",
        "current_condition": "poor/fair/good/excellent",
        "square_footage_estimate": "estimated sq ft",
        "notable_features": ["window", "fireplace", "etc"]
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
        else:
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

Return ONLY valid JSON (no markdown):
{{
    "design_options": [
        {{
            "option_number": 1,
            "name": "{style} - Light & Airy",
            "variation": "Light & Airy",
            "concept": "Bright, open design with natural light",
            "color_palette": {{"primary": "#F5F5F5", "secondary": "#E8E8E8", "accent": "#4A90A4"}},
            "key_furniture": ["light grey linen sofa", "white oak coffee table", "brass floor lamp", "cream wool rug"],
            "estimated_cost": 12000
        }},
        {{
            "option_number": 2,
            "name": "{style} - Warm & Cozy",
            "variation": "Warm & Cozy",
            "concept": "Warm, inviting atmosphere",
            "color_palette": {{"primary": "#F5E6D3", "secondary": "#D4A574", "accent": "#8B4513"}},
            "key_furniture": ["cognac leather sofa", "walnut coffee table", "ceramic table lamps", "jute area rug"],
            "estimated_cost": 15000
        }},
        {{
            "option_number": 3,
            "name": "{style} - Bold & Dramatic",
            "variation": "Bold & Dramatic",
            "concept": "Striking design with sophisticated contrast",
            "color_palette": {{"primary": "#2C3E50", "secondary": "#34495E", "accent": "#C0392B"}},
            "key_furniture": ["navy velvet sofa", "black marble coffee table", "gold pendant light", "persian style rug"],
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
        else:
            return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}


# =========================================================
# ============ OPEN-SOURCE INSTANCE EXTRACTION ============
# =========================================================

FURNITURE_QUERIES = [
    "sofa",
    "couch",
    "sectional sofa",
    "armchair",
    "chair",
    "coffee table",
    "side table",
    "dining table",
    "desk",
    "rug",
    "carpet",
    "floor lamp",
    "table lamp",
    "ceiling lamp",
    "pendant light",
    "tv stand",
    "console table",
    "bookshelf",
    "shelving",
    "curtains",
    "artwork",
    "plant",
    "mirror",
    "bed",
    "nightstand",
]

TYPE_LABELS = [
    "sofa",
    "armchair",
    "chair",
    "coffee table",
    "side table",
    "dining table",
    "desk",
    "rug",
    "floor lamp",
    "table lamp",
    "pendant light",
    "tv stand",
    "console table",
    "bookshelf",
    "curtains",
    "artwork",
    "plant",
    "mirror",
    "bed",
    "nightstand",
]

MATERIAL_LABELS = [
    "linen fabric",
    "cotton fabric",
    "wool fabric",
    "velvet fabric",
    "leather",
    "faux leather",
    "wood",
    "oak wood",
    "walnut wood",
    "metal",
    "brass metal",
    "steel metal",
    "glass",
    "marble",
    "stone",
    "ceramic",
]

STYLE_LABELS = [
    "modern",
    "scandinavian",
    "industrial",
    "mid-century modern",
    "contemporary",
    "bohemian",
    "coastal",
    "farmhouse",
    "traditional",
    "minimalist",
]


@st.cache_resource(show_spinner=False)
def load_owlvit_detector():
    # Zero-shot object detection
    return pipeline("zero-shot-object-detection", model="google/owlvit-base-patch32")


@st.cache_resource(show_spinner=False)
def load_clip_zeroshot():
    # Zero-shot image classification
    # Using a CLIP model available in HF (MIT license for OpenAI CLIP; model packaging varies).
    # This pipeline will download weights at first run.
    return pipeline(
        "zero-shot-image-classification", model="openai/clip-vit-base-patch32"
    )


@st.cache_resource(show_spinner=False)
def load_sam_predictor():
    """
    Loads SAM predictor. Requires:
      - pip install git+https://github.com/facebookresearch/segment-anything.git
      - set env SAM_CHECKPOINT to a .pth
    """
    if not SAM_AVAILABLE:
        return None, "segment-anything not installed. Install it to enable masks."

    ckpt = os.environ.get("SAM_CHECKPOINT", "").strip()
    if not ckpt or not os.path.exists(ckpt):
        return None, "SAM_CHECKPOINT not set or file missing. Masks disabled."

    # Choose model type based on checkpoint name
    # common: sam_vit_h_..., sam_vit_l_..., sam_vit_b_...
    if "vit_h" in os.path.basename(ckpt):
        model_type = "vit_h"
    elif "vit_l" in os.path.basename(ckpt):
        model_type = "vit_l"
    else:
        model_type = "vit_b"

    sam = sam_model_registry[model_type](checkpoint=ckpt)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam.to(device=device)

    predictor = SamPredictor(sam)
    return predictor, f"SAM loaded ({model_type}) on {device}"


def detect_instances_owlvit(
    pil_img: Image.Image, queries: List[str], score_thresh=0.15
) -> List[Dict[str, Any]]:
    det = load_owlvit_detector()
    out = det(pil_img, candidate_labels=queries)
    results = []
    for d in out:
        if d.get("score", 0) >= score_thresh:
            b = d["box"]
            results.append(
                {
                    "label": d["label"],
                    "score": float(d["score"]),
                    "box_xyxy": (
                        int(b["xmin"]),
                        int(b["ymin"]),
                        int(b["xmax"]),
                        int(b["ymax"]),
                    ),
                }
            )
    # Basic NMS-ish filtering: keep top per overlap
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    filtered = []
    for r in results:
        keep = True
        x1, y1, x2, y2 = r["box_xyxy"]
        area = max(0, (x2 - x1)) * max(0, (y2 - y1))
        if area < 400:  # tiny
            continue
        for f in filtered:
            fx1, fy1, fx2, fy2 = f["box_xyxy"]
            ix1, iy1 = max(x1, fx1), max(y1, fy1)
            ix2, iy2 = min(x2, fx2), min(y2, fy2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            farea = max(0, (fx2 - fx1)) * max(0, (fy2 - fy1))
            union = area + farea - inter + 1e-6
            iou = inter / union
            if iou > 0.6:
                keep = False
                break
        if keep:
            filtered.append(r)
    return filtered[:20]  # cap


def segment_box_sam(
    pil_img: Image.Image, box_xyxy: Tuple[int, int, int, int]
) -> Optional[np.ndarray]:
    predictor, status = load_sam_predictor()
    if predictor is None:
        return None

    img = np.array(pil_img.convert("RGB"))
    predictor.set_image(img)

    x1, y1, x2, y2 = box_xyxy
    box = np.array([x1, y1, x2, y2])

    masks, scores, _ = predictor.predict(box=box, multimask_output=True)
    if masks is None or len(masks) == 0:
        return None
    best_idx = int(np.argmax(scores))
    return masks[best_idx].astype(bool)


def masked_crop_rgba(
    pil_img: Image.Image, mask_bool: np.ndarray, pad: int = 10
) -> Optional[Image.Image]:
    if mask_bool is None:
        return None
    img = np.array(pil_img.convert("RGB"))
    h, w = mask_bool.shape
    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        return None
    x1 = max(int(xs.min()) - pad, 0)
    y1 = max(int(ys.min()) - pad, 0)
    x2 = min(int(xs.max()) + pad, w - 1)
    y2 = min(int(ys.max()) + pad, h - 1)

    crop = img[y1 : y2 + 1, x1 : x2 + 1]
    crop_mask = mask_bool[y1 : y2 + 1, x1 : x2 + 1]

    alpha = crop_mask.astype(np.uint8) * 255
    rgba = np.dstack([crop, alpha])
    return Image.fromarray(rgba, mode="RGBA")


def dominant_color_names_from_mask(
    pil_img: Image.Image, mask_bool: np.ndarray
) -> Dict[str, Any]:
    """
    Deterministic color extraction:
    - take masked pixels
    - kmeans in LAB for robustness
    - map dominant cluster to simple color names
    """
    img = np.array(pil_img.convert("RGB"))
    mask = mask_bool.astype(bool)
    pixels = img[mask]
    if pixels.shape[0] < 200:
        return {"primary_color": None, "secondary_colors": [], "color_confidence": 0.2}

    # sample for speed
    if pixels.shape[0] > 20000:
        idx = np.random.choice(pixels.shape[0], 20000, replace=False)
        pixels = pixels[idx]

    lab = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
    k = 3 if lab.shape[0] >= 800 else 2
    km = KMeans(n_clusters=k, n_init=5, random_state=0)
    labels = km.fit_predict(lab)
    centers = km.cluster_centers_

    counts = np.bincount(labels)
    order = np.argsort(counts)[::-1]

    def lab_to_rgb_tuple(lab_vec):
        lab_px = np.array(lab_vec, dtype=np.float32).reshape(1, 1, 3)
        rgb = cv2.cvtColor(lab_px, cv2.COLOR_LAB2RGB).reshape(
            3,
        )
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return int(rgb[0]), int(rgb[1]), int(rgb[2])

    def rgb_to_name(r, g, b):
        # very simple naming heuristic (good enough for shopping queries)
        hsv = cv2.cvtColor(np.uint8([[[r, g, b]]]), cv2.COLOR_RGB2HSV)[0, 0]
        h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

        if v < 40:
            return "black"
        if v > 220 and s < 30:
            return "white"
        if s < 35 and 40 <= v <= 220:
            return "gray"
        if h < 10 or h >= 170:
            return "red"
        if 10 <= h < 25:
            return "orange"
        if 25 <= h < 35:
            return "yellow"
        if 35 <= h < 85:
            return "green"
        if 85 <= h < 130:
            return "blue"
        if 130 <= h < 170:
            return "purple"
        return "neutral"

    colors = [rgb_to_name(*lab_to_rgb_tuple(centers[i])) for i in order[:k]]
    primary = colors[0] if colors else None
    secondary = []
    for c in colors[1:]:
        if c and c != primary and c not in secondary:
            secondary.append(c)

    conf = float(counts[order[0]] / counts.sum())
    return {
        "primary_color": primary,
        "secondary_colors": secondary,
        "color_confidence": round(conf, 3),
    }


def clip_best_label(pil_img: Image.Image, labels: List[str]) -> Dict[str, Any]:
    clf = load_clip_zeroshot()
    # pipeline accepts PIL directly
    out = clf(pil_img, candidate_labels=labels)
    if not out:
        return {"label": None, "score": 0.0}
    best = out[0]
    return {"label": best["label"], "score": float(best["score"])}


def estimate_counts_heuristic(rgba_obj: Image.Image) -> Dict[str, Any]:
    """
    Best-effort seat cushion / back pillow counting.
    This is NOT perfect. It gives a reasonable estimate on clean renders.
    """
    try:
        arr = np.array(rgba_obj.convert("RGBA"))
        rgb = arr[:, :, :3]
        a = arr[:, :, 3]
        mask = a > 10

        # focus on object region
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # edges inside mask
        edges = cv2.Canny(gray, 60, 140)
        edges[~mask] = 0

        # dilate to connect shapes
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)

        # connected components
        num, cc = cv2.connectedComponents(edges)
        # count "medium" blobs as cushions-ish
        h, w = edges.shape
        areas = []
        for i in range(1, num):
            area = int((cc == i).sum())
            areas.append(area)

        # heuristics: ignore tiny
        areas = [a for a in areas if a > (h * w * 0.002)]
        # rough splitting: cushions/pillows tend to form multiple blobs
        est = min(max(len(areas), 0), 8)

        # We can’t reliably split seat vs back with simple CV.
        # Return a conservative guess:
        seat_cushions = max(1, min(3, est // 2)) if est else None
        back_pillows = max(1, min(5, est - (seat_cushions or 0))) if est else None

        return {
            "seat_cushions_count": seat_cushions,
            "back_pillows_count": back_pillows,
            "count_confidence": 0.35 if est else 0.1,
        }
    except Exception:
        return {
            "seat_cushions_count": None,
            "back_pillows_count": None,
            "count_confidence": 0.1,
        }


def build_search_query(attrs: Dict[str, Any]) -> str:
    parts = []
    if attrs.get("primary_color"):
        parts.append(attrs["primary_color"])
    if attrs.get("material"):
        parts.append(attrs["material"])
    if attrs.get("type"):
        parts.append(attrs["type"])
    # add counts if present
    if attrs.get("seat_count"):
        parts.append(f"{attrs['seat_count']}-seater")
    if attrs.get("seat_cushions_count"):
        parts.append(f"{attrs['seat_cushions_count']} seat cushions")
    if attrs.get("back_pillows_count"):
        parts.append(f"{attrs['back_pillows_count']} back pillows")
    if attrs.get("style"):
        parts.append(attrs["style"])
    return " ".join(parts).strip() or attrs.get("type", "furniture")


def extract_instances_and_attributes(pil_img: Image.Image) -> List[Dict[str, Any]]:
    """
    Full pipeline:
      OWL-ViT boxes -> SAM masks -> per-instance RGBA crop -> attrs (color + CLIP labels + heuristic counts)
    """
    predictor, sam_status = load_sam_predictor()

    dets = detect_instances_owlvit(pil_img, FURNITURE_QUERIES, score_thresh=0.18)

    instances = []
    for idx, d in enumerate(dets, start=1):
        box = d["box_xyxy"]
        label = d["label"]

        # mask (optional)
        mask = segment_box_sam(pil_img, box)
        mask_b64 = mask_to_png_b64(mask) if mask is not None else None

        # object crop (prefer masked crop if possible)
        if mask is not None:
            rgba = masked_crop_rgba(pil_img, mask)
        else:
            # fallback: box crop
            x1, y1, x2, y2 = box
            rgba = pil_img.crop((x1, y1, x2, y2)).convert("RGBA")

        # attributes
        attrs = {}

        # Color (only reliable if we have a mask)
        if mask is not None:
            color_info = dominant_color_names_from_mask(pil_img, mask)
        else:
            color_info = {
                "primary_color": None,
                "secondary_colors": [],
                "color_confidence": 0.1,
            }

        # CLIP type/material/style on object crop
        # note: use RGB crop for CLIP
        rgb_crop = rgba.convert("RGB")

        type_pred = clip_best_label(rgb_crop, TYPE_LABELS)
        mat_pred = clip_best_label(rgb_crop, MATERIAL_LABELS)
        style_pred = clip_best_label(rgb_crop, STYLE_LABELS)

        attrs["type"] = type_pred["label"]
        attrs["type_confidence"] = round(type_pred["score"], 3)

        attrs["material"] = mat_pred["label"]
        attrs["material_confidence"] = round(mat_pred["score"], 3)

        attrs["style"] = style_pred["label"]
        attrs["style_confidence"] = round(style_pred["score"], 3)

        attrs.update(color_info)

        # cushion/pillow counts (best effort)
        counts = estimate_counts_heuristic(rgba)
        attrs.update(counts)

        # seat_count heuristic: if it's a sofa/couch/sectional, guess from width/height
        # (this is rough; you can replace later)
        seat_count = None
        if attrs.get("type") in ("sofa", "couch", "sectional sofa"):
            w = rgba.size[0]
            if w < 220:
                seat_count = 2
            elif w < 420:
                seat_count = 3
            else:
                seat_count = 4
        attrs["seat_count"] = seat_count

        search_query = build_search_query(attrs)

        instances.append(
            {
                "id": idx,
                "label": label,
                "score": round(float(d["score"]), 3),
                "box_xyxy": box,
                "mask_png_b64": mask_b64,
                "attrs": attrs,
                "search_query": search_query,
                "object_crop_png_b64": pil_to_b64_png(rgba),
            }
        )

    return instances


# =========================================================
# ============== OPEN SOURCE PRODUCT SEARCH ================
# =========================================================


class OpenSourceProductSearch:
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
            "name": "West Elm",
            "search_url": "https://www.westelm.com/search/?query={query}",
            "icon": "✨",
        },
        {
            "name": "Target",
            "search_url": "https://www.target.com/s?searchTerm={query}",
            "icon": "🎯",
        },
        {
            "name": "CB2",
            "search_url": "https://www.cb2.com/search/?query={query}",
            "icon": "🛋️",
        },
        {
            "name": "Overstock",
            "search_url": "https://www.overstock.com/Home-Garden/?keywords={query}",
            "icon": "📦",
        },
        {
            "name": "Home Depot",
            "search_url": "https://www.homedepot.com/s/{query}",
            "icon": "🔨",
        },
    ]

    @classmethod
    def search(cls, query: str) -> list:
        encoded_query = quote_plus(query)
        results = []
        for retailer in cls.RETAILERS:
            url = retailer["search_url"].replace("{query}", encoded_query)
            results.append(
                {
                    "product_name": f"{retailer['icon']} Search on {retailer['name']}",
                    "url": url,
                    "retailer": retailer["name"],
                    "price_str": "Browse",
                    "is_direct_link": False,
                    "source": "retailer_search",
                }
            )
        return results

    @classmethod
    def get_retailer_list(cls) -> list:
        return [r["name"] for r in cls.RETAILERS]


def find_products_for_item(item: dict) -> list:
    query = item.get("search_query", item.get("name", "furniture"))
    return OpenSourceProductSearch.search(query)


# =========================================================
# =================== IMAGE PROCESSING =====================
# =========================================================


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

        base64_data = pil_to_b64_jpeg(img, quality=85)
        return {"success": True, "data": base64_data, "name": uploaded_file.name}
    except Exception as e:
        return {"success": False, "error": str(e)}


# =========================================================
# ===================== UI COMPONENTS ======================
# =========================================================


def render_header():
    st.title("🏠 OmniRenovation AI")
    st.caption("AI-Powered Interior Design with Google Gemini")

    phases = ["Upload", "Analysis", "Design", "Products", "BOM", "Complete"]
    phase_map = {
        "upload": 0,
        "valuation": 1,
        "design": 2,
        "products": 3,
        "bom": 4,
        "complete": 5,
    }
    current = phase_map.get(st.session_state.project_state["phase"], 0)

    cols = st.columns(6)
    for i, phase in enumerate(phases):
        with cols[i]:
            if i < current:
                st.success(f"✓ {phase}")
            elif i == current:
                st.info(f"→ {phase}")
            else:
                st.text(f"○ {phase}")


def render_upload_phase():
    st.header("📤 Upload Your Room")

    with st.expander("🔑 API Configuration", expanded=True):
        st.markdown("### Required API Keys")
        st.info(
            """
**You need two API keys:**
1. **Google Gemini** - For AI image generation (get free at aistudio.google.com)
2. **Anthropic Claude** - For room analysis & design concept JSON (console.anthropic.com)

**Optional local CV stack for masks & attributes:**
- OWL-ViT + CLIP auto-download from HuggingFace
- SAM requires you to set `SAM_CHECKPOINT` (see top of file)
            """
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Google Gemini API Key** *")
            gemini_key = st.text_input(
                "Gemini Key",
                type="password",
                value=st.session_state.get("gemini_api_key", ""),
                label_visibility="collapsed",
                placeholder="Enter your Gemini API key",
            )
            if gemini_key:
                st.session_state.gemini_api_key = gemini_key

        with col2:
            st.markdown("**Anthropic Claude API Key** *")
            anthropic_key = st.text_input(
                "Anthropic Key",
                type="password",
                value=st.session_state.get("anthropic_api_key", ""),
                label_visibility="collapsed",
                placeholder="Enter your Claude API key (sk-ant-...)",
            )
            if anthropic_key:
                st.session_state.anthropic_api_key = anthropic_key

        st.markdown("### 📊 API Status")
        col1, col2 = st.columns(2)
        with col1:
            if st.session_state.get("gemini_api_key"):
                is_valid, msg = validate_gemini_key(
                    st.session_state.get("gemini_api_key", "")
                )
                (
                    st.success("✅ Gemini API Key: Valid")
                    if is_valid
                    else st.error(f"❌ Gemini API Key: {msg}")
                )
            else:
                st.warning("⚠️ Gemini API Key: Not entered")

        with col2:
            if st.session_state.get("anthropic_api_key"):
                is_valid, msg = validate_anthropic_key(
                    st.session_state.get("anthropic_api_key", "")
                )
                (
                    st.success("✅ Claude API Key: Format valid")
                    if is_valid
                    else st.error(f"❌ Claude API Key: {msg}")
                )
            else:
                st.warning("⚠️ Claude API Key: Not entered")

        st.markdown("### 🧩 Local Mask + Attribute Extractor Status")
        predictor, sam_status = load_sam_predictor()
        if predictor is None:
            st.warning(f"⚠️ SAM (masks): {sam_status}")
        else:
            st.success(f"✅ SAM (masks): {sam_status}")
        st.caption(
            "OWL-ViT + CLIP will download on first use (can take a few minutes)."
        )

        gemini_ready = bool(st.session_state.get("gemini_api_key"))
        claude_ready = bool(st.session_state.get("anthropic_api_key"))
        if gemini_ready and claude_ready:
            st.success("✅ All required API keys configured! You're ready to start.")
        else:
            missing = []
            if not gemini_ready:
                missing.append("Gemini")
            if not claude_ready:
                missing.append("Claude")
            st.error(f"❌ Missing API keys: {', '.join(missing)}")

    st.divider()

    st.markdown("### 📷 Upload Room Photo")
    uploaded_files = st.file_uploader(
        "Upload one or more photos of your room",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        help="Upload clear photos of the room you want to redesign",
    )

    if uploaded_files:
        cols = st.columns(min(4, len(uploaded_files)))
        for i, f in enumerate(uploaded_files):
            with cols[i % 4]:
                f.seek(0)
                st.image(Image.open(f), use_column_width=True)

    st.divider()
    st.markdown("### 🎨 Design Preferences")

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
        st.markdown("**Product Search Retailers:**")
        st.caption(
            f"Configured: {', '.join(OpenSourceProductSearch.get_retailer_list())}"
        )

    st.divider()

    can_start = (
        bool(st.session_state.get("gemini_api_key"))
        and bool(st.session_state.get("anthropic_api_key"))
        and bool(uploaded_files)
    )

    if not can_start:
        reasons = []
        if not st.session_state.get("gemini_api_key"):
            reasons.append("Enter Gemini API key")
        if not st.session_state.get("anthropic_api_key"):
            reasons.append("Enter Claude API key")
        if not uploaded_files:
            reasons.append("Upload room photo")
        st.warning(f"To start: {', '.join(reasons)}")

    if st.button("🚀 Analyze Room", type="primary", disabled=not can_start):
        processed_images = [process_uploaded_image(f) for f in uploaded_files]
        successful_images = [img for img in processed_images if img.get("success")]

        if successful_images:
            st.session_state.project_state.update(
                {
                    "images": successful_images,
                    "preferences": {
                        "budget": budget,
                        "style": style,
                        "room_type": room_type,
                    },
                    "phase": "valuation",
                    # reset instance extraction cache
                    "furniture_instances": [],
                    "furniture_instances_source_hash": None,
                    "furniture_items": [],
                    "product_matches": {},
                    "selected_products": {},
                    "bom": None,
                }
            )
            st.rerun()
        else:
            st.error("Failed to process uploaded images. Please try again.")


def render_valuation_phase():
    st.header("📊 Room Analysis")

    if not st.session_state.project_state["valuation"]:
        with st.spinner("🔍 Analyzing your room with Claude..."):
            result = analyze_room_with_claude(
                st.session_state.project_state["images"],
                st.session_state.project_state["preferences"],
            )
            st.session_state.project_state["valuation"] = result
            st.rerun()

    valuation = st.session_state.project_state["valuation"]
    if "error" in valuation:
        st.error(f"Analysis failed: {valuation['error']}")
        if st.button("🔄 Retry Analysis"):
            st.session_state.project_state["valuation"] = None
            st.rerun()
        return

    assessment = valuation.get("property_assessment", {})
    costs = valuation.get("cost_estimate", {})

    col1, col2, col3 = st.columns(3)
    col1.metric("Room Type", assessment.get("room_type", "N/A"))
    col2.metric("Condition", str(assessment.get("current_condition", "N/A")).title())
    col3.metric("Est. Renovation Cost", f"${costs.get('mid', 0):,}")

    if st.session_state.project_state["images"]:
        st.image(
            base64.b64decode(st.session_state.project_state["images"][0]["data"]),
            caption="Your Room",
            use_column_width=True,
        )

    features = assessment.get("notable_features", [])
    if features:
        st.info(f"**Notable Features:** {', '.join(features)}")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🎨 Generate Design Options", type="primary"):
            st.session_state.project_state["phase"] = "design"
            st.rerun()

    with col2:
        if st.button("← Start Over"):
            st.session_state.project_state = DEFAULT_STATE.copy()
            st.rerun()


def render_design_phase():
    st.header("🎨 Design Options")

    if not st.session_state.project_state["designs"]:
        with st.spinner("Creating design concepts..."):
            result = create_design_concepts(
                st.session_state.project_state["valuation"],
                st.session_state.project_state["preferences"],
            )
            st.session_state.project_state["designs"] = result
            st.rerun()

    designs = st.session_state.project_state["designs"]
    if "error" in designs:
        st.error(f"Failed to create designs: {designs['error']}")
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
                        use_column_width=True,
                    )
                    with st.expander("📊 Compare Before/After"):
                        c1, c2 = st.columns(2)
                        if original_image:
                            c1.image(base64.b64decode(original_image), caption="BEFORE")
                        c2.image(
                            base64.b64decode(design_images[num]["image_base64"]),
                            caption="AFTER",
                        )
                else:
                    st.error(
                        f"Generation failed: {design_images[num].get('error', 'Unknown error')}"
                    )
                    if st.button("🔄 Retry Generation", key=f"retry_{num}"):
                        del design_images[num]
                        st.session_state.project_state["design_images"] = design_images
                        st.rerun()
            else:
                palette = option.get("color_palette", {})
                st.markdown("**Color Palette:**")
                palette_cols = st.columns(3)
                for i, (name, color) in enumerate(list(palette.items())[:3]):
                    with palette_cols[i]:
                        st.color_picker(
                            name.title(), color, disabled=True, key=f"color_{num}_{i}"
                        )

                if original_image and st.button(
                    "🖼️ Generate This Design with Gemini", key=f"gen_{num}"
                ):
                    with st.spinner("Generating design with Gemini..."):
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
            st.metric("Estimated Cost", f"${option.get('estimated_cost', 0):,}")
            st.markdown("**Key Furniture:**")
            for item in option.get("key_furniture", [])[:5]:
                st.write(f"• {item}")

            if st.button("✅ Select This Design", key=f"select_{num}", type="primary"):
                st.session_state.project_state["selected_design"] = option
                if num in design_images and design_images[num].get("success"):
                    st.session_state.project_state["selected_design_image"] = (
                        design_images[num]["image_base64"]
                    )
                else:
                    st.session_state.project_state["selected_design_image"] = (
                        original_image
                    )

                # reset instance extraction when design changes
                st.session_state.project_state["furniture_instances"] = []
                st.session_state.project_state["furniture_instances_source_hash"] = None
                st.session_state.project_state["furniture_items"] = []
                st.session_state.project_state["product_matches"] = {}
                st.session_state.project_state["selected_products"] = {}
                st.success("✓ Design selected!")

        st.divider()

    if st.session_state.project_state.get("selected_design"):
        st.success(
            f"**Selected:** {st.session_state.project_state['selected_design'].get('name')}"
        )
        if st.button("🛋️ Find Products", type="primary"):
            st.session_state.project_state["phase"] = "products"
            st.rerun()

    if st.button("← Back to Analysis"):
        st.session_state.project_state["phase"] = "valuation"
        st.rerun()


def render_products_phase():
    st.header("🛋️ Product Search + Furniture Details (Mask + Attributes)")

    st.info(
        f"**Searching across:** {', '.join(OpenSourceProductSearch.get_retailer_list())}"
    )

    design_image_b64 = st.session_state.project_state.get("selected_design_image")
    if not design_image_b64:
        st.error("No selected design image found. Go back and select a design.")
        return

    with st.expander("📷 Selected Design Image", expanded=False):
        st.image(base64.b64decode(design_image_b64), use_column_width=True)

    # ---- NEW: Extract instances + attributes (cached by image hash) ----
    img_hash = sha1_short(design_image_b64[:5000])  # cheap-ish stable key
    instances = st.session_state.project_state.get("furniture_instances", [])
    prev_hash = st.session_state.project_state.get("furniture_instances_source_hash")

    colA, colB = st.columns([1, 1])
    with colA:
        extract_btn = st.button("🧩 Extract Masks + Attributes", type="primary")
    with colB:
        st.caption("First run may download models (OWL-ViT, CLIP).")

    if extract_btn or (not instances) or (prev_hash != img_hash):
        with st.spinner("Extracting furniture masks + attributes (open-source)..."):
            pil_img = b64_to_pil(design_image_b64)
            try:
                instances = extract_instances_and_attributes(pil_img)
                st.session_state.project_state["furniture_instances"] = instances
                st.session_state.project_state["furniture_instances_source_hash"] = (
                    img_hash
                )

                # Convert instances to your legacy furniture_items format (so rest of app works)
                furniture_items = []
                for inst in instances:
                    furniture_items.append(
                        {
                            "id": inst["id"],
                            "name": inst["attrs"].get("type") or inst["label"],
                            "search_query": inst["search_query"],
                            "price_low": 200,
                            "price_high": 1200,
                            "attrs": inst["attrs"],
                            "mask_png_b64": inst["mask_png_b64"],
                            "box_xyxy": inst["box_xyxy"],
                            "object_crop_png_b64": inst["object_crop_png_b64"],
                        }
                    )
                st.session_state.project_state["furniture_items"] = furniture_items

                # reset product matches because queries changed
                st.session_state.project_state["product_matches"] = {}
                st.session_state.project_state["selected_products"] = {}

                st.success(f"✅ Extracted {len(instances)} furniture instances.")
                st.rerun()
            except Exception as e:
                st.error(f"Extraction failed: {e}")

    furniture_items = st.session_state.project_state.get("furniture_items", [])
    product_matches = st.session_state.project_state.get("product_matches", {})

    if not furniture_items:
        st.warning(
            "No furniture items extracted yet. Click 'Extract Masks + Attributes'."
        )
        return

    st.write(f"**Found {len(furniture_items)} items**")
    st.divider()

    # ---- Show each item with mask + attributes + retailer links ----
    for item in furniture_items:
        item_id = str(item.get("id", 0))
        name = item.get("name", "Item")
        attrs = item.get("attrs", {})

        with st.expander(
            f"🔍 **{name}** — query: `{item.get('search_query','')}`", expanded=True
        ):
            c1, c2, c3 = st.columns([1.2, 1, 1])

            with c1:
                # show cutout
                crop_b64 = item.get("object_crop_png_b64")
                if crop_b64:
                    st.image(
                        base64.b64decode(crop_b64),
                        caption="Object cutout (RGBA)",
                        use_column_width=True,
                    )

                # show mask (if available)
                mask_b64 = item.get("mask_png_b64")
                if mask_b64:
                    st.image(
                        base64.b64decode(mask_b64),
                        caption="Mask (white = object)",
                        use_column_width=True,
                    )
                else:
                    st.caption("Mask not available (SAM not configured).")

            with c2:
                st.markdown("### 🧾 Attributes")
                st.write(
                    f"**Type:** {attrs.get('type')} (conf {attrs.get('type_confidence')})"
                )
                st.write(
                    f"**Material:** {attrs.get('material')} (conf {attrs.get('material_confidence')})"
                )
                st.write(
                    f"**Style:** {attrs.get('style')} (conf {attrs.get('style_confidence')})"
                )
                st.write(
                    f"**Primary color:** {attrs.get('primary_color')} (conf {attrs.get('color_confidence')})"
                )
                if attrs.get("secondary_colors"):
                    st.write(
                        f"**Secondary colors:** {', '.join(attrs.get('secondary_colors'))}"
                    )
                st.write(f"**Seat count (rough):** {attrs.get('seat_count')}")
                st.write(
                    f"**Seat cushions (rough):** {attrs.get('seat_cushions_count')}"
                )
                st.write(f"**Back pillows (rough):** {attrs.get('back_pillows_count')}")
                st.caption(
                    "Counts are best-effort heuristics; for perfect counts, replace with a small detector later."
                )

            with c3:
                st.markdown("### 🛒 Retailer Search Links")
                if item_id not in product_matches:
                    products = find_products_for_item(item)
                    product_matches[item_id] = products
                    st.session_state.project_state["product_matches"] = product_matches

                products = product_matches.get(item_id, [])
                for i, product in enumerate(products[:8]):
                    st.markdown(f"**{product.get('product_name','')}**")
                    url = product.get("url", "")
                    if url:
                        st.markdown(f"[🔗 Search →]({url})")

                if st.button(
                    "✓ Select First Link For BOM", key=f"selectfirst_{item_id}"
                ):
                    if products:
                        st.session_state.project_state["selected_products"][item_id] = (
                            products[0]
                        )
                        st.success("Selected!")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📦 Generate Bill of Materials", type="primary"):
            st.session_state.project_state["phase"] = "bom"
            st.rerun()
    with col2:
        if st.button("← Back to Designs"):
            st.session_state.project_state["phase"] = "design"
            st.rerun()


def render_bom_phase():
    st.header("📦 Bill of Materials")

    furniture = st.session_state.project_state.get("furniture_items", [])
    product_matches = st.session_state.project_state.get("product_matches", {})
    selected_products = st.session_state.project_state.get("selected_products", {})

    bom_items = []
    total_estimate = 0

    for item in furniture:
        item_id = str(item.get("id", 0))
        product = selected_products.get(item_id)
        if not product:
            products = product_matches.get(item_id, [])
            product = products[0] if products else {}

        price_low = item.get("price_low", 200)
        price_high = item.get("price_high", 800)
        price_est = (price_low + price_high) / 2

        bom_items.append(
            {
                "name": item.get("name", "Item"),
                "query": item.get("search_query", ""),
                "retailer": product.get("retailer", "Various"),
                "url": product.get("url", ""),
                "price_estimate": price_est,
                "attrs": item.get("attrs", {}),
            }
        )
        total_estimate += price_est

    labor = 1500
    contingency = (total_estimate + labor) * 0.1
    grand_total = total_estimate + labor + contingency

    col1, col2, col3 = st.columns(3)
    col1.metric("Products", f"${total_estimate:,.0f}")
    col2.metric("Labor (est.)", f"${labor:,}")
    col3.metric("**Total**", f"${grand_total:,.0f}")

    st.divider()
    st.markdown("### Items")

    for item in bom_items:
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.write(f"**{item['name']}**")
            st.caption(f"Query: {item.get('query','')}")
            a = item.get("attrs", {})
            if a:
                st.caption(
                    f"Attrs: {a.get('primary_color')} {a.get('material')} • {a.get('style')}"
                )
        with col2:
            st.write(f"~${item['price_estimate']:,.0f}")
        with col3:
            if item.get("url"):
                st.markdown(f"[🔗 Shop]({item['url']})")

    st.divider()

    st.session_state.project_state["bom"] = {
        "items": bom_items,
        "products_total": total_estimate,
        "labor": labor,
        "contingency": contingency,
        "grand_total": grand_total,
    }

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Complete Project", type="primary"):
            st.session_state.project_state["phase"] = "complete"
            st.rerun()
    with col2:
        if st.button("← Back to Products"):
            st.session_state.project_state["phase"] = "products"
            st.rerun()


def render_complete_phase():
    st.header("✅ Project Complete!")
    st.balloons()

    design = st.session_state.project_state.get("selected_design", {})
    bom = st.session_state.project_state.get("bom", {})
    design_image = st.session_state.project_state.get("selected_design_image")
    original_image = (
        st.session_state.project_state["images"][0]["data"]
        if st.session_state.project_state["images"]
        else None
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Before")
        if original_image:
            st.image(base64.b64decode(original_image), use_column_width=True)
    with col2:
        st.markdown("### After")
        if design_image:
            st.image(base64.b64decode(design_image), use_column_width=True)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Design:** {design.get('name', 'N/A')}")
        st.markdown(f"**Concept:** {design.get('concept', 'N/A')}")
    with col2:
        st.metric("Total Budget", f"${bom.get('grand_total', 0):,.0f}")

    st.divider()
    st.markdown("### 🛒 Shopping Links")
    for item in bom.get("items", []):
        if item.get("url"):
            st.markdown(f"- [{item['name']} - {item['retailer']}]({item['url']})")

    st.divider()
    project_data = {
        "design": design,
        "bom": bom,
        "preferences": st.session_state.project_state.get("preferences", {}),
        "furniture_instances": st.session_state.project_state.get(
            "furniture_instances", []
        ),
    }

    st.download_button(
        "📥 Download Project Data",
        data=json.dumps(project_data, indent=2),
        file_name="omnirenovation_project.json",
        mime="application/json",
    )

    if st.button("🔄 Start New Project", type="primary"):
        st.session_state.project_state = DEFAULT_STATE.copy()
        st.rerun()


# =========================================================
# ====================== MAIN APP =========================
# =========================================================


def main():
    render_header()
    st.divider()

    phase = st.session_state.project_state["phase"]
    if phase == "upload":
        render_upload_phase()
    elif phase == "valuation":
        render_valuation_phase()
    elif phase == "design":
        render_design_phase()
    elif phase == "products":
        render_products_phase()
    elif phase == "bom":
        render_bom_phase()
    elif phase == "complete":
        render_complete_phase()
    else:
        render_upload_phase()


if __name__ == "__main__":
    main()
