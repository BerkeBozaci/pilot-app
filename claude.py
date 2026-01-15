"""
OmniRenovation AI - Phase 1 Pilot v17
=====================================
COMPLETE FURNITURE ANALYSIS PIPELINE
claude.py

Features:
- Google Gemini 2.5 Flash for design generation
- Grounded SAM 2 for furniture segmentation (via API)
- CLIP for material/style classification
- Color analysis with K-means clustering
- Detailed attribute extraction
- Smart search query generation

Open-Source Components:
- Segmentation: SAM 2 (Meta)
- Detection: Grounding DINO / Florence-2
- Classification: CLIP (OpenAI)
- Color Analysis: scikit-learn K-means
"""

import streamlit as st
import requests
import json
import base64
import re
import io
import colorsys
from collections import Counter
from PIL import Image
from urllib.parse import quote_plus
import numpy as np

# ============== PAGE CONFIG ==============
st.set_page_config(page_title="OmniRenovation AI", page_icon="🏠", layout="wide")

# ============== CONSTANTS ==============

# Furniture categories for detection
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

# Materials for CLIP classification
MATERIALS = [
    "leather",
    "genuine leather",
    "faux leather",
    "fabric",
    "linen",
    "velvet",
    "cotton",
    "wool",
    "polyester",
    "microfiber",
    "wood",
    "solid wood",
    "oak wood",
    "walnut wood",
    "pine wood",
    "teak wood",
    "bamboo",
    "rattan",
    "wicker",
    "metal",
    "steel",
    "stainless steel",
    "brass",
    "gold metal",
    "chrome",
    "iron",
    "copper",
    "glass",
    "tempered glass",
    "frosted glass",
    "marble",
    "stone",
    "granite",
    "concrete",
    "ceramic",
    "porcelain",
    "plastic",
    "acrylic",
    "lucite",
]

# Styles for CLIP classification
STYLES = [
    "modern",
    "contemporary",
    "minimalist",
    "ultra-modern",
    "traditional",
    "classic",
    "vintage",
    "antique",
    "retro",
    "mid-century modern",
    "scandinavian",
    "nordic",
    "danish modern",
    "industrial",
    "rustic",
    "farmhouse",
    "country",
    "bohemian",
    "boho",
    "eclectic",
    "coastal",
    "beach",
    "nautical",
    "art deco",
    "hollywood regency",
    "glam",
    "luxury",
    "transitional",
    "casual",
]

# Color name mapping
COLOR_NAMES = {
    (255, 255, 255): "white",
    (245, 245, 220): "beige",
    (255, 248, 220): "cream",
    (255, 255, 240): "ivory",
    (210, 180, 140): "tan",
    (139, 69, 19): "brown",
    (160, 82, 45): "cognac",
    (101, 67, 33): "walnut",
    (128, 101, 64): "oak",
    (62, 43, 35): "espresso",
    (0, 0, 0): "black",
    (54, 69, 79): "charcoal",
    (128, 128, 128): "grey",
    (192, 192, 192): "silver",
    (0, 0, 128): "navy",
    (0, 0, 255): "blue",
    (0, 128, 128): "teal",
    (0, 128, 0): "green",
    (128, 128, 0): "olive",
    (255, 0, 0): "red",
    (128, 0, 32): "burgundy",
    (255, 165, 0): "orange",
    (255, 255, 0): "yellow",
    (255, 215, 0): "gold",
    (255, 192, 203): "pink",
    (128, 0, 128): "purple",
    (230, 230, 250): "lavender",
}

# Size categories based on furniture type
SIZE_CATEGORIES = {
    "sofa": ["2-seater", "3-seater", "4-seater", "sectional"],
    "chair": ["single", "oversized", "compact"],
    "table": ["small", "medium", "large", "extra large"],
    "lamp": ["small", "medium", "large", "floor-standing"],
    "rug": ["accent (3x5)", "medium (5x8)", "large (8x10)", "room-size (9x12+)"],
    "default": ["small", "medium", "large"],
}

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
    "furniture_analysis": None,  # NEW: Detailed furniture analysis
    "furniture_items": [],
    "product_matches": {},
    "selected_products": {},
    "bom": None,
}

if "project_state" not in st.session_state:
    st.session_state.project_state = DEFAULT_STATE.copy()


# ============== API KEY MANAGEMENT ==============


def get_api_key(name: str) -> str:
    """Get API key from session state"""
    key_map = {
        "GEMINI_API_KEY": "gemini_api_key",
        "ANTHROPIC_API_KEY": "anthropic_api_key",
        "REPLICATE_API_TOKEN": "replicate_api_key",
    }
    session_key = key_map.get(name, name.lower())
    return st.session_state.get(session_key, "").strip()


def validate_gemini_key(api_key: str) -> tuple:
    """Validate Gemini API key"""
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
    """Validate Anthropic API key"""
    if not api_key:
        return False, "API key is empty"
    if not api_key.startswith("sk-ant-"):
        return False, "Invalid format"
    return True, "Format valid"


# ============== COLOR ANALYSIS ==============


def rgb_to_hex(rgb):
    """Convert RGB tuple to hex string"""
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def hex_to_rgb(hex_color):
    """Convert hex string to RGB tuple"""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def get_color_name(rgb):
    """Get the closest color name for an RGB value"""
    min_dist = float("inf")
    closest_name = "unknown"

    for ref_rgb, name in COLOR_NAMES.items():
        dist = sum((a - b) ** 2 for a, b in zip(rgb, ref_rgb))
        if dist < min_dist:
            min_dist = dist
            closest_name = name

    return closest_name


def extract_dominant_colors(
    image_base64: str, n_colors: int = 5, mask_base64: str = None
) -> dict:
    """
    Extract dominant colors from an image using K-means clustering

    Args:
        image_base64: Base64 encoded image
        n_colors: Number of dominant colors to extract
        mask_base64: Optional mask to focus on specific region

    Returns:
        Dictionary with color analysis results
    """
    try:
        # Decode image
        img_data = base64.b64decode(image_base64)
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        img_array = np.array(img)

        # Apply mask if provided
        if mask_base64:
            mask_data = base64.b64decode(mask_base64)
            mask = Image.open(io.BytesIO(mask_data)).convert("L")
            mask = mask.resize(img.size)
            mask_array = np.array(mask)
            # Only use pixels where mask is non-zero
            pixels = img_array[mask_array > 128]
        else:
            # Reshape to list of pixels
            pixels = img_array.reshape(-1, 3)

        if len(pixels) == 0:
            return {"error": "No pixels to analyze"}

        # Sample pixels if too many (for performance)
        if len(pixels) > 10000:
            indices = np.random.choice(len(pixels), 10000, replace=False)
            pixels = pixels[indices]

        # K-means clustering
        from sklearn.cluster import KMeans

        kmeans = KMeans(
            n_clusters=min(n_colors, len(pixels)), random_state=42, n_init=10
        )
        kmeans.fit(pixels)

        # Get cluster centers and counts
        colors = kmeans.cluster_centers_
        labels = kmeans.labels_
        counts = Counter(labels)

        # Sort by frequency
        sorted_colors = sorted(
            [(colors[i], counts[i]) for i in range(len(colors))],
            key=lambda x: x[1],
            reverse=True,
        )

        # Build result
        result = {
            "dominant": {
                "rgb": tuple(int(c) for c in sorted_colors[0][0]),
                "hex": rgb_to_hex(sorted_colors[0][0]),
                "name": get_color_name(tuple(int(c) for c in sorted_colors[0][0])),
                "percentage": sorted_colors[0][1] / len(labels) * 100,
            },
            "palette": [],
        }

        for color, count in sorted_colors[:n_colors]:
            rgb = tuple(int(c) for c in color)
            result["palette"].append(
                {
                    "rgb": rgb,
                    "hex": rgb_to_hex(color),
                    "name": get_color_name(rgb),
                    "percentage": count / len(labels) * 100,
                }
            )

        return result

    except ImportError:
        # Fallback if sklearn not available
        return extract_dominant_colors_simple(image_base64)
    except Exception as e:
        return {"error": str(e)}


def extract_dominant_colors_simple(image_base64: str) -> dict:
    """Simple color extraction without sklearn"""
    try:
        img_data = base64.b64decode(image_base64)
        img = Image.open(io.BytesIO(img_data)).convert("RGB")

        # Resize for speed
        img = img.resize((100, 100))

        # Get most common colors
        pixels = list(img.getdata())

        # Quantize colors (reduce to 32 levels per channel)
        quantized = [(r // 8 * 8, g // 8 * 8, b // 8 * 8) for r, g, b in pixels]

        # Count occurrences
        color_counts = Counter(quantized)
        most_common = color_counts.most_common(5)

        if not most_common:
            return {"error": "No colors found"}

        result = {
            "dominant": {
                "rgb": most_common[0][0],
                "hex": rgb_to_hex(most_common[0][0]),
                "name": get_color_name(most_common[0][0]),
                "percentage": most_common[0][1] / len(pixels) * 100,
            },
            "palette": [],
        }

        for color, count in most_common:
            result["palette"].append(
                {
                    "rgb": color,
                    "hex": rgb_to_hex(color),
                    "name": get_color_name(color),
                    "percentage": count / len(pixels) * 100,
                }
            )

        return result

    except Exception as e:
        return {"error": str(e)}


# ============== FURNITURE ANALYSIS WITH CLAUDE ==============


def analyze_furniture_detailed(image_base64: str) -> dict:
    """
    Use Claude Vision to perform detailed furniture analysis

    This is the main analysis function that:
    1. Detects all furniture items
    2. Analyzes colors, materials, styles
    3. Estimates sizes
    4. Generates search queries
    """
    api_key = get_api_key("ANTHROPIC_API_KEY")

    if not api_key:
        return {"error": "Anthropic API key not configured"}

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }

    analysis_prompt = """Analyze this interior design image and identify ALL furniture and decor items.

For EACH item, provide detailed analysis in the following JSON format:

{
    "furniture_items": [
        {
            "id": 1,
            "category": "sofa",
            "subcategory": "3-seater sofa",
            "confidence": 0.95,
            "position": "center-left of room",
            "colors": {
                "primary": "beige",
                "primary_hex": "#F5F5DC",
                "secondary": "brown",
                "secondary_hex": "#8B4513",
                "accent": null
            },
            "material": {
                "primary": "linen fabric",
                "secondary": "wooden legs",
                "texture": "textured weave"
            },
            "style": {
                "primary": "modern",
                "secondary": "scandinavian",
                "era": "contemporary"
            },
            "size": {
                "category": "3-seater",
                "estimated_width": "220cm",
                "estimated_depth": "90cm",
                "estimated_height": "85cm"
            },
            "details": {
                "leg_style": "tapered wooden legs",
                "arm_style": "track arms",
                "back_style": "tight back with cushions",
                "cushions": "3 seat cushions, 2 back cushions",
                "special_features": ["removable covers", "high arms"]
            },
            "search_query": "modern 3-seater beige linen sofa tapered wooden legs",
            "alternative_queries": [
                "scandinavian beige fabric sofa",
                "contemporary linen couch wooden legs"
            ],
            "price_estimate": {
                "budget": "$800-1200",
                "mid_range": "$1200-2500",
                "premium": "$2500-5000"
            },
            "similar_brands": ["Article", "West Elm", "IKEA", "CB2"]
        }
    ],
    "room_analysis": {
        "room_type": "living room",
        "style_theme": "modern scandinavian",
        "color_scheme": "neutral with warm accents",
        "lighting": "natural daylight with ambient lighting"
    }
}

IMPORTANT INSTRUCTIONS:
1. Identify EVERY visible furniture item including small decor
2. Be specific about colors (use actual color names and hex codes)
3. Be specific about materials (e.g., "velvet fabric" not just "fabric")
4. Include both primary and secondary materials
5. Estimate realistic sizes based on room proportions
6. Generate detailed search queries that would find similar products
7. Include multiple alternative search queries
8. Suggest realistic price ranges for different quality tiers
9. List brands that typically sell similar items

Return ONLY valid JSON, no markdown formatting or extra text."""

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 8000,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_base64,
                                },
                            },
                            {"type": "text", "text": analysis_prompt},
                        ],
                    }
                ],
            },
            timeout=120,
        )

        if response.status_code == 200:
            data = response.json()
            text = data["content"][0]["text"]

            # Extract JSON from response
            # Try to find JSON object
            json_match = re.search(r"\{[\s\S]*\}", text)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                    return result
                except json.JSONDecodeError:
                    pass

            return {"error": "Could not parse response as JSON", "raw": text[:500]}
        else:
            return {
                "error": f"API Error: {response.status_code} - {response.text[:200]}"
            }

    except Exception as e:
        return {"error": str(e)}


def enhance_with_color_analysis(furniture_analysis: dict, image_base64: str) -> dict:
    """
    Enhance furniture analysis with actual color extraction from image
    """
    try:
        # Extract colors from the full image
        color_result = extract_dominant_colors(image_base64, n_colors=8)

        if "error" not in color_result:
            furniture_analysis["image_color_analysis"] = color_result

        return furniture_analysis

    except Exception as e:
        furniture_analysis["color_analysis_error"] = str(e)
        return furniture_analysis


# ============== GEMINI IMAGE GENERATION ==============


def generate_design_with_gemini(
    image_base64: str,
    style: str,
    variation: str,
    room_type: str = "room",
    max_retries: int = 3,
) -> dict:
    """Generate interior design using Google Gemini 2.5 Flash Image"""
    api_key = get_api_key("GEMINI_API_KEY")

    if not api_key:
        return {"error": "Gemini API key not configured"}

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
                st.info(f"🎨 Generating with {model_name} (attempt {attempt + 1})...")

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
                    last_error = "No image in response"
                    break
                elif response.status_code == 429:
                    import time

                    wait_time = (2**attempt) * 5
                    st.warning(f"⏳ Rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                elif response.status_code == 404:
                    last_error = f"Model {model_name} not found"
                    break
                else:
                    last_error = f"HTTP {response.status_code}"
                    break
            except Exception as e:
                last_error = str(e)
                continue

    return {"error": f"Generation failed: {last_error}"}


# ============== CLAUDE FOR ROOM ANALYSIS ==============


def analyze_room_with_claude(images: list, preferences: dict) -> dict:
    """Use Claude to analyze the room"""
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
        else:
            return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def create_design_concepts(valuation: dict, preferences: dict) -> dict:
    """Generate design concepts using Claude"""
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
        else:
            return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}


# ============== PRODUCT SEARCH ==============


class OpenSourceProductSearch:
    """Open-source product search using retailer URLs"""

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
            "name": "Article",
            "search_url": "https://www.article.com/search?query={query}",
            "icon": "🪴",
        },
        {
            "name": "Overstock",
            "search_url": "https://www.overstock.com/Home-Garden/?keywords={query}",
            "icon": "📦",
        },
    ]

    @classmethod
    def search(cls, query: str) -> list:
        encoded = quote_plus(query)
        return [
            {
                "product_name": f"{r['icon']} {r['name']}",
                "url": r["search_url"].replace("{query}", encoded),
                "retailer": r["name"],
                "source": "retailer_search",
            }
            for r in cls.RETAILERS
        ]

    @classmethod
    def get_retailer_list(cls) -> list:
        return [r["name"] for r in cls.RETAILERS]


# ============== IMAGE PROCESSING ==============


def process_uploaded_image(uploaded_file) -> dict:
    """Process uploaded image file"""
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

        return {"success": True, "data": base64_data, "name": uploaded_file.name}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============== UI COMPONENTS ==============


def render_header():
    """Render page header"""
    st.title("🏠 OmniRenovation AI")
    st.caption("AI-Powered Interior Design with Detailed Furniture Analysis")

    phases = [
        "Upload",
        "Analysis",
        "Design",
        "Furniture",
        "Products",
        "BOM",
        "Complete",
    ]
    phase_map = {
        "upload": 0,
        "valuation": 1,
        "design": 2,
        "furniture_analysis": 3,
        "products": 4,
        "bom": 5,
        "complete": 6,
    }
    current = phase_map.get(st.session_state.project_state["phase"], 0)

    cols = st.columns(7)
    for i, phase in enumerate(phases):
        with cols[i]:
            if i < current:
                st.success(f"✓ {phase}")
            elif i == current:
                st.info(f"→ {phase}")
            else:
                st.text(f"○ {phase}")


def render_upload_phase():
    """Render upload phase"""
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
                st.image(Image.open(f), use_column_width=True)

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
    """Render room analysis phase"""
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
    col2.metric("Condition", assessment.get("current_condition", "N/A").title())
    col3.metric("Est. Cost", f"${costs.get('mid', 0):,}")

    if st.session_state.project_state["images"]:
        st.image(
            base64.b64decode(st.session_state.project_state["images"][0]["data"]),
            caption="Your Room",
            use_column_width=True,
        )

    st.divider()

    if st.button("🎨 Generate Designs", type="primary"):
        st.session_state.project_state["phase"] = "design"
        st.rerun()


def render_design_phase():
    """Render design generation phase"""
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
                        use_column_width=True,
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
                    with st.spinner("Generating..."):
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
                    st.session_state.project_state["selected_design_image"] = (
                        design_images[num]["image_base64"]
                    )
                else:
                    st.session_state.project_state["selected_design_image"] = (
                        original_image
                    )
                st.success("Selected!")

        st.divider()

    if st.session_state.project_state.get("selected_design"):
        st.success(
            f"**Selected:** {st.session_state.project_state['selected_design'].get('name')}"
        )

        if st.button("🔍 Analyze Furniture", type="primary"):
            st.session_state.project_state["phase"] = "furniture_analysis"
            st.rerun()


def render_furniture_analysis_phase():
    """NEW: Render detailed furniture analysis phase"""
    st.header("🔍 Furniture Analysis")

    design_image = st.session_state.project_state.get("selected_design_image")

    if not design_image:
        st.error("No design image selected")
        return

    # Run analysis if not done
    if not st.session_state.project_state.get("furniture_analysis"):
        with st.spinner(
            "🔬 Analyzing furniture in detail... This may take 30-60 seconds."
        ):
            # Get detailed analysis from Claude
            analysis = analyze_furniture_detailed(design_image)

            # Enhance with color analysis
            if "error" not in analysis:
                analysis = enhance_with_color_analysis(analysis, design_image)

            st.session_state.project_state["furniture_analysis"] = analysis
            st.rerun()

    analysis = st.session_state.project_state["furniture_analysis"]

    if "error" in analysis:
        st.error(f"Analysis failed: {analysis['error']}")
        if st.button("🔄 Retry Analysis"):
            st.session_state.project_state["furniture_analysis"] = None
            st.rerun()
        return

    # Show the design image
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(
            base64.b64decode(design_image),
            caption="Analyzed Design",
            use_column_width=True,
        )

    with col2:
        # Room analysis summary
        room_info = analysis.get("room_analysis", {})
        st.markdown("### 🏠 Room Summary")
        st.write(f"**Type:** {room_info.get('room_type', 'N/A')}")
        st.write(f"**Style:** {room_info.get('style_theme', 'N/A')}")
        st.write(f"**Color Scheme:** {room_info.get('color_scheme', 'N/A')}")

        # Image color analysis
        if "image_color_analysis" in analysis:
            st.markdown("### 🎨 Detected Colors")
            color_data = analysis["image_color_analysis"]
            if "palette" in color_data:
                color_cols = st.columns(min(5, len(color_data["palette"])))
                for i, color in enumerate(color_data["palette"][:5]):
                    with color_cols[i]:
                        st.color_picker(
                            color["name"],
                            color["hex"],
                            disabled=True,
                            key=f"detected_color_{i}",
                        )
                        st.caption(f"{color['percentage']:.0f}%")

    st.divider()

    # Detailed furniture items
    st.markdown("## 🛋️ Detected Furniture Items")

    furniture_items = analysis.get("furniture_items", [])

    if not furniture_items:
        st.warning("No furniture items detected")

    # Store for product search
    search_items = []

    for item in furniture_items:
        with st.expander(
            f"**{item.get('id', '?')}. {item.get('subcategory', item.get('category', 'Item')).title()}** (Confidence: {item.get('confidence', 0)*100:.0f}%)",
            expanded=True,
        ):

            col1, col2, col3 = st.columns([1, 1, 1])

            with col1:
                st.markdown("#### 🎨 Colors")
                colors = item.get("colors", {})
                if colors.get("primary"):
                    st.write(f"**Primary:** {colors.get('primary', 'N/A')}")
                    if colors.get("primary_hex"):
                        st.color_picker(
                            "",
                            colors["primary_hex"],
                            disabled=True,
                            key=f"item_{item['id']}_primary",
                        )
                if colors.get("secondary"):
                    st.write(f"**Secondary:** {colors.get('secondary', 'N/A')}")

                st.markdown("#### 📐 Size")
                size = item.get("size", {})
                st.write(f"**Category:** {size.get('category', 'N/A')}")
                if size.get("estimated_width"):
                    st.write(
                        f"**Dimensions:** {size.get('estimated_width', '')} × {size.get('estimated_depth', '')} × {size.get('estimated_height', '')}"
                    )

            with col2:
                st.markdown("#### 🧱 Materials")
                materials = item.get("material", {})
                st.write(f"**Primary:** {materials.get('primary', 'N/A')}")
                if materials.get("secondary"):
                    st.write(f"**Secondary:** {materials.get('secondary', 'N/A')}")
                if materials.get("texture"):
                    st.write(f"**Texture:** {materials.get('texture', 'N/A')}")

                st.markdown("#### 🎭 Style")
                style = item.get("style", {})
                st.write(f"**Primary:** {style.get('primary', 'N/A')}")
                if style.get("secondary"):
                    st.write(f"**Secondary:** {style.get('secondary', 'N/A')}")

            with col3:
                st.markdown("#### 💰 Price Estimates")
                prices = item.get("price_estimate", {})
                st.write(f"**Budget:** {prices.get('budget', 'N/A')}")
                st.write(f"**Mid-range:** {prices.get('mid_range', 'N/A')}")
                st.write(f"**Premium:** {prices.get('premium', 'N/A')}")

                brands = item.get("similar_brands", [])
                if brands:
                    st.markdown("#### 🏷️ Similar Brands")
                    st.write(", ".join(brands[:4]))

            # Details section
            details = item.get("details", {})
            if details:
                st.markdown("#### 📝 Details")
                detail_cols = st.columns(3)
                detail_items = list(details.items())
                for i, (key, value) in enumerate(detail_items[:6]):
                    with detail_cols[i % 3]:
                        if isinstance(value, list):
                            st.write(
                                f"**{key.replace('_', ' ').title()}:** {', '.join(value)}"
                            )
                        else:
                            st.write(f"**{key.replace('_', ' ').title()}:** {value}")

            # Search queries
            st.markdown("#### 🔍 Search Queries")
            main_query = item.get("search_query", "")
            if main_query:
                st.code(main_query, language=None)

            alt_queries = item.get("alternative_queries", [])
            if alt_queries:
                st.caption("**Alternative searches:**")
                for q in alt_queries[:3]:
                    st.caption(f"• {q}")

            # Add to search items
            search_items.append(
                {
                    "id": item.get("id", 0),
                    "name": item.get("subcategory", item.get("category", "Item")),
                    "search_query": main_query,
                    "alternative_queries": alt_queries,
                    "price_estimate": prices,
                }
            )

    # Save for product search
    st.session_state.project_state["furniture_items"] = search_items

    st.divider()

    # Navigation
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🛒 Find Products", type="primary"):
            st.session_state.project_state["phase"] = "products"
            st.rerun()

    with col2:
        if st.button("← Back to Designs"):
            st.session_state.project_state["furniture_analysis"] = None
            st.session_state.project_state["phase"] = "design"
            st.rerun()


def render_products_phase():
    """Render product search phase"""
    st.header("🛒 Product Search")

    st.info(f"**Searching:** {', '.join(OpenSourceProductSearch.get_retailer_list())}")

    furniture_items = st.session_state.project_state.get("furniture_items", [])
    product_matches = st.session_state.project_state.get("product_matches", {})

    if not furniture_items:
        st.warning("No furniture items to search")
        return

    for item in furniture_items:
        item_id = str(item.get("id", 0))

        with st.expander(f"🔍 **{item.get('name', 'Item')}**", expanded=True):
            # Show the search query
            query = item.get("search_query", item.get("name", "furniture"))
            st.caption(f"Searching: *{query}*")

            # Get products if not cached
            if item_id not in product_matches:
                products = OpenSourceProductSearch.search(query)
                product_matches[item_id] = products
                st.session_state.project_state["product_matches"] = product_matches

            products = product_matches.get(item_id, [])

            if products:
                cols = st.columns(4)
                for i, product in enumerate(products[:4]):
                    with cols[i]:
                        st.markdown(f"**{product.get('product_name', '')}**")
                        url = product.get("url", "")
                        if url:
                            st.markdown(f"[🔗 Search →]({url})")

                        if st.button("✓ Select", key=f"prod_{item_id}_{i}"):
                            st.session_state.project_state["selected_products"][
                                item_id
                            ] = product
                            st.success("Selected!")

            # Show alternative queries
            alt_queries = item.get("alternative_queries", [])
            if alt_queries:
                st.caption("**Try also:**")
                for q in alt_queries[:2]:
                    st.caption(f"• {q}")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📦 Generate BOM", type="primary"):
            st.session_state.project_state["phase"] = "bom"
            st.rerun()

    with col2:
        if st.button("← Back to Analysis"):
            st.session_state.project_state["phase"] = "furniture_analysis"
            st.rerun()


def render_bom_phase():
    """Render Bill of Materials phase"""
    st.header("📦 Bill of Materials")

    furniture = st.session_state.project_state.get("furniture_items", [])
    product_matches = st.session_state.project_state.get("product_matches", {})
    selected_products = st.session_state.project_state.get("selected_products", {})

    bom_items = []
    total_low = 0
    total_high = 0

    for item in furniture:
        item_id = str(item.get("id", 0))

        product = selected_products.get(item_id)
        if not product:
            products = product_matches.get(item_id, [])
            product = products[0] if products else {}

        # Extract price from estimate
        price_est = item.get("price_estimate", {})

        # Parse budget range
        budget_str = price_est.get("budget", "$500-1000")
        try:
            prices = re.findall(r"\$?([\d,]+)", budget_str)
            if len(prices) >= 2:
                low = int(prices[0].replace(",", ""))
                high = int(prices[1].replace(",", ""))
            else:
                low, high = 500, 1000
        except:
            low, high = 500, 1000

        bom_items.append(
            {
                "name": item.get("name", "Item"),
                "retailer": product.get("retailer", "Various"),
                "url": product.get("url", ""),
                "price_low": low,
                "price_high": high,
                "search_query": item.get("search_query", ""),
            }
        )

        total_low += low
        total_high += high

    # Summary
    labor = 1500

    col1, col2, col3 = st.columns(3)
    col1.metric("Products (Low)", f"${total_low:,}")
    col2.metric("Products (High)", f"${total_high:,}")
    col3.metric("Est. Labor", f"${labor:,}")

    st.metric("**Total Range**", f"${total_low + labor:,} - ${total_high + labor:,}")

    st.divider()

    st.markdown("### Items")

    for item in bom_items:
        col1, col2, col3 = st.columns([3, 1, 1])

        with col1:
            st.write(f"**{item['name']}**")
            st.caption(f"Search: {item['search_query'][:50]}...")

        with col2:
            st.write(f"${item['price_low']:,} - ${item['price_high']:,}")

        with col3:
            if item.get("url"):
                st.markdown(f"[🔗 Shop]({item['url']})")

    st.session_state.project_state["bom"] = {
        "items": bom_items,
        "total_low": total_low,
        "total_high": total_high,
        "labor": labor,
    }

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        if st.button("✅ Complete", type="primary"):
            st.session_state.project_state["phase"] = "complete"
            st.rerun()

    with col2:
        if st.button("← Back"):
            st.session_state.project_state["phase"] = "products"
            st.rerun()


def render_complete_phase():
    """Render completion phase"""
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
    with col2:
        st.metric(
            "Budget Range",
            f"${bom.get('total_low', 0) + bom.get('labor', 0):,} - ${bom.get('total_high', 0) + bom.get('labor', 0):,}",
        )

    st.divider()

    st.markdown("### 🛒 Shopping Links")
    for item in bom.get("items", []):
        if item.get("url"):
            st.markdown(f"- [{item['name']}]({item['url']})")

    # Download
    project_data = {
        "design": design,
        "bom": bom,
        "furniture_analysis": st.session_state.project_state.get(
            "furniture_analysis", {}
        ),
        "preferences": st.session_state.project_state.get("preferences", {}),
    }

    st.download_button(
        "📥 Download Project",
        data=json.dumps(project_data, indent=2),
        file_name="omnirenovation_project.json",
        mime="application/json",
    )

    if st.button("🔄 New Project", type="primary"):
        st.session_state.project_state = DEFAULT_STATE.copy()
        st.rerun()


# ============== MAIN ==============


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
    elif phase == "furniture_analysis":
        render_furniture_analysis_phase()
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
