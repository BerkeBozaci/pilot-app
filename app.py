"""
OmniRenovation AI - Phase 1 Pilot v15
- FIXED: Proper image upload for Fal.ai and Replicate
- Structure-preserving design generation
- Working API endpoints
"""

import streamlit as st
import anthropic
import requests
import json
import base64
import re
import time
from datetime import datetime
from io import BytesIO
from PIL import Image
from urllib.parse import quote_plus

st.set_page_config(page_title="OmniRenovation AI", page_icon="🏠", layout="wide")

# Session state
DEFAULT_STATE = {
    "phase": "upload",
    "images": [],
    "preferences": {},
    "valuation": None,
    "designs": None,
    "design_images": {},
    "selected_design": None,
    "selected_design_image": None,
    "furniture_items": [],
    "product_matches": {},
    "selected_products": {},
    "bom": None,
}

if "project_state" not in st.session_state:
    st.session_state.project_state = DEFAULT_STATE.copy()


# ============== API HELPERS ==============


def get_api_key(name: str) -> str:
    """Get API key from secrets or session state"""
    if hasattr(st, "secrets"):
        try:
            val = st.secrets.get(name, "")
            if val:
                return val
        except:
            pass

    key_variants = {
        "FAL_KEY": ["fal_key", "FAL_KEY"],
        "REPLICATE_API_TOKEN": ["replicate_api_token", "REPLICATE_API_TOKEN"],
        "ANTHROPIC_API_KEY": ["anthropic_api_key", "ANTHROPIC_API_KEY"],
        "SERPAPI_KEY": ["serpapi_key", "SERPAPI_KEY"],
    }

    variants = key_variants.get(name, [name.lower(), name])
    for variant in variants:
        val = st.session_state.get(variant, "")
        if val:
            return val
    return ""


# ============== IMAGE UPLOAD HELPERS ==============


def upload_image_to_fal(image_base64: str, api_key: str) -> str:
    """Upload image to Fal.ai and get a URL back"""
    try:
        # Decode base64 to bytes
        image_bytes = base64.b64decode(image_base64)

        # Upload to Fal.ai's file storage
        response = requests.post(
            "https://fal.run/fal-ai/any-llm/files/upload",
            headers={
                "Authorization": f"Key {api_key}",
            },
            files={"file": ("image.jpg", image_bytes, "image/jpeg")},
            timeout=30,
        )

        if response.status_code == 200:
            data = response.json()
            return data.get("url", "")
    except Exception as e:
        st.warning(f"Fal upload error: {e}")

    return ""


def upload_to_tmpfiles(image_base64: str) -> str:
    """Upload to tmpfiles.org as fallback (temporary hosting)"""
    try:
        image_bytes = base64.b64decode(image_base64)
        response = requests.post(
            "https://tmpfiles.org/api/v1/upload",
            files={"file": ("image.jpg", image_bytes, "image/jpeg")},
            timeout=30,
        )
        if response.status_code == 200:
            data = response.json()
            url = data.get("data", {}).get("url", "")
            # Convert to direct link
            if url:
                return url.replace("tmpfiles.org/", "tmpfiles.org/dl/")
    except Exception as e:
        st.warning(f"tmpfiles upload error: {e}")
    return ""


# ============== DESIGN GENERATION ==============


def generate_with_replicate(
    image_base64: str, prompt: str, negative_prompt: str, strength: float = 0.6
) -> dict:
    """
    Generate using Replicate API with proper image handling
    Using official model references (not version hashes) for reliability
    """
    api_key = get_api_key("REPLICATE_API_TOKEN")
    if not api_key:
        return {"error": "Replicate API key not configured"}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Prefer": "wait",  # Wait for result instead of polling
    }

    # Replicate accepts data URIs for images
    image_uri = f"data:image/jpeg;base64,{image_base64}"

    # Use official model references (more reliable than version hashes)
    models = [
        {
            "name": "interior-design",
            "model": "adirik/interior-design",
            "input": {
                "image": image_uri,
                "prompt": prompt,
                "guidance_scale": 15,
                "negative_prompt": negative_prompt,
                "prompt_strength": strength,
                "num_inference_steps": 50,
            },
        },
        {
            "name": "realvisxl-v4",
            "model": "adirik/realvisxl-v4.0-lightning",
            "input": {
                "image": image_uri,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "strength": strength,
                "num_inference_steps": 8,
                "guidance_scale": 2.0,
            },
        },
    ]

    for model in models:
        try:
            st.info(f"Trying Replicate {model['name']}...")

            # Use the official model endpoint (not version-based)
            response = requests.post(
                f"https://api.replicate.com/v1/models/{model['model']}/predictions",
                headers=headers,
                json={"input": model["input"]},
                timeout=120,  # Wait up to 2 minutes
            )

            if response.status_code == 429:
                st.warning("Rate limited - waiting 15 seconds...")
                time.sleep(15)
                continue

            if response.status_code == 422:
                st.warning(f"Model error: {response.text[:200]}")
                continue

            if response.status_code not in [200, 201]:
                st.warning(
                    f"Replicate error: {response.status_code} - {response.text[:200]}"
                )
                continue

            result = response.json()

            # Check if we got output directly (with Prefer: wait header)
            output = result.get("output")
            if output:
                img_url = output[0] if isinstance(output, list) else output
                img_response = requests.get(img_url, timeout=30)
                if img_response.status_code == 200:
                    return {
                        "success": True,
                        "image_base64": base64.b64encode(img_response.content).decode(),
                        "method": f"replicate_{model['name']}",
                    }

            # If not, we need to poll
            prediction_id = result.get("id")
            if not prediction_id:
                continue

            # Poll for result
            poll_headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            progress = st.progress(0)
            for i in range(60):  # 2 minutes max
                time.sleep(2)
                progress.progress(min(i * 2, 100))

                poll_result = requests.get(
                    f"https://api.replicate.com/v1/predictions/{prediction_id}",
                    headers=poll_headers,
                    timeout=30,
                ).json()

                status = poll_result.get("status")

                if status == "succeeded":
                    progress.progress(100)
                    output = poll_result.get("output")
                    if output:
                        img_url = output[0] if isinstance(output, list) else output
                        img_response = requests.get(img_url, timeout=30)
                        if img_response.status_code == 200:
                            return {
                                "success": True,
                                "image_base64": base64.b64encode(
                                    img_response.content
                                ).decode(),
                                "method": f"replicate_{model['name']}",
                            }
                    break

                elif status in ["failed", "canceled"]:
                    st.warning(
                        f"Model failed: {poll_result.get('error', 'Unknown error')}"
                    )
                    break

            progress.empty()

        except Exception as e:
            st.warning(f"Replicate {model['name']} error: {e}")
            continue

    return {
        "error": "All Replicate models failed. Try again in a few minutes (rate limit)."
    }


def generate_with_fal(
    image_base64: str, prompt: str, negative_prompt: str, strength: float = 0.6
) -> dict:
    """
    Generate using Fal.ai with proper image URL handling
    """
    api_key = get_api_key("FAL_KEY")
    if not api_key:
        return {"error": "Fal.ai API key not configured"}

    headers = {"Authorization": f"Key {api_key}", "Content-Type": "application/json"}

    # First, upload the image to get a URL
    # Try tmpfiles as it's simpler
    st.info("Uploading image...")
    image_url = upload_to_tmpfiles(image_base64)

    if not image_url:
        return {"error": "Failed to upload image for Fal.ai"}

    st.info(f"Image uploaded, generating design...")

    # Use Fal.ai's SDXL img2img endpoint (more reliable than ControlNet)
    endpoints = [
        {
            "name": "fast-sdxl-img2img",
            "url": "https://fal.run/fal-ai/fast-sdxl",
            "payload": {
                "image_url": image_url,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "strength": strength,
                "num_inference_steps": 25,
                "guidance_scale": 7.5,
                "image_size": "landscape_16_9",
                "num_images": 1,
                "enable_safety_checker": False,
                "sync_mode": True,
            },
        },
        {
            "name": "lcm-img2img",
            "url": "https://fal.run/fal-ai/lcm",
            "payload": {
                "image_url": image_url,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "strength": strength,
                "num_inference_steps": 8,
                "guidance_scale": 2.0,
                "sync_mode": True,
            },
        },
    ]

    for endpoint in endpoints:
        try:
            st.info(f"Trying Fal.ai {endpoint['name']}...")

            response = requests.post(
                endpoint["url"], headers=headers, json=endpoint["payload"], timeout=120
            )

            if response.status_code == 200:
                data = response.json()
                images = data.get("images", [])

                if images:
                    img_url = images[0].get("url")
                    if img_url:
                        img_response = requests.get(img_url, timeout=30)
                        if img_response.status_code == 200:
                            return {
                                "success": True,
                                "image_base64": base64.b64encode(
                                    img_response.content
                                ).decode(),
                                "method": f"fal_{endpoint['name']}",
                            }
            else:
                st.warning(
                    f"Fal.ai {endpoint['name']}: {response.status_code} - {response.text[:200]}"
                )

        except Exception as e:
            st.warning(f"Fal.ai {endpoint['name']} error: {e}")
            continue

    return {"error": "All Fal.ai endpoints failed"}


def generate_design(
    image_base64: str, style: str, variation: str, structure_preservation: float = 0.7
) -> dict:
    """
    Main design generation function

    structure_preservation: 0.5-0.9
    - Higher = more of original image preserved
    - 0.7 = recommended balance
    """

    # Build prompts
    style_prompts = {
        "Modern Minimalist": "modern minimalist interior design, clean lines, neutral colors white beige grey, minimal furniture, natural light, sleek surfaces, professional photo",
        "Scandinavian": "scandinavian interior design, light oak wood, white walls, cozy textiles, plants, warm lighting, hygge atmosphere, professional photo",
        "Industrial": "industrial loft interior, exposed brick, metal fixtures, Edison bulbs, concrete, leather furniture, professional photo",
        "Mid-Century Modern": "mid-century modern interior, walnut wood, organic curves, 1960s furniture, mustard teal accents, professional photo",
        "Contemporary": "contemporary interior design, mixed materials, neutral with bold accents, comfortable furniture, professional photo",
        "Bohemian": "bohemian interior, eclectic colorful, layered textiles, plants, artistic collected feel, professional photo",
        "Coastal": "coastal interior, blue white palette, natural rattan, light airy beach house, professional photo",
        "Farmhouse": "modern farmhouse interior, shiplap walls, rustic wood, neutral cozy, professional photo",
    }

    variation_mods = {
        "Light & Airy": "bright airy, natural light, white cream colors, spacious",
        "Warm & Cozy": "warm cozy, warm lighting, earth tones, layered textures",
        "Bold & Dramatic": "bold dramatic, rich deep colors, high contrast, statement pieces",
    }

    prompt = f"{style_prompts.get(style, 'modern interior')}, {variation_mods.get(variation, '')}, high quality realistic interior photograph, detailed textures, magazine quality"

    negative_prompt = "blurry, low quality, distorted, cartoon, painting, sketch, people, text, watermark, bad proportions"

    # Convert preservation to strength (inverted)
    # Higher preservation = lower strength (less change)
    strength = max(0.3, min(0.8, 1.0 - structure_preservation + 0.2))

    st.write(f"🎨 Style: {style}, Variation: {variation}")
    st.write(
        f"🔧 Strength: {strength:.2f} (preservation: {structure_preservation:.2f})"
    )

    # Try Replicate first (more reliable for interior design)
    if get_api_key("REPLICATE_API_TOKEN"):
        result = generate_with_replicate(
            image_base64, prompt, negative_prompt, strength
        )
        if result.get("success"):
            return result
        st.warning(f"Replicate failed: {result.get('error')}")

    # Try Fal.ai second
    if get_api_key("FAL_KEY"):
        result = generate_with_fal(image_base64, prompt, negative_prompt, strength)
        if result.get("success"):
            return result
        st.warning(f"Fal.ai failed: {result.get('error')}")

    return {
        "error": "All generation methods failed. Please check your API keys and try again."
    }


# ============== PRODUCT SCRAPING ==============


class ProductScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            }
        )

    def search_serpapi(self, query: str, api_key: str, max_results: int = 5) -> list:
        try:
            response = self.session.get(
                "https://serpapi.com/search",
                params={
                    "engine": "google_shopping",
                    "q": query,
                    "api_key": api_key,
                    "num": max_results,
                    "hl": "en",
                    "gl": "us",
                },
                timeout=15,
            )
            data = response.json()
            if "error" in data:
                return []

            return [
                {
                    "product_name": item.get("title", "Unknown")[:80],
                    "price": item.get("extracted_price", 0),
                    "price_str": item.get("price", "See site"),
                    "retailer": item.get("source", "Unknown"),
                    "url": item.get("link", ""),
                    "image_url": item.get("thumbnail", ""),
                    "is_direct_link": True,
                    "source": "serpapi",
                }
                for item in data.get("shopping_results", [])[:max_results]
            ]
        except:
            return []

    def generate_retailer_urls(self, query: str) -> list:
        encoded = quote_plus(query)
        retailers = [
            ("Amazon", f"https://www.amazon.com/s?k={encoded}", "🛒"),
            (
                "Wayfair",
                f"https://www.wayfair.com/keyword.html?keyword={encoded}",
                "🏠",
            ),
            ("IKEA", f"https://www.ikea.com/us/en/search/?q={encoded}", "🪑"),
            ("West Elm", f"https://www.westelm.com/search/?query={encoded}", "✨"),
            ("Target", f"https://www.target.com/s?searchTerm={encoded}", "🎯"),
        ]
        return [
            {
                "product_name": f"{logo} Search on {name}",
                "url": url,
                "retailer": name,
                "price_str": "Browse",
                "is_direct_link": False,
                "source": "retailer",
            }
            for name, url, logo in retailers
        ]

    def search(self, query: str) -> tuple:
        serpapi_key = get_api_key("SERPAPI_KEY")
        if serpapi_key:
            products = self.search_serpapi(query, serpapi_key)
            if products:
                return products, "serpapi", True
        return self.generate_retailer_urls(query), "retailer", False


scraper = ProductScraper()


def find_products_for_item(item: dict) -> tuple:
    keywords = item.get("search_keywords", [])
    query = keywords[0] if keywords else item.get("name", "furniture")
    return scraper.search(query)


# ============== IMAGE PROCESSING ==============


def process_image(f):
    f.seek(0)
    try:
        img = Image.open(f)
        if img.mode in ("RGBA", "P", "LA"):
            img = img.convert("RGB")
        max_size = 1024
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            img = img.resize(
                (int(img.size[0] * ratio), int(img.size[1] * ratio)), Image.LANCZOS
            )
        buf = BytesIO()
        img.save(buf, "JPEG", quality=85)
        return {
            "name": f.name,
            "data": base64.b64encode(buf.getvalue()).decode(),
            "success": True,
        }
    except Exception as e:
        return {"error": str(e), "success": False}


# ============== CLAUDE AGENTS ==============


def get_claude():
    key = get_api_key("ANTHROPIC_API_KEY")
    return anthropic.Anthropic(api_key=key) if key else None


def run_valuation(images: list, prefs: dict) -> dict:
    client = get_claude()
    if not client:
        return {"error": "Claude API key not configured"}

    content = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": img["data"],
            },
        }
        for img in images
        if img.get("success")
    ]
    content.append(
        {
            "type": "text",
            "text": f"""Analyze this room. Style: {prefs.get('style')}.
Return JSON: {{"property_assessment": {{"room_type": "type", "current_condition": "fair", "square_footage_estimate": "X sq ft"}}, "cost_estimate": {{"low": 10000, "mid": 20000, "high": 35000}}}}""",
        }
    )

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[{"role": "user", "content": content}],
        )
        match = re.search(r"\{[\s\S]*\}", response.content[0].text)
        return json.loads(match.group()) if match else {"error": "Parse failed"}
    except Exception as e:
        return {"error": str(e)}


def run_design_concepts(valuation: dict, prefs: dict) -> dict:
    client = get_claude()
    if not client:
        return {"error": "Claude API key not configured"}

    room = valuation.get("property_assessment", {}).get("room_type", "room")
    style = prefs.get("style", "Modern")

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": f"""Create 3 {style} design variations for a {room}.
Return JSON: {{"design_options": [
    {{"option_number": 1, "name": "{style} - Light & Airy", "variation": "Light & Airy", "concept": "Bright design", "color_palette": {{"primary": "#F5F5F5", "secondary": "#E8E8E8", "accent": "#4A90A4"}}, "key_furniture": ["grey sofa", "white table", "lamp"], "estimated_cost": 15000}},
    {{"option_number": 2, "name": "{style} - Warm & Cozy", "variation": "Warm & Cozy", "concept": "Warm design", "color_palette": {{"primary": "#F5E6D3", "secondary": "#D4A574", "accent": "#8B4513"}}, "key_furniture": ["leather sofa", "wood table"], "estimated_cost": 18000}},
    {{"option_number": 3, "name": "{style} - Bold & Dramatic", "variation": "Bold & Dramatic", "concept": "Bold design", "color_palette": {{"primary": "#2C3E50", "secondary": "#34495E", "accent": "#C0392B"}}, "key_furniture": ["velvet sofa", "marble table"], "estimated_cost": 25000}}
]}}""",
                }
            ],
        )
        match = re.search(r"\{[\s\S]*\}", response.content[0].text)
        return json.loads(match.group()) if match else {"error": "Parse failed"}
    except Exception as e:
        return {"error": str(e)}


def segment_furniture(image_base64: str) -> list:
    client = get_claude()
    if not client:
        return []

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[
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
                        {
                            "type": "text",
                            "text": """Identify ALL furniture in this image. Return JSON array:
[{"id": 1, "name": "Grey Sofa", "search_keywords": ["grey modern sofa"], "price_estimate_low": 800, "price_estimate_high": 2000}]""",
                        },
                    ],
                }
            ],
        )
        match = re.search(r"\[[\s\S]*\]", response.content[0].text)
        return json.loads(match.group()) if match else []
    except:
        return []


# ============== UI ==============


def render_header():
    st.title("🏠 OmniRenovation AI")
    st.caption("AI Interior Design with Real Product Links")

    phases = ["Upload", "Analysis", "Design", "Products", "BOM", "Done"]
    current = {
        "upload": 0,
        "valuation": 1,
        "design": 2,
        "products": 3,
        "bom": 4,
        "complete": 5,
    }.get(st.session_state.project_state["phase"], 0)

    cols = st.columns(6)
    for i, p in enumerate(phases):
        with cols[i]:
            if i < current:
                st.success(f"✓ {p}")
            elif i == current:
                st.info(f"→ {p}")
            else:
                st.text(p)


def render_upload():
    st.header("📤 Upload Your Room")

    with st.expander("🔑 API Configuration", expanded=True):
        st.markdown("### Required")
        key = st.text_input(
            "Anthropic API Key *",
            type="password",
            value=st.session_state.get("anthropic_api_key", ""),
        )
        if key:
            st.session_state.anthropic_api_key = key

        st.markdown("### Design Generation")
        st.info(
            "**Replicate** is recommended - more reliable for interior design. Get free credits at replicate.com"
        )

        col1, col2 = st.columns(2)
        with col1:
            k1 = st.text_input(
                "Replicate Token (Recommended)",
                type="password",
                value=st.session_state.get("replicate_api_token", ""),
            )
            if k1:
                st.session_state.replicate_api_token = k1
        with col2:
            k2 = st.text_input(
                "Fal.ai Key (Backup)",
                type="password",
                value=st.session_state.get("fal_key", ""),
            )
            if k2:
                st.session_state.fal_key = k2

        st.markdown("### Product Search")
        k3 = st.text_input(
            "SerpAPI Key (100 free/month)",
            type="password",
            value=st.session_state.get("serpapi_key", ""),
        )
        if k3:
            st.session_state.serpapi_key = k3

        # Status
        st.markdown("### 📊 Status")
        cols = st.columns(4)
        (
            cols[0].success("✅ Claude")
            if st.session_state.get("anthropic_api_key")
            else cols[0].error("❌ Claude")
        )
        (
            cols[1].success("✅ Replicate")
            if st.session_state.get("replicate_api_token")
            else cols[1].warning("⚪ Replicate")
        )
        (
            cols[2].success("✅ Fal.ai")
            if st.session_state.get("fal_key")
            else cols[2].warning("⚪ Fal.ai")
        )
        (
            cols[3].success("✅ SerpAPI")
            if st.session_state.get("serpapi_key")
            else cols[3].warning("⚪ SerpAPI")
        )

        has_design = st.session_state.get(
            "replicate_api_token"
        ) or st.session_state.get("fal_key")
        if has_design:
            st.success("✅ Design generation ready!")
        else:
            st.error("❌ Add Replicate or Fal.ai key for design generation")

    st.divider()

    files = st.file_uploader(
        "📷 Upload room photos",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
    )
    if files:
        cols = st.columns(min(4, len(files)))
        for i, f in enumerate(files):
            with cols[i % 4]:
                f.seek(0)
                st.image(Image.open(f), use_column_width=True)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        budget = st.select_slider(
            "Budget", ["<$5K", "$5K-15K", "$15K-30K", "$30K-50K", ">$50K"], "$15K-30K"
        )
        style = st.selectbox(
            "Style",
            [
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
        room = st.selectbox(
            "Room Type",
            ["Living Room", "Bedroom", "Kitchen", "Bathroom", "Home Office"],
        )
        preservation = st.slider(
            "Structure Preservation",
            0.5,
            0.9,
            0.7,
            0.05,
            help="Higher = more original structure kept (0.7 recommended)",
        )

    st.session_state["structure_preservation"] = preservation

    st.divider()

    can_start = bool(st.session_state.get("anthropic_api_key")) and bool(files)
    if st.button("🚀 Analyze Room", type="primary", disabled=not can_start):
        processed = [process_image(f) for f in files]
        st.session_state.project_state.update(
            {
                "images": [p for p in processed if p.get("success")],
                "preferences": {"budget": budget, "style": style, "room_type": room},
                "phase": "valuation",
            }
        )
        st.rerun()


def render_valuation():
    st.header("📊 Room Analysis")

    if not st.session_state.project_state["valuation"]:
        with st.spinner("🔍 Analyzing..."):
            st.session_state.project_state["valuation"] = run_valuation(
                st.session_state.project_state["images"],
                st.session_state.project_state["preferences"],
            )
            st.rerun()

    val = st.session_state.project_state["valuation"]
    if "error" in val:
        st.error(val["error"])
        if st.button("🔄 Retry"):
            st.session_state.project_state["valuation"] = None
            st.rerun()
        return

    a = val.get("property_assessment", {})
    c = val.get("cost_estimate", {})

    cols = st.columns(3)
    cols[0].metric("Room", a.get("room_type", "N/A"))
    cols[1].metric("Condition", a.get("current_condition", "N/A"))
    cols[2].metric("Est. Cost", f"${c.get('mid', 0):,}")

    if st.session_state.project_state["images"]:
        st.image(
            base64.b64decode(st.session_state.project_state["images"][0]["data"]),
            caption="Your Room",
            use_column_width=True,
        )

    st.divider()
    col1, col2 = st.columns(2)
    if col1.button("🎨 Generate Designs", type="primary"):
        st.session_state.project_state["phase"] = "design"
        st.rerun()
    if col2.button("← Start Over"):
        st.session_state.project_state = DEFAULT_STATE.copy()
        st.rerun()


def render_design():
    st.header("🎨 Design Options")

    has_api = st.session_state.get("replicate_api_token") or st.session_state.get(
        "fal_key"
    )

    if has_api:
        st.success("✅ Design generation ready!")
    else:
        st.error("❌ Add Replicate or Fal.ai API key")

    if not st.session_state.project_state["designs"]:
        with st.spinner("Creating concepts..."):
            st.session_state.project_state["designs"] = run_design_concepts(
                st.session_state.project_state["valuation"],
                st.session_state.project_state["preferences"],
            )
            st.rerun()

    designs = st.session_state.project_state["designs"]
    if "error" in designs:
        st.error(designs["error"])
        if st.button("Retry"):
            st.session_state.project_state["designs"] = None
            st.rerun()
        return

    design_images = st.session_state.project_state.get("design_images", {})
    original = (
        st.session_state.project_state["images"][0]["data"]
        if st.session_state.project_state["images"]
        else None
    )

    for opt in designs.get("design_options", []):
        num = opt["option_number"]
        st.subheader(f"Option {num}: {opt.get('name')}")

        col1, col2 = st.columns([2, 1])

        with col1:
            if num in design_images and design_images[num].get("success"):
                st.image(
                    base64.b64decode(design_images[num]["image_base64"]),
                    caption=f"Generated ({design_images[num].get('method')})",
                    use_column_width=True,
                )

                with st.expander("📊 Compare"):
                    c1, c2 = st.columns(2)
                    if original:
                        c1.image(base64.b64decode(original), caption="Before")
                    c2.image(
                        base64.b64decode(design_images[num]["image_base64"]),
                        caption="After",
                    )

            elif num in design_images and design_images[num].get("error"):
                st.error(f"Failed: {design_images[num]['error']}")
                if st.button(f"🔄 Retry", key=f"retry_{num}"):
                    del design_images[num]
                    st.session_state.project_state["design_images"] = design_images
                    st.rerun()

            else:
                pal = opt.get("color_palette", {})
                st.write("**Colors:**")
                pc = st.columns(3)
                for i, (k, v) in enumerate(list(pal.items())[:3]):
                    pc[i].color_picker(k, v, disabled=True, key=f"c{num}{i}")

                if has_api and original:
                    if st.button(f"🖼️ Generate Design", key=f"gen{num}"):
                        result = generate_design(
                            original,
                            st.session_state.project_state["preferences"]["style"],
                            opt.get("variation", "Light & Airy"),
                            st.session_state.get("structure_preservation", 0.7),
                        )
                        design_images[num] = result
                        st.session_state.project_state["design_images"] = design_images
                        st.rerun()

        with col2:
            st.write(f"**Concept:** {opt.get('concept')}")
            st.metric("Cost", f"${opt.get('estimated_cost', 0):,}")
            st.write("**Furniture:**")
            for item in opt.get("key_furniture", [])[:4]:
                st.write(f"• {item}")

            if st.button(f"✅ Select", key=f"sel{num}", type="primary"):
                st.session_state.project_state["selected_design"] = opt
                if num in design_images and design_images[num].get("success"):
                    st.session_state.project_state["selected_design_image"] = (
                        design_images[num]["image_base64"]
                    )
                elif original:
                    st.session_state.project_state["selected_design_image"] = original
                st.success("✓ Selected!")

        st.divider()

    if st.session_state.project_state.get("selected_design"):
        st.success(
            f"Selected: **{st.session_state.project_state['selected_design'].get('name')}**"
        )
        if st.button("🛋️ Find Products", type="primary"):
            st.session_state.project_state["phase"] = "products"
            st.rerun()

    if st.button("← Back"):
        st.session_state.project_state["phase"] = "valuation"
        st.rerun()


def render_products():
    st.header("🛋️ Product Search")

    has_serpapi = bool(st.session_state.get("serpapi_key"))
    if has_serpapi:
        st.success("✅ SerpAPI - Direct product links!")
    else:
        st.warning("🔗 Using retailer search links")

    img = st.session_state.project_state.get("selected_design_image") or (
        st.session_state.project_state["images"][0]["data"]
        if st.session_state.project_state["images"]
        else None
    )

    if not st.session_state.project_state.get("furniture_items"):
        with st.spinner("Identifying furniture..."):
            furniture = segment_furniture(img) if img else []
            if not furniture:
                design = st.session_state.project_state.get("selected_design", {})
                furniture = [
                    {"id": i + 1, "name": item, "search_keywords": [item]}
                    for i, item in enumerate(
                        design.get("key_furniture", ["sofa", "table"])
                    )
                ]
            st.session_state.project_state["furniture_items"] = furniture
            st.rerun()

    furniture = st.session_state.project_state["furniture_items"]
    product_matches = st.session_state.project_state.get("product_matches", {})

    if img:
        with st.expander("📷 Design", expanded=False):
            st.image(base64.b64decode(img), use_column_width=True)

    st.write(f"**{len(furniture)} items** to find")
    st.divider()

    for item in furniture:
        item_id = str(item.get("id", 0))

        with st.expander(f"🔍 **{item.get('name')}**", expanded=True):
            if item_id not in product_matches:
                with st.spinner("Searching..."):
                    products, source, has_direct = find_products_for_item(item)
                    product_matches[item_id] = {
                        "products": products,
                        "source": source,
                        "has_direct": has_direct,
                    }
                    st.session_state.project_state["product_matches"] = product_matches

            data = product_matches.get(item_id, {})
            products = data.get("products", [])

            if products:
                cols = st.columns(min(4, len(products)))
                for i, p in enumerate(products[:4]):
                    with cols[i]:
                        if p.get("image_url"):
                            try:
                                st.image(p["image_url"], width=120)
                            except:
                                pass
                        st.write(f"**{p.get('product_name', '')[:35]}**")
                        st.write(f"🏪 {p.get('retailer', 'N/A')}")
                        st.write(f"💰 **{p.get('price_str', 'See site')}**")

                        url = p.get("url", "")
                        if url:
                            icon = "🛒" if p.get("is_direct_link") else "🔍"
                            st.markdown(f"### [{icon} View]({url})")

                        if st.button("✓", key=f"p{item_id}_{i}"):
                            st.session_state.project_state["selected_products"][
                                item_id
                            ] = p
                            st.success("✓")

    st.divider()
    col1, col2 = st.columns(2)
    if col1.button("📦 Generate BOM", type="primary"):
        st.session_state.project_state["phase"] = "bom"
        st.rerun()
    if col2.button("← Back"):
        st.session_state.project_state["phase"] = "design"
        st.rerun()


def render_bom():
    st.header("📦 Bill of Materials")

    furniture = st.session_state.project_state.get("furniture_items", [])
    product_matches = st.session_state.project_state.get("product_matches", {})
    selected = st.session_state.project_state.get("selected_products", {})

    items = []
    total = 0

    for f in furniture:
        fid = str(f.get("id", 0))
        p = selected.get(fid) or (
            product_matches.get(fid, {}).get("products", [{}])[0]
            if product_matches.get(fid, {}).get("products")
            else {}
        )
        price = p.get("price", 0) if isinstance(p.get("price"), (int, float)) else 0
        items.append(
            {
                "item": f.get("name", "Item"),
                "product": p.get("product_name", "Not found"),
                "retailer": p.get("retailer", "-"),
                "price": price,
                "url": p.get("url", ""),
                "is_direct": p.get("is_direct_link", False),
            }
        )
        total += price

    if total == 0:
        for f in furniture:
            total += (
                f.get("price_estimate_low", 500) + f.get("price_estimate_high", 1500)
            ) / 2

    labor, contingency = 1880, (total + 1880) * 0.1
    grand = total + labor + contingency

    cols = st.columns(3)
    cols[0].metric("Products", f"${total:,.0f}")
    cols[1].metric("Labor", f"${labor:,}")
    cols[2].metric("**Total**", f"${grand:,.0f}")

    st.divider()
    for item in items:
        c1, c2, c3 = st.columns([3, 1, 1])
        c1.write(f"**{item['item']}**")
        c1.caption(f"{item['product']} ({item['retailer']})")
        c2.write(f"${item['price']:,.0f}" if item["price"] else "—")
        if item["url"]:
            c3.markdown(f"[{'🛒' if item['is_direct'] else '🔍'} Link]({item['url']})")

    st.divider()
    col1, col2 = st.columns(2)
    if col1.button("✅ Complete", type="primary"):
        st.session_state.project_state["bom"] = {"items": items, "total": grand}
        st.session_state.project_state["phase"] = "complete"
        st.rerun()
    if col2.button("← Back"):
        st.session_state.project_state["phase"] = "products"
        st.rerun()


def render_complete():
    st.header("✅ Project Complete!")
    st.balloons()

    design = st.session_state.project_state.get("selected_design", {})
    bom = st.session_state.project_state.get("bom", {})

    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.project_state.get("selected_design_image"):
            st.image(
                base64.b64decode(
                    st.session_state.project_state["selected_design_image"]
                ),
                use_column_width=True,
            )
        st.write(f"**{design.get('name')}**")
    with col2:
        st.metric("Total", f"${bom.get('total', 0):,.0f}")

    st.divider()
    st.subheader("🔗 Shopping Links")
    for item in bom.get("items", []):
        if item.get("url"):
            st.markdown(
                f"- [{'🛒' if item['is_direct'] else '🔍'} {item['item']} - {item['retailer']}]({item['url']})"
            )

    st.divider()
    st.download_button(
        "📥 Download",
        json.dumps({"design": design, "bom": bom}, indent=2),
        "renovation.json",
    )

    if st.button("🔄 New Project"):
        st.session_state.project_state = DEFAULT_STATE.copy()
        st.rerun()


def main():
    render_header()
    st.divider()
    {
        "upload": render_upload,
        "valuation": render_valuation,
        "design": render_design,
        "products": render_products,
        "bom": render_bom,
        "complete": render_complete,
    }.get(st.session_state.project_state["phase"], render_upload)()


if __name__ == "__main__":
    main()
