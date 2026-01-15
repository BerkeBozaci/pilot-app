"""
OmniRenovation AI - Phase 1 Pilot v16
=====================================
- Google Gemini 2.0 Flash for image generation (ONLY)
- Open-source product search (retailer links you can customize)
- Proper API key validation
- No backup models - Gemini only
"""

import streamlit as st
import requests
import json
import base64
import re
import io
from PIL import Image
from urllib.parse import quote_plus

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
    }
    session_key = key_map.get(name, name.lower())
    return st.session_state.get(session_key, "").strip()


def validate_gemini_key(api_key: str) -> tuple:
    """Validate Gemini API key by making a test request"""
    if not api_key:
        return False, "API key is empty"

    try:
        # Test the key with a simple request
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
    """Validate Anthropic API key"""
    if not api_key:
        return False, "API key is empty"

    if not api_key.startswith("sk-ant-"):
        return False, "Invalid format (should start with sk-ant-)"

    return True, "Format valid"


# ============== GEMINI IMAGE GENERATION ==============


def generate_design_with_gemini(
    image_base64: str, style: str, variation: str, room_type: str = "room"
) -> dict:
    """
    Generate interior design using Google Gemini 2.0 Flash

    Uses the official Gemini API with image input and output
    """
    api_key = get_api_key("GEMINI_API_KEY")

    if not api_key:
        return {
            "error": "Gemini API key not configured. Please add your API key in the settings."
        }

    # Build the design prompt
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

    try:
        # Gemini API endpoint for image generation
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={api_key}"

        headers = {"Content-Type": "application/json"}

        # Request body with image input and requesting image output
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": image_base64,
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"],
                "temperature": 0.8,
                "topP": 0.95,
                "topK": 40,
            },
        }

        st.info("🎨 Generating design with Gemini 2.0 Flash...")

        response = requests.post(url, headers=headers, json=payload, timeout=120)

        if response.status_code == 200:
            data = response.json()

            # Extract image from response
            if "candidates" in data and len(data["candidates"]) > 0:
                parts = data["candidates"][0].get("content", {}).get("parts", [])

                for part in parts:
                    # Check for inline image data
                    if "inlineData" in part:
                        image_data = part["inlineData"].get("data", "")
                        if image_data:
                            return {
                                "success": True,
                                "image_base64": image_data,
                                "method": "gemini-2.0-flash",
                            }

                # If no image in response, return the text response for debugging
                text_parts = [p.get("text", "") for p in parts if "text" in p]
                if text_parts:
                    return {
                        "error": f"Gemini returned text instead of image: {text_parts[0][:200]}"
                    }

            return {"error": "No image generated in response"}

        elif response.status_code == 400:
            error_msg = response.json().get("error", {}).get("message", "Bad request")
            return {"error": f"API Error: {error_msg}"}
        elif response.status_code == 403:
            return {
                "error": "API key invalid or image generation not enabled for your account"
            }
        elif response.status_code == 429:
            return {"error": "Rate limited. Please wait a moment and try again."}
        else:
            return {"error": f"API Error {response.status_code}: {response.text[:200]}"}

    except requests.exceptions.Timeout:
        return {"error": "Request timed out. Please try again."}
    except Exception as e:
        return {"error": f"Generation failed: {str(e)}"}


# ============== CLAUDE FOR ANALYSIS ==============


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
            # Extract JSON from response
            match = re.search(r"\{[\s\S]*\}", text)
            if match:
                return json.loads(match.group())
            return {"error": "Could not parse response"}
        else:
            return {"error": f"API Error: {response.status_code}"}

    except Exception as e:
        return {"error": str(e)}


def create_design_concepts(valuation: dict, preferences: dict) -> dict:
    """Generate design concept descriptions using Claude"""
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


def identify_furniture(image_base64: str) -> list:
    """Use Claude to identify furniture in the design image"""
    api_key = get_api_key("ANTHROPIC_API_KEY")

    if not api_key:
        return []

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }

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
                                "text": """Identify ALL furniture and decor items in this interior design image.

Return ONLY a JSON array (no markdown):
[
    {"id": 1, "name": "Grey Linen Sofa", "search_query": "modern grey linen sofa", "price_low": 800, "price_high": 2000},
    {"id": 2, "name": "Oak Coffee Table", "search_query": "white oak coffee table modern", "price_low": 300, "price_high": 800}
]

Include: sofas, chairs, tables, lamps, rugs, artwork, plants, shelving, curtains, decor items.""",
                            },
                        ],
                    }
                ],
            },
            timeout=60,
        )

        if response.status_code == 200:
            data = response.json()
            text = data["content"][0]["text"]
            match = re.search(r"\[[\s\S]*\]", text)
            if match:
                return json.loads(match.group())
        return []

    except:
        return []


# ============== OPEN SOURCE PRODUCT SEARCH ==============


class OpenSourceProductSearch:
    """
    Open-source product search using retailer search URLs

    You can customize these retailers by editing the RETAILERS list below.
    Each retailer has:
    - name: Display name
    - search_url: URL template with {query} placeholder
    - icon: Emoji icon
    """

    # ========================================
    # CUSTOMIZE YOUR RETAILERS HERE
    # ========================================
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
    # ========================================

    @classmethod
    def search(cls, query: str) -> list:
        """Generate search links for all configured retailers"""
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
        """Get list of configured retailers for display"""
        return [r["name"] for r in cls.RETAILERS]


def find_products_for_item(item: dict) -> list:
    """Find products for a furniture item using open-source search"""
    query = item.get("search_query", item.get("name", "furniture"))
    return OpenSourceProductSearch.search(query)


# ============== IMAGE PROCESSING ==============


def process_uploaded_image(uploaded_file) -> dict:
    """Process uploaded image file"""
    try:
        uploaded_file.seek(0)
        img = Image.open(uploaded_file)

        # Convert to RGB if needed
        if img.mode in ("RGBA", "P", "LA"):
            img = img.convert("RGB")

        # Resize if too large (max 1024px)
        max_size = 1024
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.LANCZOS)

        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        base64_data = base64.b64encode(buffer.getvalue()).decode()

        return {"success": True, "data": base64_data, "name": uploaded_file.name}

    except Exception as e:
        return {"success": False, "error": str(e)}


# ============== UI COMPONENTS ==============


def render_header():
    """Render the page header with progress"""
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
    """Render the upload and configuration phase"""
    st.header("📤 Upload Your Room")

    # API Configuration Section
    with st.expander("🔑 API Configuration", expanded=True):
        st.markdown("### Required API Keys")
        st.info(
            """
        **You need two API keys:**
        1. **Google Gemini** - For AI image generation (get free at [aistudio.google.com](https://aistudio.google.com))
        2. **Anthropic Claude** - For room analysis (get at [console.anthropic.com](https://console.anthropic.com))
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

        # Validate and show status
        st.markdown("### 📊 API Status")

        col1, col2 = st.columns(2)

        with col1:
            if st.session_state.get("gemini_api_key"):
                is_valid, msg = validate_gemini_key(
                    st.session_state.get("gemini_api_key", "")
                )
                if is_valid:
                    st.success("✅ Gemini API Key: Valid")
                else:
                    st.error(f"❌ Gemini API Key: {msg}")
            else:
                st.warning("⚠️ Gemini API Key: Not entered")

        with col2:
            if st.session_state.get("anthropic_api_key"):
                is_valid, msg = validate_anthropic_key(
                    st.session_state.get("anthropic_api_key", "")
                )
                if is_valid:
                    st.success("✅ Claude API Key: Format valid")
                else:
                    st.error(f"❌ Claude API Key: {msg}")
            else:
                st.warning("⚠️ Claude API Key: Not entered")

        # Check if ready
        gemini_ready = bool(st.session_state.get("gemini_api_key"))
        claude_ready = bool(st.session_state.get("anthropic_api_key"))

        if gemini_ready and claude_ready:
            st.success("✅ All API keys configured! You're ready to start.")
        else:
            missing = []
            if not gemini_ready:
                missing.append("Gemini")
            if not claude_ready:
                missing.append("Claude")
            st.error(f"❌ Missing API keys: {', '.join(missing)}")

    st.divider()

    # File Upload Section
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

    # Preferences Section
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

    # Start Button
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
        # Process images
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
                }
            )
            st.rerun()
        else:
            st.error("Failed to process uploaded images. Please try again.")


def render_valuation_phase():
    """Render the room analysis phase"""
    st.header("📊 Room Analysis")

    # Run analysis if not done
    if not st.session_state.project_state["valuation"]:
        with st.spinner("🔍 Analyzing your room with Claude..."):
            result = analyze_room_with_claude(
                st.session_state.project_state["images"],
                st.session_state.project_state["preferences"],
            )
            st.session_state.project_state["valuation"] = result
            st.rerun()

    valuation = st.session_state.project_state["valuation"]

    # Check for errors
    if "error" in valuation:
        st.error(f"Analysis failed: {valuation['error']}")
        if st.button("🔄 Retry Analysis"):
            st.session_state.project_state["valuation"] = None
            st.rerun()
        return

    # Display results
    assessment = valuation.get("property_assessment", {})
    costs = valuation.get("cost_estimate", {})

    col1, col2, col3 = st.columns(3)
    col1.metric("Room Type", assessment.get("room_type", "N/A"))
    col2.metric("Condition", assessment.get("current_condition", "N/A").title())
    col3.metric("Est. Renovation Cost", f"${costs.get('mid', 0):,}")

    # Show original image
    if st.session_state.project_state["images"]:
        st.image(
            base64.b64decode(st.session_state.project_state["images"][0]["data"]),
            caption="Your Room",
            use_column_width=True,
        )

    # Show features
    features = assessment.get("notable_features", [])
    if features:
        st.info(f"**Notable Features:** {', '.join(features)}")

    st.divider()

    # Navigation
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
    """Render the design generation phase"""
    st.header("🎨 Design Options")

    # Generate design concepts if not done
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

    # Get original image and design images
    original_image = (
        st.session_state.project_state["images"][0]["data"]
        if st.session_state.project_state["images"]
        else None
    )
    design_images = st.session_state.project_state.get("design_images", {})
    room_type = st.session_state.project_state["preferences"].get("room_type", "room")

    # Render each design option
    for option in designs.get("design_options", []):
        num = option["option_number"]

        st.subheader(f"Option {num}: {option.get('name', 'Design')}")

        col1, col2 = st.columns([2, 1])

        with col1:
            # Check if we have a generated image for this option
            if num in design_images:
                if design_images[num].get("success"):
                    # Show generated image
                    st.image(
                        base64.b64decode(design_images[num]["image_base64"]),
                        caption=f"AI Generated Design",
                        use_column_width=True,
                    )

                    # Show comparison
                    with st.expander("📊 Compare Before/After"):
                        c1, c2 = st.columns(2)
                        if original_image:
                            c1.image(base64.b64decode(original_image), caption="BEFORE")
                        c2.image(
                            base64.b64decode(design_images[num]["image_base64"]),
                            caption="AFTER",
                        )
                else:
                    # Show error
                    st.error(
                        f"Generation failed: {design_images[num].get('error', 'Unknown error')}"
                    )
                    if st.button(f"🔄 Retry Generation", key=f"retry_{num}"):
                        del design_images[num]
                        st.session_state.project_state["design_images"] = design_images
                        st.rerun()
            else:
                # Show color palette preview
                palette = option.get("color_palette", {})
                st.markdown("**Color Palette:**")
                palette_cols = st.columns(3)
                for i, (name, color) in enumerate(list(palette.items())[:3]):
                    with palette_cols[i]:
                        st.color_picker(
                            name.title(), color, disabled=True, key=f"color_{num}_{i}"
                        )

                # Generate button
                if original_image:
                    if st.button(
                        f"🖼️ Generate This Design with Gemini", key=f"gen_{num}"
                    ):
                        with st.spinner(
                            "Generating design with Gemini 2.0 Flash... This may take 30-60 seconds."
                        ):
                            result = generate_design_with_gemini(
                                original_image,
                                st.session_state.project_state["preferences"]["style"],
                                option.get("variation", "Light & Airy"),
                                room_type,
                            )
                            design_images[num] = result
                            st.session_state.project_state["design_images"] = (
                                design_images
                            )
                            st.rerun()

        with col2:
            st.markdown(f"**Concept:** {option.get('concept', 'N/A')}")
            st.metric("Estimated Cost", f"${option.get('estimated_cost', 0):,}")

            st.markdown("**Key Furniture:**")
            for item in option.get("key_furniture", [])[:5]:
                st.write(f"• {item}")

            # Select button
            if st.button(f"✅ Select This Design", key=f"select_{num}", type="primary"):
                st.session_state.project_state["selected_design"] = option
                # Use generated image if available, otherwise original
                if num in design_images and design_images[num].get("success"):
                    st.session_state.project_state["selected_design_image"] = (
                        design_images[num]["image_base64"]
                    )
                else:
                    st.session_state.project_state["selected_design_image"] = (
                        original_image
                    )
                st.success("✓ Design selected!")

        st.divider()

    # Show selected design and next button
    if st.session_state.project_state.get("selected_design"):
        st.success(
            f"**Selected:** {st.session_state.project_state['selected_design'].get('name')}"
        )

        if st.button("🛋️ Find Products", type="primary"):
            st.session_state.project_state["phase"] = "products"
            st.rerun()

    # Back button
    if st.button("← Back to Analysis"):
        st.session_state.project_state["phase"] = "valuation"
        st.rerun()


def render_products_phase():
    """Render the product search phase"""
    st.header("🛋️ Product Search")

    st.info(
        f"**Searching across:** {', '.join(OpenSourceProductSearch.get_retailer_list())}"
    )

    # Get the design image
    design_image = st.session_state.project_state.get("selected_design_image")

    # Identify furniture if not done
    if not st.session_state.project_state.get("furniture_items"):
        with st.spinner("Identifying furniture in design..."):
            if design_image:
                furniture = identify_furniture(design_image)
            else:
                furniture = []

            # Fallback to design concept furniture
            if not furniture:
                design = st.session_state.project_state.get("selected_design", {})
                furniture = [
                    {"id": i + 1, "name": item, "search_query": item}
                    for i, item in enumerate(
                        design.get("key_furniture", ["sofa", "table", "lamp"])
                    )
                ]

            st.session_state.project_state["furniture_items"] = furniture
            st.rerun()

    furniture_items = st.session_state.project_state["furniture_items"]
    product_matches = st.session_state.project_state.get("product_matches", {})

    # Show design image
    if design_image:
        with st.expander("📷 Selected Design", expanded=False):
            st.image(base64.b64decode(design_image), use_column_width=True)

    st.write(f"**Found {len(furniture_items)} items** to search for")
    st.divider()

    # Search for each item
    for item in furniture_items:
        item_id = str(item.get("id", 0))

        with st.expander(f"🔍 **{item.get('name')}**", expanded=True):
            # Get products if not cached
            if item_id not in product_matches:
                products = find_products_for_item(item)
                product_matches[item_id] = products
                st.session_state.project_state["product_matches"] = product_matches

            products = product_matches.get(item_id, [])

            if products:
                # Display as columns
                cols = st.columns(min(4, len(products)))
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

    st.divider()

    # Navigation
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
    """Render the Bill of Materials phase"""
    st.header("📦 Bill of Materials")

    furniture = st.session_state.project_state.get("furniture_items", [])
    product_matches = st.session_state.project_state.get("product_matches", {})
    selected_products = st.session_state.project_state.get("selected_products", {})

    # Build BOM
    bom_items = []
    total_estimate = 0

    for item in furniture:
        item_id = str(item.get("id", 0))

        # Get selected product or first available
        product = selected_products.get(item_id)
        if not product:
            products = product_matches.get(item_id, [])
            product = products[0] if products else {}

        # Estimate price from item data
        price_low = item.get("price_low", 200)
        price_high = item.get("price_high", 800)
        price_est = (price_low + price_high) / 2

        bom_items.append(
            {
                "name": item.get("name", "Item"),
                "retailer": product.get("retailer", "Various"),
                "url": product.get("url", ""),
                "price_estimate": price_est,
            }
        )

        total_estimate += price_est

    # Add labor estimate
    labor = 1500
    contingency = (total_estimate + labor) * 0.1
    grand_total = total_estimate + labor + contingency

    # Display summary
    col1, col2, col3 = st.columns(3)
    col1.metric("Products", f"${total_estimate:,.0f}")
    col2.metric("Labor (est.)", f"${labor:,}")
    col3.metric("**Total**", f"${grand_total:,.0f}")

    st.divider()

    # Item list
    st.markdown("### Items")

    for item in bom_items:
        col1, col2, col3 = st.columns([3, 1, 1])

        with col1:
            st.write(f"**{item['name']}**")
            st.caption(f"Retailer: {item['retailer']}")

        with col2:
            st.write(f"~${item['price_estimate']:,.0f}")

        with col3:
            if item.get("url"):
                st.markdown(f"[🔗 Shop]({item['url']})")

    st.divider()

    # Save BOM data
    st.session_state.project_state["bom"] = {
        "items": bom_items,
        "products_total": total_estimate,
        "labor": labor,
        "contingency": contingency,
        "grand_total": grand_total,
    }

    # Navigation
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
    """Render the project completion phase"""
    st.header("✅ Project Complete!")
    st.balloons()

    # Get data
    design = st.session_state.project_state.get("selected_design", {})
    bom = st.session_state.project_state.get("bom", {})
    design_image = st.session_state.project_state.get("selected_design_image")
    original_image = (
        st.session_state.project_state["images"][0]["data"]
        if st.session_state.project_state["images"]
        else None
    )

    # Show before/after
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

    # Summary
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**Design:** {design.get('name', 'N/A')}")
        st.markdown(f"**Concept:** {design.get('concept', 'N/A')}")

    with col2:
        st.metric("Total Budget", f"${bom.get('grand_total', 0):,.0f}")

    st.divider()

    # Shopping links
    st.markdown("### 🛒 Shopping Links")

    for item in bom.get("items", []):
        if item.get("url"):
            st.markdown(f"- [{item['name']} - {item['retailer']}]({item['url']})")

    st.divider()

    # Download project data
    project_data = {
        "design": design,
        "bom": bom,
        "preferences": st.session_state.project_state.get("preferences", {}),
    }

    st.download_button(
        "📥 Download Project Data",
        data=json.dumps(project_data, indent=2),
        file_name="omnirenovation_project.json",
        mime="application/json",
    )

    # New project button
    if st.button("🔄 Start New Project", type="primary"):
        st.session_state.project_state = DEFAULT_STATE.copy()
        st.rerun()


# ============== MAIN APP ==============


def main():
    render_header()
    st.divider()

    # Route to appropriate phase
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
