"""
OmniRenovation AI - Phase 1 Pilot v8b
- No Replicate dependency (Python 3.14 compatible)
- Uses OpenAI DALL-E 3 for design generation (optional)
- SerpAPI for real product search with actual links
"""

import streamlit as st
import streamlit.components.v1 as components
import anthropic
import requests
import json
import base64
import re
import struct
from datetime import datetime
from io import BytesIO
from PIL import Image
from urllib.parse import quote_plus

# Optional OpenAI import
try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Page config
st.set_page_config(page_title="OmniRenovation AI", page_icon="🏠", layout="wide")

# Initialize session state
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
    "has_3d_scan": False,
    "scan_metadata": None,
    "glb_base64": None,
}

if "project_state" not in st.session_state:
    st.session_state.project_state = DEFAULT_STATE.copy()


# ============== API CLIENTS ==============


def get_claude_client():
    api_key = st.secrets.get("ANTHROPIC_API_KEY") or st.session_state.get(
        "anthropic_key"
    )
    return anthropic.Anthropic(api_key=api_key) if api_key else None


def get_openai_client():
    if not OPENAI_AVAILABLE:
        return None
    api_key = st.secrets.get("OPENAI_API_KEY") or st.session_state.get("openai_key")
    return openai.OpenAI(api_key=api_key) if api_key else None


def get_serpapi_key():
    return st.secrets.get("SERPAPI_KEY") or st.session_state.get("serpapi_key")


# ============== 3D VIEWER ==============


def create_3d_viewer_html(glb_base64: str, height: int = 400) -> str:
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ margin: 0; }}
            #container {{ width: 100%; height: {height}px; background: linear-gradient(135deg, #1a1a2e, #16213e); border-radius: 10px; }}
            #info {{ position: absolute; top: 10px; left: 10px; color: white; font: 12px Arial; background: rgba(0,0,0,0.5); padding: 8px; border-radius: 8px; }}
            #controls {{ position: absolute; bottom: 10px; left: 50%; transform: translateX(-50%); color: white; font: 12px Arial; background: rgba(0,0,0,0.5); padding: 8px 16px; border-radius: 20px; }}
        </style>
    </head>
    <body>
        <div id="container"><div id="info"></div><div id="controls">🖱️ Drag to rotate • Scroll to zoom</div></div>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/GLTFLoader.js"></script>
        <script>
            const container = document.getElementById('container');
            const scene = new THREE.Scene();
            const camera = new THREE.PerspectiveCamera(60, container.clientWidth/container.clientHeight, 0.1, 1000);
            camera.position.set(5, 5, 5);
            const renderer = new THREE.WebGLRenderer({{antialias: true, alpha: true}});
            renderer.setSize(container.clientWidth, container.clientHeight);
            renderer.outputEncoding = THREE.sRGBEncoding;
            container.appendChild(renderer.domElement);
            const controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            scene.add(new THREE.AmbientLight(0xffffff, 0.6));
            const light = new THREE.DirectionalLight(0xffffff, 0.8);
            light.position.set(5, 10, 7);
            scene.add(light);
            scene.add(new THREE.GridHelper(20, 20, 0x444444, 0x222222));
            
            const loader = new THREE.GLTFLoader();
            const b64 = "{glb_base64}";
            const bin = atob(b64);
            const bytes = new Uint8Array(bin.length);
            for(let i=0; i<bin.length; i++) bytes[i] = bin.charCodeAt(i);
            
            loader.parse(bytes.buffer, '', gltf => {{
                const model = gltf.scene;
                const box = new THREE.Box3().setFromObject(model);
                const center = box.getCenter(new THREE.Vector3());
                const size = box.getSize(new THREE.Vector3());
                model.position.sub(center);
                model.scale.multiplyScalar(5 / Math.max(size.x, size.y, size.z));
                scene.add(model);
                
                let verts = 0;
                model.traverse(c => {{ if(c.isMesh && c.geometry.attributes.position) verts += c.geometry.attributes.position.count; }});
                document.getElementById('info').innerHTML = 'Vertices: ' + verts.toLocaleString();
            }});
            
            function animate() {{ requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera); }}
            animate();
        </script>
    </body>
    </html>
    """


# ============== IMAGE PROCESSING ==============


def extract_glb_metadata(file_bytes):
    try:
        if struct.unpack("<I", file_bytes[0:4])[0] != 0x46546C67:
            return None, "Invalid GLB"
        chunk_len = struct.unpack("<I", file_bytes[12:16])[0]
        gltf = json.loads(file_bytes[20 : 20 + chunk_len].decode("utf-8"))
        return {
            "meshes": len(gltf.get("meshes", [])),
            "materials": len(gltf.get("materials", [])),
            "textures": len(gltf.get("textures", [])),
            "file_size_mb": len(file_bytes) / 1048576,
        }, None
    except Exception as e:
        return None, str(e)


def process_uploaded_image(uploaded_file):
    uploaded_file.seek(0)
    data = uploaded_file.read()
    uploaded_file.seek(0)
    try:
        img = Image.open(BytesIO(data))
        fmt = (img.format or "jpeg").lower()
        mime = {
            "jpeg": "image/jpeg",
            "jpg": "image/jpeg",
            "png": "image/png",
            "webp": "image/webp",
        }.get(fmt, "image/jpeg")
        if fmt not in ["jpeg", "jpg", "png", "webp"]:
            buf = BytesIO()
            img.convert("RGB").save(buf, "JPEG", quality=95)
            data = buf.getvalue()
            mime = "image/jpeg"
        return {
            "name": uploaded_file.name,
            "type": mime,
            "data": base64.b64encode(data).decode(),
            "success": True,
        }
    except Exception as e:
        return {"name": uploaded_file.name, "error": str(e), "success": False}


# ============== DESIGN GENERATION (DALL-E 3) ==============


def generate_design_dalle(room_description: str, style: str, variation: str) -> dict:
    """Generate design image using DALL-E 3"""
    client = get_openai_client()
    if not client:
        return {"error": "OpenAI API key not configured"}

    style_details = {
        "Modern Minimalist": "clean lines, neutral colors (white, grey, beige), minimal furniture, lots of natural light, simple geometric shapes, uncluttered spaces",
        "Scandinavian": "light wood (oak, birch), white walls, cozy textiles (wool, linen), plants, functional furniture, hygge atmosphere",
        "Industrial": "exposed brick walls, metal fixtures, Edison bulbs, dark colors, raw materials, concrete floors, loft aesthetic",
        "Mid-Century Modern": "retro 1950s-60s furniture, organic curves, wood and leather, bold accent colors, iconic designer pieces",
        "Contemporary": "current trends, mixed materials, neutral base with bold accents, comfortable yet stylish",
        "Bohemian": "eclectic patterns, rich jewel colors, layered textiles, global influences, plants everywhere",
        "Coastal": "blue and white palette, natural textures (rattan, jute), light and airy, nautical accents",
        "Farmhouse": "rustic wood, shiplap walls, neutral colors, vintage pieces, comfortable and welcoming",
    }

    variation_mood = {
        "Light & Airy": "bright, spacious, lots of natural light streaming through windows, white and light colors dominate",
        "Warm & Cozy": "warm lighting, layered textures, rich warm tones, inviting and comfortable atmosphere",
        "Bold & Dramatic": "striking contrasts, statement furniture pieces, rich deep colors, sophisticated and memorable",
    }

    prompt = f"""Professional interior design photograph of a beautifully renovated {room_description}.

Style: {style} - {style_details.get(style, style)}
Atmosphere: {variation} - {variation_mood.get(variation, variation)}

The room features carefully selected furniture, perfect lighting, styled accessories.
Ultra realistic, high-end interior design magazine quality, 8K resolution, 
professional architectural photography, natural lighting, detailed textures.

NO people, NO text, NO watermarks, NO logos."""

    try:
        response = client.images.generate(
            model="dall-e-3", prompt=prompt, size="1792x1024", quality="hd", n=1
        )

        image_url = response.data[0].url
        img_response = requests.get(image_url)

        if img_response.status_code == 200:
            return {
                "success": True,
                "image_base64": base64.b64encode(img_response.content).decode(),
                "prompt": prompt,
                "revised_prompt": response.data[0].revised_prompt,
            }
        return {"error": "Failed to download image"}

    except Exception as e:
        return {"error": str(e)}


# ============== SERPAPI - REAL PRODUCT SEARCH ==============


def search_products_serpapi(query: str, max_results: int = 5) -> list:
    """Search for real products using SerpAPI Google Shopping"""
    api_key = get_serpapi_key()
    if not api_key:
        return []

    try:
        params = {
            "engine": "google_shopping",
            "q": query,
            "api_key": api_key,
            "num": max_results,
            "hl": "en",
            "gl": "us",
        }

        response = requests.get("https://serpapi.com/search", params=params)
        data = response.json()

        products = []
        for item in data.get("shopping_results", [])[:max_results]:
            products.append(
                {
                    "product_name": item.get("title", "Unknown"),
                    "price": item.get("extracted_price", 0),
                    "price_str": item.get("price", "N/A"),
                    "retailer": item.get("source", "Unknown"),
                    "url": item.get("product_link") or item.get("link") or "",
                    "image_url": item.get("thumbnail", ""),
                    "rating": item.get("rating"),
                    "reviews": item.get("reviews"),
                    "delivery": item.get("delivery", ""),
                }
            )

        return products

    except Exception as e:
        st.error(f"SerpAPI error: {e}")
        return []


def search_products_fallback(query: str) -> list:
    """Fallback: Generate working search URLs for major retailers"""
    encoded = quote_plus(query)

    return [
        {
            "product_name": f"Search on Amazon",
            "retailer": "Amazon",
            "url": f"https://www.amazon.com/s?k={encoded}&i=garden",
            "price_str": "See results",
            "image_url": "",
        },
        {
            "product_name": f"Search on Wayfair",
            "retailer": "Wayfair",
            "url": f"https://www.wayfair.com/keyword.html?keyword={encoded}",
            "price_str": "See results",
            "image_url": "",
        },
        {
            "product_name": f"Search on IKEA",
            "retailer": "IKEA",
            "url": f"https://www.ikea.com/us/en/search/?q={encoded}",
            "price_str": "See results",
            "image_url": "",
        },
        {
            "product_name": f"Search on West Elm",
            "retailer": "West Elm",
            "url": f"https://www.westelm.com/search/?query={encoded}",
            "price_str": "See results",
            "image_url": "",
        },
        {
            "product_name": f"Search on Target",
            "retailer": "Target",
            "url": f"https://www.target.com/s?searchTerm={encoded}",
            "price_str": "See results",
            "image_url": "",
        },
    ]


def find_products_for_item(item: dict) -> list:
    """Find real products for a furniture item"""
    keywords = item.get("search_keywords", [])
    if keywords:
        query = keywords[0]
    else:
        parts = [
            item.get("color", ""),
            item.get("material", ""),
            item.get("style", ""),
            item.get("name", ""),
        ]
        query = " ".join([p for p in parts if p])[:80]

    if get_serpapi_key():
        products = search_products_serpapi(query, 5)
        if products:
            return products

    return search_products_fallback(query)


# ============== CLAUDE AGENTS ==============


def run_valuation_agent(
    images: list, preferences: dict, scan_metadata: dict = None
) -> dict:
    client = get_claude_client()
    if not client:
        return {"error": "Claude API key not configured"}

    image_content = []
    for img in images:
        if img.get("success") and "data" in img:
            image_content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": img["type"],
                        "data": img["data"],
                    },
                }
            )

    prompt = f"""Analyze this room for renovation.

Preferences: Budget {preferences.get('budget', 'N/A')}, Style: {preferences.get('style', 'Modern')}

Return ONLY valid JSON:
{{
    "property_assessment": {{
        "room_type": "living room/bedroom/kitchen/bathroom/etc",
        "current_condition": "poor/fair/good",
        "square_footage_estimate": "X sq ft",
        "current_style": "description",
        "natural_light": "poor/adequate/good/excellent",
        "notable_features": ["feature1", "feature2"]
    }},
    "renovation_scope": {{
        "recommended_work": [{{"area": "name", "work": "description", "priority": "high/medium/low"}}]
    }},
    "cost_estimate": {{
        "low": 0, "mid": 0, "high": 0
    }}
}}"""

    content = (
        image_content + [{"type": "text", "text": prompt}] if image_content else prompt
    )

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[{"role": "user", "content": content}],
        )
        text = response.content[0].text
        match = re.search(r"\{[\s\S]*\}", text)
        return json.loads(match.group()) if match else {"raw": text}
    except Exception as e:
        return {"error": str(e)}


def segment_furniture(design_image_base64: str) -> list:
    """Identify furniture items in a design image"""
    client = get_claude_client()
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
                                "media_type": "image/png",
                                "data": design_image_base64,
                            },
                        },
                        {
                            "type": "text",
                            "text": """Identify ALL furniture and decor in this interior design image.

Return ONLY a JSON array:
[
    {
        "id": 1,
        "item_type": "sofa",
        "name": "3-Seater Modern Sofa",
        "description": "Grey fabric sofa with wooden legs",
        "style": "mid-century modern",
        "color": "grey",
        "material": "fabric",
        "search_keywords": ["grey modern sofa", "mid century sofa", "3 seater grey sofa"],
        "price_estimate_low": 800,
        "price_estimate_high": 2000,
        "importance": "primary"
    }
]

Include: sofas, chairs, tables, lamps, rugs, artwork, plants, shelves, curtains, etc.
Be SPECIFIC about colors and materials for accurate product matching.""",
                        },
                    ],
                }
            ],
        )

        text = response.content[0].text
        match = re.search(r"\[[\s\S]*\]", text)
        return json.loads(match.group()) if match else []
    except Exception as e:
        st.error(f"Segmentation error: {e}")
        return []


def generate_design_concepts(valuation: dict, preferences: dict) -> dict:
    """Generate 3 design concept descriptions"""
    client = get_claude_client()
    if not client:
        return {"error": "API key not configured"}

    room = valuation.get("property_assessment", {})
    style = preferences.get("style", "Modern Minimalist")

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": f"""Create 3 design concepts for a {room.get('room_type', 'living room')} in {style} style.

Return JSON:
{{
    "design_options": [
        {{
            "option_number": 1,
            "name": "{style} - Light & Airy",
            "variation": "Light & Airy",
            "concept": "Brief description of this bright, spacious design",
            "color_palette": {{"primary": "#F5F5F5", "secondary": "#E0E0E0", "accent": "#4A90A4"}},
            "key_furniture": ["white sofa", "glass coffee table", "floor lamp"],
            "estimated_cost": 15000,
            "mood": "bright, spacious, calming"
        }},
        {{
            "option_number": 2,
            "name": "{style} - Warm & Cozy",
            "variation": "Warm & Cozy",
            "concept": "Description of warm, inviting design",
            "color_palette": {{"primary": "#F5E6D3", "secondary": "#D4A574", "accent": "#8B4513"}},
            "key_furniture": ["leather sofa", "wood coffee table", "table lamps"],
            "estimated_cost": 18000,
            "mood": "warm, inviting, comfortable"
        }},
        {{
            "option_number": 3,
            "name": "{style} - Bold & Dramatic",
            "variation": "Bold & Dramatic", 
            "concept": "Description of striking design",
            "color_palette": {{"primary": "#2C3E50", "secondary": "#34495E", "accent": "#E74C3C"}},
            "key_furniture": ["velvet sofa", "marble coffee table", "statement chandelier"],
            "estimated_cost": 25000,
            "mood": "dramatic, sophisticated, memorable"
        }}
    ]
}}""",
                }
            ],
        )

        text = response.content[0].text
        match = re.search(r"\{[\s\S]*\}", text)
        return json.loads(match.group()) if match else {"error": "Parse failed"}
    except Exception as e:
        return {"error": str(e)}


def generate_bom(selected_design: dict, furniture: list, products: dict) -> dict:
    """Generate final BOM with selected products"""
    product_lines = []
    total = 0

    for item in furniture:
        item_id = str(item.get("id", 0))
        matches = products.get(item_id, [])
        if matches:
            best = matches[0]
            price = best.get("price", 0)
            if isinstance(price, str):
                price = 0
            product_lines.append(
                {
                    "item": item.get("name"),
                    "product": best.get("product_name"),
                    "retailer": best.get("retailer"),
                    "price": price,
                    "url": best.get("url"),
                }
            )
            total += price

    # If no prices from SerpAPI, estimate from furniture
    if total == 0:
        for item in furniture:
            avg_price = (
                item.get("price_estimate_low", 500)
                + item.get("price_estimate_high", 1500)
            ) / 2
            total += avg_price

    return {
        "bill_of_materials": {
            "design": selected_design.get("name", "Design"),
            "items": product_lines,
            "products_total": total,
        },
        "labor_estimates": [
            {"trade": "Painter", "hours": 16, "rate": 50, "total": 800},
            {"trade": "Electrician", "hours": 8, "rate": 75, "total": 600},
            {"trade": "General Labor", "hours": 12, "rate": 40, "total": 480},
        ],
        "total_summary": {
            "products_total": total,
            "labor_total": 1880,
            "contingency": (total + 1880) * 0.1,
            "grand_total": (total + 1880) * 1.1,
        },
    }


# ============== UI ==============


def render_header():
    st.title("🏠 OmniRenovation AI")
    st.caption("AI Interior Design with Real Product Matching")

    phases = [
        "📤 Upload",
        "📊 Analysis",
        "🎨 Design",
        "🛋️ Products",
        "📦 BOM",
        "✅ Done",
    ]
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
                st.success(p)
            elif i == current:
                st.info(p)
            else:
                st.text(p)


def render_upload_phase():
    st.header("📤 Upload Your Room")

    # API Keys
    with st.expander("🔑 API Keys", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            key = st.text_input(
                "Anthropic API Key *",
                type="password",
                value=st.session_state.get("anthropic_key", ""),
                help="Required - for analysis and segmentation",
            )
            if key:
                st.session_state.anthropic_key = key

            key2 = st.text_input(
                "OpenAI API Key",
                type="password",
                value=st.session_state.get("openai_key", ""),
                help="Optional - for DALL-E design images",
            )
            if key2:
                st.session_state.openai_key = key2

        with col2:
            key3 = st.text_input(
                "SerpAPI Key",
                type="password",
                value=st.session_state.get("serpapi_key", ""),
                help="Optional - for real product links (100 free/month)",
            )
            if key3:
                st.session_state.serpapi_key = key3

            st.caption("💡 Without SerpAPI, you'll get search links to retailers")

        # Status
        status = []
        if st.session_state.get("anthropic_key"):
            status.append("✅ Claude")
        else:
            status.append("❌ Claude")
        if st.session_state.get("openai_key"):
            status.append("✅ DALL-E")
        if st.session_state.get("serpapi_key"):
            status.append("✅ Products")
        st.write(" | ".join(status))

    st.divider()

    # Upload
    tab1, tab2 = st.tabs(["📷 Photo", "🎯 3D Scan"])

    uploaded_images = []

    with tab1:
        uploaded_images = st.file_uploader(
            "Upload room photo(s)",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
        )

        if uploaded_images:
            cols = st.columns(min(4, len(uploaded_images)))
            for i, f in enumerate(uploaded_images):
                with cols[i % 4]:
                    f.seek(0)
                    st.image(Image.open(f), caption=f.name, use_column_width=True)

    with tab2:
        uploaded_3d = st.file_uploader("Upload GLB/GLTF", type=["glb", "gltf"])

        if uploaded_3d:
            uploaded_3d.seek(0)
            data = uploaded_3d.read()
            glb_b64 = base64.b64encode(data).decode()
            meta, _ = extract_glb_metadata(data)

            if meta:
                cols = st.columns(3)
                cols[0].metric("Meshes", meta["meshes"])
                cols[1].metric("Materials", meta["materials"])
                cols[2].metric("Size", f"{meta['file_size_mb']:.1f}MB")

            components.html(create_3d_viewer_html(glb_b64, 350), height=380)

            st.session_state.project_state["glb_base64"] = glb_b64
            st.session_state.project_state["scan_metadata"] = meta
            st.session_state.project_state["has_3d_scan"] = True

    st.divider()

    # Preferences
    col1, col2 = st.columns(2)
    with col1:
        budget = st.select_slider(
            "Budget",
            ["<$5K", "$5-15K", "$15-30K", "$30-50K", "$50-100K", ">$100K"],
            "$15-30K",
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
        goals = st.multiselect(
            "Goals",
            ["Update style", "Increase value", "Improve function", "More storage"],
            ["Update style"],
        )
        room_type = st.selectbox(
            "Room Type",
            [
                "Living Room",
                "Bedroom",
                "Kitchen",
                "Bathroom",
                "Home Office",
                "Dining Room",
            ],
        )

    st.divider()

    can_proceed = bool(st.session_state.get("anthropic_key")) and bool(uploaded_images)

    if st.button("🚀 Analyze Room", type="primary", disabled=not can_proceed):
        images = [process_uploaded_image(f) for f in uploaded_images]
        images = [i for i in images if i["success"]]

        st.session_state.project_state["images"] = images
        st.session_state.project_state["preferences"] = {
            "budget": budget,
            "style": style,
            "goals": ", ".join(goals),
            "room_type": room_type,
        }
        st.session_state.project_state["phase"] = "valuation"
        st.rerun()

    if not st.session_state.get("anthropic_key"):
        st.warning("⚠️ Add Anthropic API key to continue")


def render_valuation_phase():
    st.header("📊 Room Analysis")

    if not st.session_state.project_state["valuation"]:
        with st.spinner("🔍 Analyzing your room..."):
            val = run_valuation_agent(
                st.session_state.project_state["images"],
                st.session_state.project_state["preferences"],
                st.session_state.project_state.get("scan_metadata"),
            )
            st.session_state.project_state["valuation"] = val
            st.rerun()

    val = st.session_state.project_state["valuation"]

    if "error" in val:
        st.error(val["error"])
        if st.button("🔄 Retry"):
            st.session_state.project_state["valuation"] = None
            st.rerun()
        return

    # Display
    assessment = val.get("property_assessment", {})
    cost = val.get("cost_estimate", {})

    col1, col2, col3 = st.columns(3)
    col1.metric("Room Type", assessment.get("room_type", "N/A"))
    col2.metric("Condition", assessment.get("current_condition", "N/A"))
    col3.metric("Est. Cost", f"${cost.get('mid', 0):,}")

    with st.expander("Details", expanded=True):
        st.write(f"**Size:** {assessment.get('square_footage_estimate', 'N/A')}")
        st.write(f"**Current Style:** {assessment.get('current_style', 'N/A')}")
        st.write(f"**Natural Light:** {assessment.get('natural_light', 'N/A')}")
        features = assessment.get("notable_features", [])
        if features:
            st.write(f"**Features:** {', '.join(features)}")

    # Recommended work
    scope = val.get("renovation_scope", {})
    work = scope.get("recommended_work", [])
    if work:
        with st.expander("Recommended Work"):
            for w in work:
                priority_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
                    w.get("priority", ""), "⚪"
                )
                st.write(
                    f"{priority_icon} **{w.get('area', 'N/A')}:** {w.get('work', 'N/A')}"
                )

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎨 Generate Designs", type="primary"):
            st.session_state.project_state["phase"] = "design"
            st.rerun()
    with col2:
        if st.button("← Start Over"):
            st.session_state.project_state = DEFAULT_STATE.copy()
            st.rerun()


def render_design_phase():
    st.header("🎨 AI-Generated Designs")

    # Generate concepts
    if not st.session_state.project_state["designs"]:
        with st.spinner("Creating design concepts..."):
            designs = generate_design_concepts(
                st.session_state.project_state["valuation"],
                st.session_state.project_state["preferences"],
            )
            st.session_state.project_state["designs"] = designs
            st.rerun()

    designs = st.session_state.project_state["designs"]
    if "error" in designs:
        st.error(designs["error"])
        if st.button("🔄 Retry"):
            st.session_state.project_state["designs"] = None
            st.rerun()
        return

    has_openai = bool(st.session_state.get("openai_key")) and OPENAI_AVAILABLE
    design_images = st.session_state.project_state.get("design_images", {})

    # Show original
    if st.session_state.project_state["images"]:
        with st.expander("📷 Your Original Room", expanded=False):
            img_data = st.session_state.project_state["images"][0]
            st.image(base64.b64decode(img_data["data"]), use_column_width=True)

    # Display options
    options = designs.get("design_options", [])

    for opt in options:
        opt_num = opt["option_number"]

        st.subheader(f"Option {opt_num}: {opt.get('name', 'Design')}")

        col1, col2 = st.columns([2, 1])

        with col1:
            # Show image or generate button
            if opt_num in design_images and design_images[opt_num].get("success"):
                st.image(
                    base64.b64decode(design_images[opt_num]["image_base64"]),
                    caption=f"AI-Generated: {opt.get('name')}",
                    use_column_width=True,
                )
            elif opt_num in design_images and design_images[opt_num].get("error"):
                st.error(f"Generation failed: {design_images[opt_num]['error']}")
                if st.button(f"🔄 Retry Option {opt_num}", key=f"retry_{opt_num}"):
                    del design_images[opt_num]
                    st.session_state.project_state["design_images"] = design_images
                    st.rerun()
            else:
                # Show color palette as preview
                palette = opt.get("color_palette", {})
                pcols = st.columns(3)
                for i, (k, v) in enumerate(list(palette.items())[:3]):
                    pcols[i].color_picker(
                        k.title(), v, disabled=True, key=f"c_{opt_num}_{i}"
                    )

                if has_openai:
                    if st.button(f"🖼️ Generate Image", key=f"gen_{opt_num}"):
                        room_type = (
                            st.session_state.project_state["valuation"]
                            .get("property_assessment", {})
                            .get("room_type", "living room")
                        )
                        with st.spinner(
                            f"Generating {opt.get('variation', 'design')} image... (~30 sec)"
                        ):
                            result = generate_design_dalle(
                                room_type,
                                st.session_state.project_state["preferences"]["style"],
                                opt.get("variation", "Light & Airy"),
                            )
                            design_images[opt_num] = result
                            st.session_state.project_state["design_images"] = (
                                design_images
                            )
                            st.rerun()
                else:
                    st.info("💡 Add OpenAI API key to generate design images")

        with col2:
            st.write(f"**Mood:** {opt.get('mood', 'N/A')}")
            st.write(f"**Concept:** {opt.get('concept', 'N/A')}")
            st.metric("Est. Cost", f"${opt.get('estimated_cost', 0):,}")

            st.write("**Key Furniture:**")
            for item in opt.get("key_furniture", [])[:5]:
                st.write(f"• {item}")

            if st.button(
                f"✅ Select Option {opt_num}", key=f"sel_{opt_num}", type="primary"
            ):
                st.session_state.project_state["selected_design"] = opt
                if opt_num in design_images and design_images[opt_num].get("success"):
                    st.session_state.project_state["selected_design_image"] = (
                        design_images[opt_num]["image_base64"]
                    )
                st.success("Selected!")

        st.divider()

    # Generate all button
    if has_openai and len(design_images) < len(options):
        if st.button("🎨 Generate All Design Images"):
            room_type = (
                st.session_state.project_state["valuation"]
                .get("property_assessment", {})
                .get("room_type", "living room")
            )
            for opt in options:
                if opt["option_number"] not in design_images:
                    with st.spinner(f"Generating {opt.get('name')}..."):
                        result = generate_design_dalle(
                            room_type,
                            st.session_state.project_state["preferences"]["style"],
                            opt.get("variation", "Light & Airy"),
                        )
                        design_images[opt["option_number"]] = result
            st.session_state.project_state["design_images"] = design_images
            st.rerun()

    # Continue
    if st.session_state.project_state.get("selected_design"):
        st.success(
            f"✓ Selected: {st.session_state.project_state['selected_design'].get('name')}"
        )
        if st.button("🛋️ Find Real Products", type="primary"):
            st.session_state.project_state["phase"] = "products"
            st.rerun()

    if st.button("← Back"):
        st.session_state.project_state["phase"] = "valuation"
        st.rerun()


def render_products_phase():
    st.header("🛋️ Real Product Matching")

    # Get image for segmentation
    design_img = st.session_state.project_state.get("selected_design_image")
    if not design_img and st.session_state.project_state["images"]:
        design_img = st.session_state.project_state["images"][0]["data"]

    if not design_img:
        st.warning("No image for product matching")
        if st.button("← Back"):
            st.session_state.project_state["phase"] = "design"
            st.rerun()
        return

    # Segment furniture
    if not st.session_state.project_state.get("furniture_items"):
        with st.spinner("🔍 Identifying furniture..."):
            furniture = segment_furniture(design_img)
            st.session_state.project_state["furniture_items"] = furniture
            st.rerun()

    furniture = st.session_state.project_state["furniture_items"]

    if not furniture:
        st.warning("Could not identify items. Using design suggestions instead.")
        # Use key_furniture from selected design
        design = st.session_state.project_state.get("selected_design", {})
        furniture = [
            {"id": i + 1, "name": item, "search_keywords": [item]}
            for i, item in enumerate(
                design.get("key_furniture", ["sofa", "table", "lamp"])
            )
        ]
        st.session_state.project_state["furniture_items"] = furniture

    # Display
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Design")
        st.image(base64.b64decode(design_img), use_column_width=True)

    with col2:
        st.subheader(f"Found {len(furniture)} Items")
        for item in furniture:
            st.write(f"**{item.get('id')}. {item.get('name', 'Item')}**")
            if item.get("color") or item.get("material"):
                st.caption(f"{item.get('color', '')} {item.get('material', '')}")

    st.divider()
    st.subheader("🛒 Product Matches")

    has_serpapi = bool(get_serpapi_key())
    if has_serpapi:
        st.success("✅ Using SerpAPI - Real product links!")
    else:
        st.info(
            "💡 Add SerpAPI key for direct product links. Currently showing search links."
        )

    product_matches = st.session_state.project_state.get("product_matches", {})

    for item in furniture:
        item_id = str(item.get("id", 0))

        with st.expander(f"🔍 {item.get('name', 'Item')}", expanded=True):
            if item_id not in product_matches:
                with st.spinner(f"Searching..."):
                    matches = find_products_for_item(item)
                    product_matches[item_id] = matches
                    st.session_state.project_state["product_matches"] = product_matches

            matches = product_matches.get(item_id, [])

            if matches:
                cols = st.columns(min(3, len(matches)))

                for i, match in enumerate(matches[:3]):
                    with cols[i]:
                        # Image
                        if match.get("image_url"):
                            try:
                                st.image(match["image_url"], width=120)
                            except:
                                pass

                        st.write(f"**{match.get('product_name', 'Product')[:40]}**")
                        st.write(f"🏪 {match.get('retailer', 'N/A')}")
                        st.write(f"💰 {match.get('price_str', 'N/A')}")

                        if match.get("rating"):
                            st.write(f"⭐ {match['rating']}")

                        # REAL CLICKABLE LINK
                        url = match.get("url", "")
                        if url:
                            st.link_button("🔗 View product", url)

                        if st.button(
                            "✓", key=f"p_{item_id}_{i}", help="Select this product"
                        ):
                            st.session_state.project_state["selected_products"][
                                item_id
                            ] = match
                            st.success("✓")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📦 Generate BOM", type="primary"):
            st.session_state.project_state["phase"] = "bom"
            st.rerun()
    with col2:
        if st.button("← Back"):
            st.session_state.project_state["phase"] = "design"
            st.rerun()


def render_bom_phase():
    st.header("📦 Bill of Materials")

    if not st.session_state.project_state.get("bom"):
        bom = generate_bom(
            st.session_state.project_state.get("selected_design", {}),
            st.session_state.project_state.get("furniture_items", []),
            st.session_state.project_state.get("product_matches", {}),
        )
        st.session_state.project_state["bom"] = bom

    bom = st.session_state.project_state["bom"]
    summary = bom.get("total_summary", {})

    col1, col2, col3 = st.columns(3)
    col1.metric("Products", f"${summary.get('products_total', 0):,.0f}")
    col2.metric("Labor", f"${summary.get('labor_total', 0):,.0f}")
    col3.metric("**Total**", f"${summary.get('grand_total', 0):,.0f}")

    st.divider()

    st.subheader("🛍️ Products")
    for item in bom.get("bill_of_materials", {}).get("items", []):
        col1, col2, col3 = st.columns([3, 1, 1])
        col1.write(f"**{item.get('item')}**")
        col1.caption(f"{item.get('product', '')} ({item.get('retailer', '')})")
        col2.write(f"${item.get('price', 0):,.0f}" if item.get("price") else "See link")
        if item.get("url"):
            col3.link_button("🔗 Link", item["url"])

    st.subheader("👷 Labor")
    for labor in bom.get("labor_estimates", []):
        st.write(
            f"• {labor['trade']}: {labor['hours']}hrs @ ${labor['rate']}/hr = **${labor['total']}**"
        )

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Complete Project", type="primary"):
            st.session_state.project_state["phase"] = "complete"
            st.rerun()
    with col2:
        if st.button("← Back"):
            st.session_state.project_state["phase"] = "products"
            st.rerun()


def render_complete_phase():
    st.header("✅ Project Complete!")
    st.balloons()

    design = st.session_state.project_state.get("selected_design", {})
    bom = st.session_state.project_state.get("bom", {})

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Selected Design")
        if st.session_state.project_state.get("selected_design_image"):
            st.image(
                base64.b64decode(
                    st.session_state.project_state["selected_design_image"]
                ),
                use_column_width=True,
            )
        st.write(f"**{design.get('name', 'N/A')}**")
        st.write(design.get("concept", ""))

    with col2:
        st.subheader("Budget Summary")
        summary = bom.get("total_summary", {})
        st.metric("Total Project Cost", f"${summary.get('grand_total', 0):,.0f}")
        st.write(f"• Products: ${summary.get('products_total', 0):,.0f}")
        st.write(f"• Labor: ${summary.get('labor_total', 0):,.0f}")
        st.write(f"• Contingency (10%): ${summary.get('contingency', 0):,.0f}")

    st.divider()

    # Product links summary
    st.subheader("🔗 Quick Links")
    for item in bom.get("bill_of_materials", {}).get("items", []):
        if item.get("url"):
            st.write(f"• {item.get('item')} - {item.get('retailer')}")
            st.link_button("Open", item["url"])

    st.divider()

    # Export
    export = {
        "timestamp": datetime.now().isoformat(),
        "design": design,
        "furniture": st.session_state.project_state.get("furniture_items", []),
        "products": st.session_state.project_state.get("product_matches", {}),
        "bom": bom,
    }

    st.download_button(
        "📥 Download Project JSON",
        json.dumps(export, indent=2),
        f"renovation_{datetime.now().strftime('%Y%m%d')}.json",
        "application/json",
    )

    if st.button("🔄 Start New Project"):
        st.session_state.project_state = DEFAULT_STATE.copy()
        st.rerun()


def main():
    render_header()
    st.divider()

    phase = st.session_state.project_state["phase"]

    {
        "upload": render_upload_phase,
        "valuation": render_valuation_phase,
        "design": render_design_phase,
        "products": render_products_phase,
        "bom": render_bom_phase,
        "complete": render_complete_phase,
    }.get(phase, render_upload_phase)()


if __name__ == "__main__":
    main()
