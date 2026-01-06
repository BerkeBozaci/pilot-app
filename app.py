"""
OmniRenovation AI - Phase 1 Pilot v7
With AI-generated design images and real product matching
"""

import streamlit as st
import streamlit.components.v1 as components
import anthropic
import openai
import json
import base64
from datetime import datetime
from io import BytesIO
from PIL import Image
import struct
import requests
import re

# Page config
st.set_page_config(page_title="OmniRenovation AI", page_icon="🏠", layout="wide")

# Initialize session state
if "project_state" not in st.session_state:
    st.session_state.project_state = {
        "phase": "upload",
        "images": [],
        "preferences": {},
        "valuation": None,
        "designs": None,
        "design_images": {},  # Generated design images
        "selected_design": None,
        "furniture_items": [],  # Segmented furniture
        "product_matches": {},  # Real product matches
        "bom": None,
        "gate_1_approved": False,
        "gate_2_approved": False,
        "has_3d_scan": False,
        "scan_metadata": None,
        "glb_base64": None,
    }


# ============== 3D VIEWER HTML ==============


def create_3d_viewer_html(glb_base64: str, height: int = 500) -> str:
    """Create Three.js viewer for GLB files"""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ margin: 0; padding: 0; overflow: hidden; }}
            #container {{ 
                width: 100%; 
                height: {height}px; 
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                border-radius: 10px;
            }}
            #loading {{
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                color: white;
                font-family: Arial, sans-serif;
            }}
            #controls {{
                position: absolute;
                bottom: 10px;
                left: 50%;
                transform: translateX(-50%);
                color: white;
                font-family: Arial, sans-serif;
                font-size: 12px;
                background: rgba(0,0,0,0.5);
                padding: 8px 16px;
                border-radius: 20px;
            }}
            #info {{
                position: absolute;
                top: 10px;
                left: 10px;
                color: white;
                font-family: Arial, sans-serif;
                font-size: 12px;
                background: rgba(0,0,0,0.5);
                padding: 8px 12px;
                border-radius: 8px;
            }}
        </style>
    </head>
    <body>
        <div id="container">
            <div id="loading">Loading 3D model...</div>
            <div id="info"></div>
            <div id="controls">🖱️ Drag to rotate • Scroll to zoom • Right-click to pan</div>
        </div>
        
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/GLTFLoader.js"></script>
        
        <script>
            const glbBase64 = "{glb_base64}";
            
            function base64ToArrayBuffer(base64) {{
                const binaryString = atob(base64);
                const bytes = new Uint8Array(binaryString.length);
                for (let i = 0; i < binaryString.length; i++) {{
                    bytes[i] = binaryString.charCodeAt(i);
                }}
                return bytes.buffer;
            }}
            
            const container = document.getElementById('container');
            const scene = new THREE.Scene();
            
            const camera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 0.1, 1000);
            camera.position.set(5, 5, 5);
            
            const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
            renderer.setSize(container.clientWidth, container.clientHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
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
            loader.parse(base64ToArrayBuffer(glbBase64), '', 
                function(gltf) {{
                    const model = gltf.scene;
                    const box = new THREE.Box3().setFromObject(model);
                    const center = box.getCenter(new THREE.Vector3());
                    const size = box.getSize(new THREE.Vector3());
                    
                    model.position.sub(center);
                    const maxDim = Math.max(size.x, size.y, size.z);
                    model.scale.multiplyScalar(5 / maxDim);
                    
                    scene.add(model);
                    
                    let vertexCount = 0, triangleCount = 0;
                    model.traverse(child => {{
                        if (child.isMesh && child.geometry.attributes.position) {{
                            vertexCount += child.geometry.attributes.position.count;
                            triangleCount += child.geometry.index ? 
                                child.geometry.index.count / 3 : 
                                child.geometry.attributes.position.count / 3;
                        }}
                    }});
                    
                    document.getElementById('info').innerHTML = 
                        `Vertices: ${{vertexCount.toLocaleString()}}<br>Triangles: ${{Math.floor(triangleCount).toLocaleString()}}`;
                    document.getElementById('loading').style.display = 'none';
                }}
            );
            
            function animate() {{
                requestAnimationFrame(animate);
                controls.update();
                renderer.render(scene, camera);
            }}
            animate();
        </script>
    </body>
    </html>
    """
    return html


# ============== API CLIENTS ==============


def get_claude_client():
    """Get Claude client"""
    api_key = st.secrets.get("ANTHROPIC_API_KEY") or st.session_state.get(
        "anthropic_key"
    )
    if not api_key:
        return None
    return anthropic.Anthropic(api_key=api_key)


def get_openai_client():
    """Get OpenAI client for DALL-E"""
    api_key = st.secrets.get("OPENAI_API_KEY") or st.session_state.get("openai_key")
    if not api_key:
        return None
    return openai.OpenAI(api_key=api_key)


# ============== IMAGE PROCESSING ==============


def extract_glb_metadata(file_bytes):
    """Extract metadata from GLB file"""
    try:
        magic = struct.unpack("<I", file_bytes[0:4])[0]
        if magic != 0x46546C67:
            return None, "Not a valid GLB file"

        version = struct.unpack("<I", file_bytes[4:8])[0]
        chunk_length = struct.unpack("<I", file_bytes[12:16])[0]
        json_data = file_bytes[20 : 20 + chunk_length].decode("utf-8")
        gltf = json.loads(json_data)

        return {
            "format": "GLB",
            "version": version,
            "file_size_mb": len(file_bytes) / (1024 * 1024),
            "meshes": len(gltf.get("meshes", [])),
            "materials": len(gltf.get("materials", [])),
            "textures": len(gltf.get("textures", [])),
        }, None
    except Exception as e:
        return None, str(e)


def process_uploaded_image(uploaded_file):
    """Process uploaded image to base64"""
    uploaded_file.seek(0)
    bytes_data = uploaded_file.read()
    uploaded_file.seek(0)

    try:
        img = Image.open(BytesIO(bytes_data))
        actual_format = img.format.lower() if img.format else "jpeg"

        mime_map = {
            "jpeg": "image/jpeg",
            "jpg": "image/jpeg",
            "png": "image/png",
            "gif": "image/gif",
            "webp": "image/webp",
        }
        mime_type = mime_map.get(actual_format, "image/jpeg")

        if actual_format not in mime_map:
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGB")
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=95)
            bytes_data = buffer.getvalue()
            mime_type = "image/jpeg"

        return {
            "name": uploaded_file.name,
            "type": mime_type,
            "data": base64.standard_b64encode(bytes_data).decode("utf-8"),
            "success": True,
        }
    except Exception as e:
        return {"name": uploaded_file.name, "error": str(e), "success": False}


# ============== AI AGENTS ==============


def run_valuation_agent(
    images: list, preferences: dict, scan_metadata: dict = None
) -> dict:
    """Valuation Agent - analyzes property"""
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

    metadata_text = ""
    if scan_metadata:
        metadata_text = f"\n3D Scan: {scan_metadata.get('meshes', 0)} meshes, {scan_metadata.get('file_size_mb', 0):.1f}MB"

    prompt = f"""Analyze this property for renovation.{metadata_text}

Preferences: Budget {preferences.get('budget', 'N/A')}, Style: {preferences.get('style', 'Modern')}, 
Goals: {preferences.get('goals', 'N/A')}, Priorities: {preferences.get('priorities', 'N/A')}

Return JSON:
{{
    "property_assessment": {{
        "room_type": "living room/bedroom/kitchen/etc",
        "room_types_identified": ["rooms"],
        "current_condition": "poor/fair/good",
        "condition_details": "description",
        "square_footage_estimate": "X sq ft",
        "current_style": "description of current style",
        "natural_light": "poor/adequate/good/excellent"
    }},
    "renovation_scope": {{
        "recommended_work": [{{"area": "name", "work_needed": "desc", "priority": "high/medium/low"}}],
        "quick_wins": ["wins"]
    }},
    "cost_estimate": {{
        "low_estimate": 0, "mid_estimate": 0, "high_estimate": 0,
        "breakdown": [{{"category": "name", "low": 0, "high": 0}}]
    }},
    "roi_analysis": {{"estimated_value_increase": 0, "roi_percentage": 0}},
    "timeline_estimate": {{"minimum_weeks": 0, "maximum_weeks": 0}}
}}"""

    if image_content:
        image_content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": image_content}]
    else:
        messages = [{"role": "user", "content": prompt}]

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=4096, messages=messages
        )
        text = response.content[0].text
        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            return json.loads(json_match.group())
        return {"raw_response": text}
    except Exception as e:
        return {"error": str(e)}


def generate_design_prompt(room_info: dict, style: str, option_num: int) -> str:
    """Generate a detailed prompt for DALL-E based on room analysis"""

    style_descriptions = {
        "Modern Minimalist": "clean lines, neutral colors, minimal furniture, open space, white walls, natural light, simple geometric shapes",
        "Contemporary": "current trends, mixed materials, neutral with bold accents, comfortable yet stylish, artistic elements",
        "Scandinavian": "light wood, white and grey tones, cozy textures, functional furniture, plants, hygge atmosphere, natural materials",
        "Industrial": "exposed brick, metal fixtures, dark colors, Edison bulbs, raw materials, urban loft aesthetic, concrete elements",
        "Mid-Century Modern": "retro 1950s-60s style, organic shapes, wood furniture, bold colors, iconic designer pieces, tapered legs",
        "Traditional": "classic furniture, rich colors, elegant patterns, crown molding, symmetry, timeless design",
        "Bohemian": "eclectic mix, rich colors, global patterns, layered textiles, plants, artistic, collected-over-time feel",
        "Coastal": "beach-inspired, blue and white palette, natural textures, light and airy, nautical accents, weathered wood",
        "Farmhouse": "rustic charm, shiplap walls, barn doors, vintage pieces, neutral colors, cozy and welcoming",
    }

    style_desc = style_descriptions.get(style, style_descriptions["Modern Minimalist"])
    room_type = room_info.get("room_type", "living room")
    sq_ft = room_info.get("square_footage_estimate", "medium-sized")

    # Different variations for each option
    variations = [
        "bright and airy with emphasis on natural light",
        "cozy and warm with layered textures",
        "bold and dramatic with statement pieces",
    ]
    variation = variations[option_num % 3]

    prompt = f"""Professional interior design photograph of a beautifully renovated {room_type}, {style} style.
    
Design characteristics: {style_desc}
Atmosphere: {variation}
Room size: {sq_ft}

The space features carefully curated furniture, perfect lighting, styled accessories, and professional staging.
Photorealistic, high-end interior design magazine quality, 4K, detailed textures, natural lighting from windows.
NO people, NO text, NO watermarks."""

    return prompt


def generate_design_images(valuation: dict, preferences: dict) -> dict:
    """Generate 3 design images using DALL-E 3"""
    client = get_openai_client()
    if not client:
        return {
            "error": "OpenAI API key not configured. Add OPENAI_API_KEY to secrets."
        }

    room_info = valuation.get("property_assessment", {})
    style = preferences.get("style", "Modern Minimalist")

    design_images = {}
    design_names = [
        f"{style} - Light & Airy",
        f"{style} - Warm & Cozy",
        f"{style} - Bold & Dramatic",
    ]

    for i in range(3):
        try:
            prompt = generate_design_prompt(room_info, style, i)

            response = client.images.generate(
                model="dall-e-3", prompt=prompt, size="1792x1024", quality="hd", n=1
            )

            image_url = response.data[0].url

            # Download image and convert to base64
            img_response = requests.get(image_url)
            if img_response.status_code == 200:
                img_base64 = base64.b64encode(img_response.content).decode("utf-8")
                design_images[i + 1] = {
                    "name": design_names[i],
                    "prompt": prompt,
                    "image_base64": img_base64,
                    "revised_prompt": response.data[0].revised_prompt,
                }

        except Exception as e:
            design_images[i + 1] = {"name": design_names[i], "error": str(e)}

    return design_images


def segment_furniture_from_design(design_image_base64: str, design_name: str) -> list:
    """Use Claude to identify and segment furniture items from a design image"""
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
                            "text": """Analyze this interior design image and identify ALL furniture and decor items visible.

For each item, provide:
1. Item name/type
2. Detailed description (style, color, material, approximate size)
3. Search keywords for finding similar products online
4. Estimated price range (USD)
5. Position in image (left/center/right, foreground/background)

Return JSON array:
[
    {
        "id": 1,
        "item_type": "sofa",
        "name": "3-Seater Modern Sofa",
        "description": "Light grey fabric, clean lines, wooden legs, approximately 84 inches wide",
        "style": "mid-century modern",
        "color": "light grey",
        "material": "fabric upholstery, wood legs",
        "search_keywords": ["grey modern sofa", "mid-century 3 seater sofa", "fabric sofa wooden legs"],
        "estimated_price_low": 800,
        "estimated_price_high": 2000,
        "position": "center foreground",
        "importance": "primary"
    }
]

Include ALL visible items: sofas, chairs, tables, lamps, rugs, artwork, plants, shelving, etc.
Be specific about colors, materials, and dimensions.""",
                        },
                    ],
                }
            ],
        )

        text = response.content[0].text
        json_match = re.search(r"\[[\s\S]*\]", text)
        if json_match:
            return json.loads(json_match.group())
        return []

    except Exception as e:
        st.error(f"Furniture segmentation error: {e}")
        return []


def find_real_products(furniture_item: dict) -> list:
    """Find real products matching a furniture item using web search"""
    client = get_claude_client()
    if not client:
        return []

    # For MVP, we'll use Claude to generate realistic product suggestions
    # In production, you'd use actual APIs: Google Shopping, Amazon Product API, etc.

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": f"""Find 3 REAL products that match this furniture item:

Item: {furniture_item.get('name', 'Unknown')}
Description: {furniture_item.get('description', '')}
Style: {furniture_item.get('style', '')}
Color: {furniture_item.get('color', '')}
Material: {furniture_item.get('material', '')}
Price Range: ${furniture_item.get('estimated_price_low', 0)} - ${furniture_item.get('estimated_price_high', 1000)}

Return JSON array with 3 REAL products from major retailers (IKEA, Wayfair, West Elm, CB2, Target, Amazon, Article, etc.):
[
    {{
        "product_name": "Actual product name",
        "retailer": "Store name",
        "price": 599,
        "url": "https://www.retailer.com/product-page",
        "match_score": 95,
        "match_notes": "Why this matches",
        "color_options": ["Grey", "Blue", "Beige"],
        "dimensions": "84W x 36D x 32H inches"
    }}
]

Use REAL product names and realistic URLs (format: https://www.retailer.com/product-category/product-name).
Prioritize popular, well-reviewed products.""",
                }
            ],
        )

        text = response.content[0].text
        json_match = re.search(r"\[[\s\S]*\]", text)
        if json_match:
            return json.loads(json_match.group())
        return []

    except Exception as e:
        return []


def run_design_agent_with_images(
    images: list, preferences: dict, valuation: dict
) -> dict:
    """Design Agent that generates actual design images"""

    # First, generate design concepts (text)
    client = get_claude_client()
    if not client:
        return {"error": "Claude API key not configured"}

    room_info = valuation.get("property_assessment", {})
    style = preferences.get("style", "Modern Minimalist")
    budget = preferences.get("budget", "$15,000 - $30,000")

    # Generate design concepts
    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": f"""Create 3 distinct design concepts for this {room_info.get('room_type', 'room')}:

Style: {style}
Budget: {budget}
Current Condition: {room_info.get('current_condition', 'N/A')}
Room Size: {room_info.get('square_footage_estimate', 'N/A')}

Return JSON:
{{
    "design_options": [
        {{
            "option_number": 1,
            "name": "Light & Airy {style}",
            "concept": "Description of this design variation",
            "color_palette": {{"primary": "#hex", "secondary": "#hex", "accent": "#hex"}},
            "key_furniture": ["main pieces needed"],
            "key_features": ["design features"],
            "estimated_cost": 15000,
            "mood": "bright, spacious, welcoming"
        }},
        {{
            "option_number": 2,
            "name": "Warm & Cozy {style}",
            "concept": "Description",
            "color_palette": {{"primary": "#hex", "secondary": "#hex", "accent": "#hex"}},
            "key_furniture": ["pieces"],
            "key_features": ["features"],
            "estimated_cost": 18000,
            "mood": "warm, inviting, comfortable"
        }},
        {{
            "option_number": 3,
            "name": "Bold & Dramatic {style}",
            "concept": "Description",
            "color_palette": {{"primary": "#hex", "secondary": "#hex", "accent": "#hex"}},
            "key_furniture": ["pieces"],
            "key_features": ["features"],
            "estimated_cost": 22000,
            "mood": "striking, sophisticated, memorable"
        }}
    ]
}}""",
                }
            ],
        )

        text = response.content[0].text
        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            designs = json.loads(json_match.group())
        else:
            return {"error": "Could not parse design concepts"}

        return designs

    except Exception as e:
        return {"error": str(e)}


def run_procurement_agent(
    selected_design: dict, furniture_items: list, product_matches: dict
) -> dict:
    """Generate BOM based on selected design and matched products"""
    client = get_claude_client()
    if not client:
        return {"error": "API key not configured"}

    # Build product list from matches
    products_text = ""
    for item in furniture_items:
        item_id = item.get("id", 0)
        matches = product_matches.get(str(item_id), [])
        if matches:
            best_match = matches[0]
            products_text += f"- {item.get('name')}: {best_match.get('product_name')} from {best_match.get('retailer')} - ${best_match.get('price', 0)}\n"

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": f"""Create a Bill of Materials for this renovation:

Design: {selected_design.get('name', 'Renovation')}
Budget: ${selected_design.get('estimated_cost', 20000)}

Selected Products:
{products_text}

Return JSON:
{{
    "bill_of_materials": {{
        "categories": [
            {{
                "category_name": "Furniture",
                "items": [
                    {{"item_name": "Product name", "retailer": "Store", "quantity": 1, "unit_price": 0, "total_price": 0, "url": "link"}}
                ],
                "category_total": 0
            }}
        ]
    }},
    "labor_estimates": [{{"trade": "Trade", "hours": 0, "rate": 50, "total": 0}}],
    "total_summary": {{
        "products_total": 0,
        "labor_total": 0,
        "contingency": 0,
        "grand_total": 0
    }}
}}""",
                }
            ],
        )

        text = response.content[0].text
        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            return json.loads(json_match.group())
        return {"raw_response": text}

    except Exception as e:
        return {"error": str(e)}


# ============== UI COMPONENTS ==============


def render_header():
    st.title("🏠 OmniRenovation AI")
    st.caption("AI-Native Renovation Platform with Visual Design Generation")

    phases = [
        "📤 Upload",
        "📊 Valuation",
        "🎨 Design",
        "🛋️ Products",
        "📦 BOM",
        "✅ Done",
    ]
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
                st.success(phase)
            elif i == current:
                st.info(phase)
            else:
                st.text(phase)


def render_upload_phase():
    st.header("📤 Upload Property Images or 3D Scan")

    # API Keys
    with st.expander("🔑 API Configuration", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            anthropic_key = st.text_input(
                "Anthropic API Key (Required)",
                type="password",
                value=st.session_state.get("anthropic_key", ""),
                help="For analysis and product matching",
            )
            if anthropic_key:
                st.session_state.anthropic_key = anthropic_key

        with col2:
            openai_key = st.text_input(
                "OpenAI API Key (Required for design images)",
                type="password",
                value=st.session_state.get("openai_key", ""),
                help="For DALL-E 3 image generation",
            )
            if openai_key:
                st.session_state.openai_key = openai_key

        if anthropic_key and openai_key:
            st.success("✅ Both API keys configured!")
        elif anthropic_key:
            st.warning("⚠️ Add OpenAI key for AI-generated design images")

    st.divider()

    # Upload
    tab1, tab2 = st.tabs(["📷 Photos", "🎯 3D Scan"])

    uploaded_images = []
    uploaded_3d = None

    with tab1:
        uploaded_images = st.file_uploader(
            "Upload room photos",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            key="img_upload",
        )

        if uploaded_images:
            cols = st.columns(4)
            for i, f in enumerate(uploaded_images):
                with cols[i % 4]:
                    f.seek(0)
                    st.image(Image.open(f), caption=f.name, use_column_width=True)

    with tab2:
        uploaded_3d = st.file_uploader(
            "Upload GLB file", type=["glb", "gltf"], key="3d_upload"
        )

        if uploaded_3d:
            uploaded_3d.seek(0)
            file_bytes = uploaded_3d.read()
            glb_base64 = base64.b64encode(file_bytes).decode("utf-8")
            metadata, _ = extract_glb_metadata(file_bytes)

            st.success(f"✅ {uploaded_3d.name} ({len(file_bytes)/1024/1024:.1f} MB)")

            if metadata:
                cols = st.columns(3)
                cols[0].metric("Meshes", metadata.get("meshes", 0))
                cols[1].metric("Materials", metadata.get("materials", 0))
                cols[2].metric("Textures", metadata.get("textures", 0))

            st.subheader("🎮 3D Preview")
            components.html(create_3d_viewer_html(glb_base64, 400), height=430)

            st.session_state.project_state["glb_base64"] = glb_base64
            st.session_state.project_state["scan_metadata"] = metadata

    st.divider()

    # Preferences
    st.subheader("Preferences")
    col1, col2 = st.columns(2)

    with col1:
        budget = st.select_slider(
            "Budget",
            ["< $5K", "$5K-$15K", "$15K-$30K", "$30K-$50K", "$50K-$100K", "> $100K"],
            value="$15K-$30K",
        )
        style = st.selectbox(
            "Design Style",
            [
                "Modern Minimalist",
                "Contemporary",
                "Scandinavian",
                "Industrial",
                "Mid-Century Modern",
                "Traditional",
                "Bohemian",
                "Coastal",
                "Farmhouse",
            ],
        )

    with col2:
        goals = st.multiselect(
            "Goals",
            [
                "Increase value",
                "Improve function",
                "Update style",
                "Fix issues",
                "Energy efficiency",
            ],
            default=["Update style"],
        )
        priorities = st.multiselect(
            "Priority Areas",
            ["Kitchen", "Bathroom", "Living Room", "Bedroom", "Flooring", "Lighting"],
            default=["Living Room"],
        )

    st.divider()

    has_keys = bool(st.session_state.get("anthropic_key"))
    has_content = bool(uploaded_images or uploaded_3d)

    if st.button(
        "🚀 Start Analysis", type="primary", disabled=not has_keys or not has_content
    ):
        images = []
        for f in uploaded_images or []:
            result = process_uploaded_image(f)
            if result["success"]:
                images.append(result)

        if uploaded_3d:
            st.session_state.project_state["has_3d_scan"] = True

        st.session_state.project_state["images"] = images
        st.session_state.project_state["preferences"] = {
            "budget": budget,
            "style": style,
            "goals": ", ".join(goals),
            "priorities": ", ".join(priorities),
        }
        st.session_state.project_state["phase"] = "valuation"
        st.rerun()


def render_valuation_phase():
    st.header("📊 Property Valuation")

    if not st.session_state.project_state["valuation"]:
        with st.spinner("🔍 Analyzing property..."):
            valuation = run_valuation_agent(
                st.session_state.project_state["images"],
                st.session_state.project_state["preferences"],
                st.session_state.project_state.get("scan_metadata"),
            )
            st.session_state.project_state["valuation"] = valuation
            st.rerun()

    valuation = st.session_state.project_state["valuation"]

    if "error" in valuation:
        st.error(valuation["error"])
        if st.button("🔄 Retry"):
            st.session_state.project_state["valuation"] = None
            st.rerun()
        return

    # Display
    cost = valuation.get("cost_estimate", {})
    col1, col2, col3 = st.columns(3)
    col1.metric("Low", f"${cost.get('low_estimate', 0):,.0f}")
    col2.metric("Mid", f"${cost.get('mid_estimate', 0):,.0f}")
    col3.metric("High", f"${cost.get('high_estimate', 0):,.0f}")

    assessment = valuation.get("property_assessment", {})
    with st.expander("🏠 Assessment", expanded=True):
        st.write(f"**Room Type:** {assessment.get('room_type', 'N/A')}")
        st.write(f"**Condition:** {assessment.get('current_condition', 'N/A')}")
        st.write(f"**Size:** {assessment.get('square_footage_estimate', 'N/A')}")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("👍 Continue to Design", type="primary"):
            st.session_state.project_state["phase"] = "design"
            st.rerun()
    with col2:
        if st.button("🔄 Start Over"):
            for key in list(st.session_state.project_state.keys()):
                if key != "phase":
                    st.session_state.project_state[key] = (
                        [] if "images" in key else None
                    )
            st.session_state.project_state["phase"] = "upload"
            st.rerun()


def render_design_phase():
    st.header("🎨 AI-Generated Design Options")

    # Generate design concepts first
    if not st.session_state.project_state["designs"]:
        with st.spinner("🎨 Creating design concepts..."):
            designs = run_design_agent_with_images(
                st.session_state.project_state["images"],
                st.session_state.project_state["preferences"],
                st.session_state.project_state["valuation"],
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

    # Generate images if not already done
    if not st.session_state.project_state.get("design_images"):
        if st.session_state.get("openai_key"):
            with st.spinner(
                "🖼️ Generating design images with DALL-E 3... (this may take 30-60 seconds)"
            ):
                design_images = generate_design_images(
                    st.session_state.project_state["valuation"],
                    st.session_state.project_state["preferences"],
                )
                st.session_state.project_state["design_images"] = design_images
                st.rerun()
        else:
            st.warning("⚠️ Add OpenAI API key to generate design images")

    design_images = st.session_state.project_state.get("design_images", {})
    design_options = designs.get("design_options", [])

    # Display designs
    for i, opt in enumerate(design_options):
        opt_num = opt.get("option_number", i + 1)

        with st.container():
            st.subheader(f"Option {opt_num}: {opt.get('name', 'Design')}")

            col1, col2 = st.columns([2, 1])

            with col1:
                # Show generated image if available
                img_data = design_images.get(opt_num, {})
                if img_data.get("image_base64"):
                    img_bytes = base64.b64decode(img_data["image_base64"])
                    st.image(
                        img_bytes,
                        caption=f"AI-Generated: {opt.get('name')}",
                        use_column_width=True,
                    )
                elif img_data.get("error"):
                    st.error(f"Image generation failed: {img_data['error']}")
                else:
                    st.info("🖼️ Image not generated (add OpenAI key)")
                    # Show color palette as fallback
                    palette = opt.get("color_palette", {})
                    pcols = st.columns(3)
                    pcols[0].color_picker(
                        "Primary", palette.get("primary", "#000"), disabled=True
                    )
                    pcols[1].color_picker(
                        "Secondary", palette.get("secondary", "#666"), disabled=True
                    )
                    pcols[2].color_picker(
                        "Accent", palette.get("accent", "#999"), disabled=True
                    )

            with col2:
                st.write(f"**Concept:** {opt.get('concept', 'N/A')}")
                st.write(f"**Mood:** {opt.get('mood', 'N/A')}")
                st.metric("Estimated Cost", f"${opt.get('estimated_cost', 0):,.0f}")

                st.write("**Key Furniture:**")
                for item in opt.get("key_furniture", [])[:5]:
                    st.write(f"• {item}")

            if st.button(
                f"✓ Select Option {opt_num}", key=f"sel_{opt_num}", type="primary"
            ):
                st.session_state.project_state["selected_design"] = opt
                st.session_state.project_state["selected_design_image"] = img_data.get(
                    "image_base64"
                )
                st.success(f"✅ Option {opt_num} selected!")

            st.divider()

    # Continue button
    if st.session_state.project_state.get("selected_design"):
        st.success(
            f"✓ Selected: {st.session_state.project_state['selected_design'].get('name')}"
        )
        if st.button("🛋️ Find Matching Products", type="primary"):
            st.session_state.project_state["phase"] = "products"
            st.rerun()

    if st.button("← Back to Valuation"):
        st.session_state.project_state["phase"] = "valuation"
        st.rerun()


def render_products_phase():
    st.header("🛋️ Real Product Matching")

    selected_image = st.session_state.project_state.get("selected_design_image")

    if not selected_image:
        st.warning("No design image available for product matching")
        if st.button("← Back to Design"):
            st.session_state.project_state["phase"] = "design"
            st.rerun()
        return

    # Segment furniture if not done
    if not st.session_state.project_state.get("furniture_items"):
        with st.spinner("🔍 Identifying furniture in design..."):
            furniture = segment_furniture_from_design(
                selected_image,
                st.session_state.project_state["selected_design"].get("name", "Design"),
            )
            st.session_state.project_state["furniture_items"] = furniture
            st.rerun()

    furniture_items = st.session_state.project_state.get("furniture_items", [])

    if not furniture_items:
        st.warning("Could not identify furniture items")
        if st.button("🔄 Retry"):
            st.session_state.project_state["furniture_items"] = []
            st.rerun()
        return

    # Show design image with identified items
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Selected Design")
        img_bytes = base64.b64decode(selected_image)
        st.image(img_bytes, use_column_width=True)

    with col2:
        st.subheader(f"Identified Items ({len(furniture_items)})")
        for item in furniture_items:
            with st.expander(
                f"🪑 {item.get('name', 'Item')} - {item.get('position', '')}",
                expanded=False,
            ):
                st.write(f"**Type:** {item.get('item_type', 'N/A')}")
                st.write(f"**Style:** {item.get('style', 'N/A')}")
                st.write(f"**Color:** {item.get('color', 'N/A')}")
                st.write(f"**Material:** {item.get('material', 'N/A')}")
                st.write(
                    f"**Est. Price:** ${item.get('estimated_price_low', 0):,} - ${item.get('estimated_price_high', 0):,}"
                )

    st.divider()
    st.subheader("🔗 Real Product Matches")

    # Find products for each item
    product_matches = st.session_state.project_state.get("product_matches", {})

    for item in furniture_items:
        item_id = str(item.get("id", 0))

        with st.expander(f"🛒 {item.get('name', 'Item')}", expanded=True):
            # Find matches if not already done
            if item_id not in product_matches:
                with st.spinner(f"Finding products for {item.get('name')}..."):
                    matches = find_real_products(item)
                    product_matches[item_id] = matches
                    st.session_state.project_state["product_matches"] = product_matches

            matches = product_matches.get(item_id, [])

            if matches:
                cols = st.columns(len(matches))
                for j, match in enumerate(matches):
                    with cols[j]:
                        st.write(f"**{match.get('product_name', 'Product')}**")
                        st.write(f"🏪 {match.get('retailer', 'N/A')}")
                        st.write(f"💰 ${match.get('price', 0):,}")
                        st.write(f"📏 {match.get('dimensions', 'N/A')}")
                        st.write(f"🎯 {match.get('match_score', 0)}% match")

                        url = match.get("url", "#")
                        st.markdown(f"[🔗 View Product]({url})")

                        if st.button(f"✓ Select", key=f"prod_{item_id}_{j}"):
                            if (
                                "selected_products"
                                not in st.session_state.project_state
                            ):
                                st.session_state.project_state["selected_products"] = {}
                            st.session_state.project_state["selected_products"][
                                item_id
                            ] = match
                            st.success("Selected!")
            else:
                st.info("No matches found - searching...")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📦 Generate Final BOM", type="primary"):
            st.session_state.project_state["phase"] = "bom"
            st.rerun()
    with col2:
        if st.button("← Back to Design"):
            st.session_state.project_state["phase"] = "design"
            st.rerun()


def render_bom_phase():
    st.header("📦 Bill of Materials")

    if not st.session_state.project_state.get("bom"):
        with st.spinner("📦 Generating BOM..."):
            bom = run_procurement_agent(
                st.session_state.project_state.get("selected_design", {}),
                st.session_state.project_state.get("furniture_items", []),
                st.session_state.project_state.get("product_matches", {}),
            )
            st.session_state.project_state["bom"] = bom
            st.rerun()

    bom = st.session_state.project_state["bom"]

    if "error" in bom:
        st.error(bom["error"])
        if st.button("🔄 Retry"):
            st.session_state.project_state["bom"] = None
            st.rerun()
        return

    # Summary
    summary = bom.get("total_summary", {})
    col1, col2, col3 = st.columns(3)
    col1.metric("Products", f"${summary.get('products_total', 0):,.0f}")
    col2.metric("Labor", f"${summary.get('labor_total', 0):,.0f}")
    col3.metric("Total", f"${summary.get('grand_total', 0):,.0f}")

    st.divider()

    # Categories
    for cat in bom.get("bill_of_materials", {}).get("categories", []):
        with st.expander(
            f"📁 {cat.get('category_name')} - ${cat.get('category_total', 0):,.0f}"
        ):
            for item in cat.get("items", []):
                col1, col2, col3 = st.columns([3, 1, 1])
                col1.write(
                    f"**{item.get('item_name')}** ({item.get('retailer', 'N/A')})"
                )
                col2.write(f"${item.get('unit_price', 0):,.0f}")
                if item.get("url"):
                    col3.markdown(f"[🔗 Link]({item.get('url')})")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Complete", type="primary"):
            st.session_state.project_state["phase"] = "complete"
            st.rerun()
    with col2:
        if st.button("← Back to Products"):
            st.session_state.project_state["phase"] = "products"
            st.rerun()


def render_complete_phase():
    st.header("✅ Project Complete!")
    st.balloons()

    # Summary
    design = st.session_state.project_state.get("selected_design", {})
    bom = st.session_state.project_state.get("bom", {})

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Selected Design")
        if st.session_state.project_state.get("selected_design_image"):
            img_bytes = base64.b64decode(
                st.session_state.project_state["selected_design_image"]
            )
            st.image(img_bytes, use_column_width=True)
        st.write(f"**{design.get('name', 'Design')}**")

    with col2:
        st.subheader("Budget Summary")
        summary = bom.get("total_summary", {})
        st.metric("Total Project Cost", f"${summary.get('grand_total', 0):,.0f}")
        st.write(f"Products: ${summary.get('products_total', 0):,.0f}")
        st.write(f"Labor: ${summary.get('labor_total', 0):,.0f}")

    st.divider()

    # Export
    export_data = {
        "timestamp": datetime.now().isoformat(),
        "design": design,
        "furniture_items": st.session_state.project_state.get("furniture_items", []),
        "product_matches": st.session_state.project_state.get("product_matches", {}),
        "bom": bom,
    }

    st.download_button(
        "📥 Download Project",
        json.dumps(export_data, indent=2),
        f"renovation_{datetime.now().strftime('%Y%m%d')}.json",
        "application/json",
    )

    if st.button("🔄 New Project"):
        st.session_state.project_state = {
            "phase": "upload",
            "images": [],
            "preferences": {},
            "valuation": None,
            "designs": None,
            "design_images": {},
            "selected_design": None,
            "furniture_items": [],
            "product_matches": {},
            "bom": None,
            "gate_1_approved": False,
            "gate_2_approved": False,
            "has_3d_scan": False,
            "scan_metadata": None,
            "glb_base64": None,
        }
        st.rerun()


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


if __name__ == "__main__":
    main()
