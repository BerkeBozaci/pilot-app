"""
OmniRenovation AI - Phase 1 Pilot
With 3D GLB/GLTF file support
"""

import streamlit as st
import anthropic
import json
import base64
from datetime import datetime
from io import BytesIO
from PIL import Image
import tempfile
import os
import struct

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
        "bom": None,
        "gate_1_approved": False,
        "gate_2_approved": False,
        "has_3d_scan": False,
        "scan_metadata": None,
    }

# Check for 3D libraries
TRIMESH_AVAILABLE = False
try:
    import trimesh
    import numpy as np

    TRIMESH_AVAILABLE = True
except ImportError:
    pass


def extract_glb_metadata(file_bytes):
    """
    Extract basic metadata from GLB file without full 3D library.
    GLB format: 12-byte header + JSON chunk + binary chunk
    """
    try:
        # GLB Header (12 bytes)
        magic = struct.unpack("<I", file_bytes[0:4])[0]
        if magic != 0x46546C67:  # 'glTF' in little-endian
            return None, "Not a valid GLB file"

        version = struct.unpack("<I", file_bytes[4:8])[0]
        total_length = struct.unpack("<I", file_bytes[8:12])[0]

        # JSON Chunk
        chunk_length = struct.unpack("<I", file_bytes[12:16])[0]
        chunk_type = struct.unpack("<I", file_bytes[16:20])[0]

        if chunk_type != 0x4E4F534A:  # 'JSON' in little-endian
            return None, "Invalid GLB structure"

        json_data = file_bytes[20 : 20 + chunk_length].decode("utf-8")
        gltf = json.loads(json_data)

        # Extract metadata
        metadata = {
            "format": "GLB",
            "version": version,
            "file_size_mb": len(file_bytes) / (1024 * 1024),
            "meshes": len(gltf.get("meshes", [])),
            "materials": len(gltf.get("materials", [])),
            "textures": len(gltf.get("textures", [])),
            "nodes": len(gltf.get("nodes", [])),
            "scenes": len(gltf.get("scenes", [])),
        }

        # Try to get mesh names
        mesh_names = [
            m.get("name", f"Mesh_{i}") for i, m in enumerate(gltf.get("meshes", []))
        ]
        metadata["mesh_names"] = mesh_names[:10]  # First 10

        # Try to extract primitive counts
        total_primitives = 0
        for mesh in gltf.get("meshes", []):
            total_primitives += len(mesh.get("primitives", []))
        metadata["primitives"] = total_primitives

        return metadata, None

    except Exception as e:
        return None, f"Error parsing GLB: {str(e)}"


def render_glb_with_trimesh(file_bytes, filename):
    """
    Render GLB file to multiple 2D views using trimesh.
    """
    if not TRIMESH_AVAILABLE:
        return None, None, "trimesh not available"

    try:
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        # Load the 3D model
        scene = trimesh.load(tmp_path)

        # Get mesh(es)
        if isinstance(scene, trimesh.Scene):
            if len(scene.geometry) > 0:
                meshes = list(scene.geometry.values())
                # Combine all meshes
                mesh = trimesh.util.concatenate(meshes)
            else:
                os.unlink(tmp_path)
                return None, None, "No geometry found in GLB file"
        else:
            mesh = scene

        # Extract detailed metadata
        bounds = mesh.bounds
        dimensions = bounds[1] - bounds[0]

        metadata = {
            "format": "GLB (processed)",
            "vertices": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)) if hasattr(mesh, "faces") else 0,
            "dimensions": {
                "width": float(dimensions[0]),
                "depth": float(dimensions[1]),
                "height": float(dimensions[2]),
            },
            "bounds": {
                "min": [float(x) for x in bounds[0]],
                "max": [float(x) for x in bounds[1]],
            },
            "is_watertight": bool(mesh.is_watertight),
            "volume": float(mesh.volume) if mesh.is_watertight else None,
            "surface_area": float(mesh.area) if hasattr(mesh, "area") else None,
        }

        # Generate renders
        images = []

        # Try to render views
        try:
            # Create a scene for rendering
            render_scene = trimesh.Scene(mesh)

            # Get scene bounds for camera positioning
            scene_bounds = render_scene.bounds
            scene_center = (scene_bounds[0] + scene_bounds[1]) / 2
            scene_size = np.max(scene_bounds[1] - scene_bounds[0])

            # Define camera angles
            angles = [
                ("Front", [0, -1, 0.3]),
                ("Back", [0, 1, 0.3]),
                ("Left", [-1, 0, 0.3]),
                ("Right", [1, 0, 0.3]),
                ("Top", [0, 0, 1]),
                ("Perspective", [1, -1, 0.5]),
            ]

            for name, direction in angles:
                try:
                    # Normalize direction
                    direction = np.array(direction)
                    direction = direction / np.linalg.norm(direction)

                    # Position camera
                    camera_distance = scene_size * 2
                    camera_pos = scene_center + direction * camera_distance

                    # Create transformation matrix (look at center)
                    render_scene.set_camera(
                        angles=[0, 0, 0], distance=camera_distance, center=scene_center
                    )

                    # Render to PNG
                    png_bytes = render_scene.save_image(
                        resolution=[800, 600], visible=True
                    )

                    if png_bytes and len(png_bytes) > 0:
                        img = Image.open(BytesIO(png_bytes))
                        images.append({"image": img, "name": name})

                except Exception as render_error:
                    # Individual render failed, continue with others
                    pass

        except Exception as e:
            # Rendering failed entirely - that's OK, we still have metadata
            pass

        # Clean up
        os.unlink(tmp_path)

        return images, metadata, None

    except Exception as e:
        return None, None, f"Error processing GLB: {str(e)}"


def process_3d_file(uploaded_file):
    """
    Process a 3D scan file (GLB, GLTF, OBJ, etc.)
    """
    uploaded_file.seek(0)
    file_bytes = uploaded_file.read()
    uploaded_file.seek(0)

    filename = uploaded_file.name.lower()

    # Always try to extract basic metadata first
    basic_metadata = None
    if filename.endswith(".glb"):
        basic_metadata, error = extract_glb_metadata(file_bytes)
        if error:
            st.warning(f"Basic parsing warning: {error}")

    # Try full processing with trimesh if available
    if TRIMESH_AVAILABLE:
        images, full_metadata, error = render_glb_with_trimesh(
            file_bytes, uploaded_file.name
        )

        if error:
            st.warning(f"3D processing: {error}")

        if full_metadata:
            metadata = full_metadata
        elif basic_metadata:
            metadata = basic_metadata
        else:
            metadata = {
                "format": "3D file",
                "note": "Could not extract detailed metadata",
            }

        # Convert images to base64
        processed_images = []
        if images:
            for img_data in images:
                img = img_data["image"]
                buffer = BytesIO()
                img.save(buffer, format="PNG")
                base64_data = base64.standard_b64encode(buffer.getvalue()).decode(
                    "utf-8"
                )
                processed_images.append(
                    {
                        "name": f"{uploaded_file.name} - {img_data['name']}",
                        "type": "image/png",
                        "data": base64_data,
                        "success": True,
                        "source": "3d_render",
                    }
                )

        return {
            "success": True,
            "images": processed_images,
            "metadata": metadata,
            "name": uploaded_file.name,
            "has_renders": len(processed_images) > 0,
        }

    else:
        # No trimesh - return basic metadata only
        if basic_metadata:
            return {
                "success": True,
                "images": [],
                "metadata": basic_metadata,
                "name": uploaded_file.name,
                "has_renders": False,
                "note": "Install trimesh for 3D rendering: pip install trimesh pyglet",
            }
        else:
            return {
                "success": False,
                "error": "Cannot process 3D file. Install trimesh: pip install trimesh pyglet",
                "name": uploaded_file.name,
            }


def process_uploaded_image(uploaded_file):
    """Process uploaded image and return base64 data with correct MIME type."""
    uploaded_file.seek(0)
    bytes_data = uploaded_file.read()
    uploaded_file.seek(0)

    try:
        img = Image.open(BytesIO(bytes_data))
        actual_format = img.format.lower() if img.format else "jpeg"

        format_to_mime = {
            "jpeg": "image/jpeg",
            "jpg": "image/jpeg",
            "png": "image/png",
            "gif": "image/gif",
            "webp": "image/webp",
        }

        mime_type = format_to_mime.get(actual_format, "image/jpeg")

        if actual_format not in ["jpeg", "jpg", "png", "gif", "webp"]:
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGB")
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=95)
            bytes_data = buffer.getvalue()
            mime_type = "image/jpeg"

        base64_data = base64.standard_b64encode(bytes_data).decode("utf-8")

        return {
            "name": uploaded_file.name,
            "type": mime_type,
            "data": base64_data,
            "success": True,
        }

    except Exception as e:
        return {"name": uploaded_file.name, "error": str(e), "success": False}


def get_claude_client():
    """Initialize Claude client"""
    api_key = st.secrets.get("ANTHROPIC_API_KEY") or st.session_state.get("api_key")
    if not api_key:
        return None
    return anthropic.Anthropic(api_key=api_key)


def run_valuation_agent(
    images: list, preferences: dict, scan_metadata: dict = None
) -> dict:
    """VALUATION AGENT - Analyzes images, estimates costs, assesses risks"""
    client = get_claude_client()
    if not client:
        return {"error": "No API key configured"}

    image_content = []
    for img in images:
        if img.get("success", True) and "data" in img:
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

    if not image_content:
        # No images but we have metadata - use text-only analysis
        if scan_metadata:
            image_content = []
        else:
            return {"error": "No valid images to analyze"}

    # Build metadata section
    metadata_section = ""
    if scan_metadata:
        dims = scan_metadata.get("dimensions", {})
        metadata_section = f"""
3D SCAN METADATA (from uploaded scan file):
- Format: {scan_metadata.get('format', 'Unknown')}
- Dimensions: {dims.get('width', 'N/A'):.2f} x {dims.get('depth', 'N/A'):.2f} x {dims.get('height', 'N/A'):.2f} units
- Vertices: {scan_metadata.get('vertices', 'N/A'):,}
- Faces: {scan_metadata.get('faces', 'N/A'):,}
- Meshes: {scan_metadata.get('meshes', 'N/A')}
- Is Watertight: {scan_metadata.get('is_watertight', 'Unknown')}
- Volume: {f"{scan_metadata.get('volume', 0):.2f} cubic units" if scan_metadata.get('volume') else 'Not calculated'}
- Surface Area: {f"{scan_metadata.get('surface_area', 0):.2f} square units" if scan_metadata.get('surface_area') else 'Not calculated'}

Use this 3D data for more accurate room size and layout estimates.
"""

    prompt_text = f"""You are an expert renovation valuation agent. Analyze the property {"images and " if image_content else ""}data and provide a comprehensive assessment.
{metadata_section}
User Preferences:
- Budget Range: {preferences.get('budget', 'Not specified')}
- Renovation Goals: {preferences.get('goals', 'General renovation')}
- Style Preference: {preferences.get('style', 'Modern')}
- Priority Areas: {preferences.get('priorities', 'Not specified')}

Please provide your analysis in the following JSON format:
{{
    "property_assessment": {{
        "room_types_identified": ["list of rooms"],
        "current_condition": "poor/fair/good/excellent",
        "condition_details": "detailed description",
        "square_footage_estimate": "estimated sq ft/m2",
        "age_estimate": "estimated property age"
    }},
    "renovation_scope": {{
        "recommended_work": [
            {{"area": "area name", "work_needed": "description", "priority": "high/medium/low"}}
        ],
        "structural_concerns": ["list any concerns"],
        "quick_wins": ["easy improvements with high impact"]
    }},
    "cost_estimate": {{
        "currency": "USD",
        "low_estimate": 0,
        "mid_estimate": 0,
        "high_estimate": 0,
        "breakdown": [
            {{"category": "category name", "low": 0, "high": 0}}
        ],
        "contingency_percentage": 15
    }},
    "roi_analysis": {{
        "estimated_value_increase": 0,
        "roi_percentage": 0,
        "payback_assessment": "description"
    }},
    "risk_assessment": {{
        "overall_risk": "low/medium/high",
        "risks": [
            {{"risk": "description", "likelihood": "low/medium/high", "mitigation": "suggestion"}}
        ]
    }},
    "timeline_estimate": {{
        "minimum_weeks": 0,
        "maximum_weeks": 0,
        "phases": ["list of phases"]
    }}
}}

Be realistic and detailed. Base costs on current US market rates."""

    if image_content:
        image_content.append({"type": "text", "text": prompt_text})
        messages = [{"role": "user", "content": image_content}]
    else:
        messages = [{"role": "user", "content": prompt_text}]

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=4096, messages=messages
        )

        response_text = response.content[0].text
        try:
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                return json.loads(response_text[json_start:json_end])
        except json.JSONDecodeError:
            pass

        return {"raw_response": response_text}

    except anthropic.BadRequestError as e:
        return {"error": f"API Error: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


def run_design_agent(images: list, preferences: dict, valuation: dict) -> dict:
    """DESIGN AGENT - Generates design options"""
    client = get_claude_client()
    if not client:
        return {"error": "No API key configured"}

    image_content = []
    for img in images:
        if img.get("success", True) and "data" in img:
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

    prompt_text = f"""You are an expert interior design agent. Based on the property {"images and " if image_content else ""}valuation below, create 3 distinct design options.

VALUATION SUMMARY:
{json.dumps(valuation, indent=2)}

USER PREFERENCES:
- Style: {preferences.get('style', 'Modern')}
- Budget: {preferences.get('budget', 'Mid-range')}
- Goals: {preferences.get('goals', 'General renovation')}

Please provide 3 design options in this JSON format:
{{
    "design_options": [
        {{
            "option_number": 1,
            "name": "Creative name for this design",
            "style": "Design style",
            "concept": "Brief concept description",
            "color_palette": {{
                "primary": "#hexcode",
                "secondary": "#hexcode",
                "accent": "#hexcode",
                "description": "Color palette description"
            }},
            "key_features": ["feature 1", "feature 2", "feature 3"],
            "room_by_room": [
                {{
                    "room": "room name",
                    "changes": ["change 1", "change 2"],
                    "furniture_suggestions": ["item 1", "item 2"],
                    "materials": ["material 1", "material 2"]
                }}
            ],
            "estimated_cost": 0,
            "pros": ["advantage 1", "advantage 2"],
            "cons": ["consideration 1", "consideration 2"],
            "best_for": "Who this design is best suited for"
        }}
    ],
    "design_recommendations": {{
        "recommended_option": 1,
        "reasoning": "Why this option is recommended",
        "customization_suggestions": ["suggestion 1", "suggestion 2"]
    }},
    "execution_notes": {{
        "suggested_order": ["phase 1", "phase 2"],
        "diy_friendly_tasks": ["task 1", "task 2"],
        "professional_required": ["task 1", "task 2"]
    }}
}}

Be creative but practical. Ensure designs are achievable within the budget constraints."""

    if image_content:
        image_content.append({"type": "text", "text": prompt_text})
        messages = [{"role": "user", "content": image_content}]
    else:
        messages = [{"role": "user", "content": prompt_text}]

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=4096, messages=messages
        )

        response_text = response.content[0].text
        try:
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                return json.loads(response_text[json_start:json_end])
        except json.JSONDecodeError:
            pass

        return {"raw_response": response_text}

    except anthropic.BadRequestError as e:
        return {"error": f"API Error: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


def run_procurement_agent(designs: dict, selected_option: int) -> dict:
    """PROCUREMENT AGENT - Creates BOM, finds suppliers"""
    client = get_claude_client()
    if not client:
        return {"error": "No API key configured"}

    selected_design = None
    for opt in designs.get("design_options", []):
        if opt.get("option_number") == selected_option:
            selected_design = opt
            break

    if not selected_design:
        return {"error": "Selected design not found"}

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=8192,
            messages=[
                {
                    "role": "user",
                    "content": f"""You are a procurement specialist. Create a CONCISE Bill of Materials for this design:

DESIGN: {selected_design.get('name', 'Renovation')}
STYLE: {selected_design.get('style', 'Modern')}
ESTIMATED COST: ${selected_design.get('estimated_cost', 25000)}
KEY FEATURES: {', '.join(selected_design.get('key_features', []))}

IMPORTANT: Keep the response SHORT. Include only 4-5 main categories with 2-3 items each.

Return ONLY valid JSON (no markdown, no explanation) in this exact format:
{{
    "bill_of_materials": {{
        "project_summary": {{
            "design_name": "{selected_design.get('name', 'Renovation')}",
            "total_estimated_cost": 0,
            "number_of_items": 0,
            "number_of_categories": 0
        }},
        "categories": [
            {{
                "category_name": "Category Name",
                "category_total": 0,
                "items": [
                    {{
                        "item_name": "Product name",
                        "quantity": 0,
                        "unit": "sq ft/pieces/etc",
                        "unit_price_low": 0,
                        "unit_price_high": 0,
                        "total_price_low": 0,
                        "total_price_high": 0,
                        "supplier": "Suggested store"
                    }}
                ]
            }}
        ]
    }},
    "labor_estimates": [
        {{
            "trade": "Trade name",
            "estimated_hours": 0,
            "hourly_rate_low": 0,
            "hourly_rate_high": 0,
            "total_low": 0,
            "total_high": 0
        }}
    ],
    "total_summary": {{
        "materials_low": 0,
        "materials_high": 0,
        "labor_low": 0,
        "labor_high": 0,
        "contingency": 0,
        "grand_total_low": 0,
        "grand_total_high": 0
    }},
    "procurement_strategy": {{
        "recommended_approach": "Brief strategy",
        "order_sequence": ["First items", "Second items"]
    }}
}}

Return ONLY the JSON, no other text.""",
                }
            ],
        )

        response_text = response.content[0].text

        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]

        response_text = response_text.strip()

        try:
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                return json.loads(response_text[json_start:json_end])
        except json.JSONDecodeError as e:
            return {
                "error": f"Failed to parse response: {str(e)}",
                "raw_response": response_text[:500],
            }

        return {"raw_response": response_text}

    except anthropic.BadRequestError as e:
        return {"error": f"API Error: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


# ============== UI COMPONENTS ==============


def render_header():
    st.title("🏠 OmniRenovation AI")
    st.caption("AI-Native Renovation & Asset Management Platform - Phase 1 Pilot")

    phases = ["📤 Upload", "📊 Valuation", "🎨 Design", "📦 Procurement", "✅ Complete"]
    phase_map = {
        "upload": 0,
        "valuation": 1,
        "design": 2,
        "procurement": 3,
        "complete": 4,
    }
    current = phase_map.get(st.session_state.project_state["phase"], 0)

    cols = st.columns(5)
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

    # API Key
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""

    with st.expander("🔑 API Configuration", expanded=not st.session_state.api_key):
        api_key = st.text_input(
            "Anthropic API Key",
            type="password",
            value=st.session_state.api_key,
            help="Enter your Claude API key. Get one at console.anthropic.com",
        )
        if api_key:
            st.session_state.api_key = api_key
            st.success("API key configured!")

    st.divider()

    # 3D library status
    if TRIMESH_AVAILABLE:
        st.success("✅ Full 3D support enabled (trimesh installed)")
    else:
        st.warning(
            "⚠️ Limited 3D support - metadata only. For full 3D rendering, install: `pip install trimesh numpy pyglet`"
        )

    st.divider()

    # Upload tabs
    tab1, tab2 = st.tabs(["📷 Photos", "🎯 3D Scan (GLB/OBJ)"])

    uploaded_images = []
    uploaded_3d = None

    with tab1:
        st.subheader("Property Photos")
        uploaded_images = st.file_uploader(
            "Upload photos of your property",
            type=["jpg", "jpeg", "png", "webp", "gif"],
            accept_multiple_files=True,
            key="image_uploader",
        )

        if uploaded_images:
            cols = st.columns(4)
            for i, file in enumerate(uploaded_images):
                with cols[i % 4]:
                    file.seek(0)
                    try:
                        img = Image.open(file)
                        st.image(img, caption=file.name, width="stretch")
                    except Exception as e:
                        st.error(f"Could not load {file.name}")

    with tab2:
        st.subheader("3D Scan File")
        st.info(
            """
        **Supported formats:** GLB, GLTF, OBJ, PLY, STL
        
        **Get 3D scans from:**
        - 📱 **Polycam** (iPhone/Android)
        - 📱 **Matterport** 
        - 📱 **3D Scanner App** (iPhone LiDAR)
        - 🖥️ **RealityCapture**
        - 🖥️ **Meshroom**
        
        Export as GLB for best results.
        """
        )

        uploaded_3d = st.file_uploader(
            "Upload a 3D scan file",
            type=["glb", "gltf", "obj", "ply", "stl"],
            accept_multiple_files=False,
            key="3d_uploader",
        )

        if uploaded_3d:
            st.success(
                f"✅ File loaded: {uploaded_3d.name} ({uploaded_3d.size / 1024 / 1024:.2f} MB)"
            )

            # Show quick metadata preview
            if uploaded_3d.name.lower().endswith(".glb"):
                uploaded_3d.seek(0)
                preview_meta, _ = extract_glb_metadata(uploaded_3d.read())
                uploaded_3d.seek(0)

                if preview_meta:
                    with st.expander("📊 Quick Preview", expanded=True):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Meshes", preview_meta.get("meshes", "N/A"))
                        with col2:
                            st.metric("Materials", preview_meta.get("materials", "N/A"))
                        with col3:
                            st.metric(
                                "File Size",
                                f"{preview_meta.get('file_size_mb', 0):.2f} MB",
                            )

    st.divider()

    # Preferences
    st.subheader("Your Preferences")

    col1, col2 = st.columns(2)

    with col1:
        budget = st.select_slider(
            "Budget Range",
            options=[
                "< $5,000",
                "$5,000 - $15,000",
                "$15,000 - $30,000",
                "$30,000 - $50,000",
                "$50,000 - $100,000",
                "> $100,000",
            ],
            value="$15,000 - $30,000",
        )

        style = st.selectbox(
            "Design Style Preference",
            [
                "Modern Minimalist",
                "Contemporary",
                "Scandinavian",
                "Industrial",
                "Mid-Century Modern",
                "Traditional",
                "Transitional",
                "Bohemian",
                "Coastal",
                "Farmhouse",
                "No Preference",
            ],
        )

    with col2:
        goals = st.multiselect(
            "Renovation Goals",
            [
                "Increase property value",
                "Improve functionality",
                "Update aesthetics",
                "Fix structural issues",
                "Increase energy efficiency",
                "Prepare for sale",
                "Personal enjoyment",
                "Accommodate family changes",
            ],
            default=["Update aesthetics", "Improve functionality"],
        )

        priorities = st.multiselect(
            "Priority Areas",
            [
                "Kitchen",
                "Bathroom(s)",
                "Living Room",
                "Bedroom(s)",
                "Flooring Throughout",
                "Lighting",
                "Paint/Walls",
                "Windows",
                "Exterior",
                "Storage",
            ],
            default=["Kitchen", "Bathroom(s)"],
        )

    additional_notes = st.text_area(
        "Additional Notes",
        placeholder="Any specific requirements, constraints, or wishes...",
    )

    st.divider()

    has_api_key = bool(
        st.secrets.get("ANTHROPIC_API_KEY") or st.session_state.get("api_key")
    )
    has_content = bool(uploaded_images or uploaded_3d)

    if st.button(
        "🚀 Start Analysis", type="primary", disabled=not has_content or not has_api_key
    ):
        images = []
        scan_metadata = None

        progress_bar = st.progress(0)
        status_text = st.empty()

        # Process 3D file
        if uploaded_3d:
            status_text.text(f"Processing 3D scan: {uploaded_3d.name}...")
            progress_bar.progress(0.3)

            result = process_3d_file(uploaded_3d)

            if result["success"]:
                if result.get("images"):
                    images.extend(result["images"])
                    st.success(f"✅ 3D scan: {len(result['images'])} renders generated")
                else:
                    st.info(
                        "ℹ️ 3D metadata extracted (no renders - trimesh not available)"
                    )

                scan_metadata = result["metadata"]
                st.session_state.project_state["has_3d_scan"] = True
                st.session_state.project_state["scan_metadata"] = scan_metadata
            else:
                st.error(
                    f"❌ 3D processing failed: {result.get('error', 'Unknown error')}"
                )

        # Process images
        total_images = len(uploaded_images) if uploaded_images else 0
        for i, file in enumerate(uploaded_images or []):
            status_text.text(f"Processing image {i+1}/{total_images}: {file.name}")
            progress_bar.progress(0.3 + (0.7 * (i + 1) / max(total_images, 1)))

            result = process_uploaded_image(file)
            if result["success"]:
                images.append(result)

        progress_bar.empty()
        status_text.empty()

        # Check if we have enough to proceed
        if images or scan_metadata:
            st.session_state.project_state["images"] = images
            st.session_state.project_state["preferences"] = {
                "budget": budget,
                "style": style,
                "goals": ", ".join(goals),
                "priorities": ", ".join(priorities),
                "notes": additional_notes,
            }
            st.session_state.project_state["phase"] = "valuation"
            st.rerun()
        else:
            st.error("No files could be processed. Please try different files.")

    if not has_api_key:
        st.warning("Please configure your Anthropic API key above to proceed.")
    if not has_content:
        st.info("Please upload at least one photo or 3D scan to proceed.")


def render_valuation_phase():
    st.header("📊 Property Valuation & Assessment")

    # Show 3D metadata
    if st.session_state.project_state.get("has_3d_scan"):
        metadata = st.session_state.project_state.get("scan_metadata", {})
        with st.expander("📐 3D Scan Data", expanded=True):
            col1, col2, col3, col4 = st.columns(4)

            dims = metadata.get("dimensions", {})
            if dims:
                with col1:
                    st.metric("Width", f"{dims.get('width', 0):.2f}")
                with col2:
                    st.metric("Depth", f"{dims.get('depth', 0):.2f}")
                with col3:
                    st.metric("Height", f"{dims.get('height', 0):.2f}")
                with col4:
                    st.metric("Vertices", f"{metadata.get('vertices', 0):,}")
            else:
                with col1:
                    st.metric("Meshes", metadata.get("meshes", "N/A"))
                with col2:
                    st.metric("Materials", metadata.get("materials", "N/A"))
                with col3:
                    st.metric("Textures", metadata.get("textures", "N/A"))
                with col4:
                    st.metric("File Size", f"{metadata.get('file_size_mb', 0):.2f} MB")

    if not st.session_state.project_state["valuation"]:
        with st.spinner("🔍 Analyzing your property... This may take a minute."):
            valuation = run_valuation_agent(
                st.session_state.project_state["images"],
                st.session_state.project_state["preferences"],
                st.session_state.project_state.get("scan_metadata"),
            )
            st.session_state.project_state["valuation"] = valuation
            st.rerun()

    valuation = st.session_state.project_state["valuation"]

    if "error" in valuation:
        st.error(f"Error: {valuation['error']}")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Try Again"):
                st.session_state.project_state["valuation"] = None
                st.rerun()
        with col2:
            if st.button("← Back to Upload"):
                st.session_state.project_state["phase"] = "upload"
                st.session_state.project_state["valuation"] = None
                st.rerun()
        return

    if "raw_response" in valuation:
        st.warning("Couldn't parse structured response. Raw output:")
        st.text(valuation["raw_response"][:2000])
        if st.button("🔄 Try Again"):
            st.session_state.project_state["valuation"] = None
            st.rerun()
        return

    # Display results
    col1, col2, col3 = st.columns(3)
    cost = valuation.get("cost_estimate", {})
    with col1:
        st.metric("Cost (Low)", f"${cost.get('low_estimate', 0):,.0f}")
    with col2:
        st.metric("Cost (Mid)", f"${cost.get('mid_estimate', 0):,.0f}")
    with col3:
        st.metric("Cost (High)", f"${cost.get('high_estimate', 0):,.0f}")

    st.divider()

    # Property Assessment
    assessment = valuation.get("property_assessment", {})
    with st.expander("🏠 Property Assessment", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Condition:** {assessment.get('current_condition', 'N/A')}")
            st.write(
                f"**Rooms:** {', '.join(assessment.get('room_types_identified', []))}"
            )
        with col2:
            st.write(
                f"**Est. Size:** {assessment.get('square_footage_estimate', 'N/A')}"
            )
            st.write(f"**Est. Age:** {assessment.get('age_estimate', 'N/A')}")
        st.write(f"**Details:** {assessment.get('condition_details', 'N/A')}")

    # Renovation Scope
    scope = valuation.get("renovation_scope", {})
    with st.expander("🔨 Recommended Work", expanded=True):
        for work in scope.get("recommended_work", []):
            priority_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
                work.get("priority", ""), "⚪"
            )
            st.write(
                f"{priority_color} **{work.get('area', 'N/A')}:** {work.get('work_needed', 'N/A')}"
            )

    # Cost Breakdown
    with st.expander("💰 Cost Breakdown"):
        for item in cost.get("breakdown", []):
            st.write(
                f"• **{item.get('category', 'N/A')}:** ${item.get('low', 0):,.0f} - ${item.get('high', 0):,.0f}"
            )

    # ROI
    roi = valuation.get("roi_analysis", {})
    with st.expander("� ROI Analysis"):
        st.write(f"**Value Increase:** ${roi.get('estimated_value_increase', 0):,.0f}")
        st.write(f"**ROI:** {roi.get('roi_percentage', 0)}%")

    # Timeline
    timeline = valuation.get("timeline_estimate", {})
    with st.expander("📅 Timeline"):
        st.write(
            f"**Duration:** {timeline.get('minimum_weeks', 0)} - {timeline.get('maximum_weeks', 0)} weeks"
        )

    st.divider()

    st.subheader("✅ Approval Gate 1")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👍 Approve & Continue to Design", type="primary"):
            st.session_state.project_state["gate_1_approved"] = True
            st.session_state.project_state["phase"] = "design"
            st.rerun()
    with col2:
        if st.button("🔄 Start Over"):
            st.session_state.project_state = {
                "phase": "upload",
                "images": [],
                "preferences": {},
                "valuation": None,
                "designs": None,
                "bom": None,
                "gate_1_approved": False,
                "gate_2_approved": False,
                "has_3d_scan": False,
                "scan_metadata": None,
            }
            st.rerun()


def render_design_phase():
    st.header("🎨 Design Options")

    if not st.session_state.project_state["designs"]:
        with st.spinner("🎨 Generating design options..."):
            designs = run_design_agent(
                st.session_state.project_state["images"],
                st.session_state.project_state["preferences"],
                st.session_state.project_state["valuation"],
            )
            st.session_state.project_state["designs"] = designs
            st.rerun()

    designs = st.session_state.project_state["designs"]

    if "error" in designs:
        st.error(f"Error: {designs['error']}")
        if st.button("🔄 Try Again"):
            st.session_state.project_state["designs"] = None
            st.rerun()
        return

    if "raw_response" in designs:
        st.warning("Couldn't parse response")
        st.text(designs["raw_response"][:1000])
        if st.button("🔄 Try Again"):
            st.session_state.project_state["designs"] = None
            st.rerun()
        return

    design_options = designs.get("design_options", [])

    if not design_options:
        st.warning("No designs generated")
        if st.button("🔄 Regenerate"):
            st.session_state.project_state["designs"] = None
            st.rerun()
        return

    tabs = st.tabs(
        [
            f"Option {opt.get('option_number', i+1)}: {opt.get('name', 'Design')}"
            for i, opt in enumerate(design_options)
        ]
    )

    for i, (tab, opt) in enumerate(zip(tabs, design_options)):
        with tab:
            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader(opt.get("name", "Design"))
                st.write(f"**Style:** {opt.get('style', 'N/A')}")
                st.write(f"**Concept:** {opt.get('concept', 'N/A')}")

                palette = opt.get("color_palette", {})
                st.write("**Colors:**")
                pcols = st.columns(3)
                for j, (name, val) in enumerate(
                    [
                        ("Primary", palette.get("primary", "#000")),
                        ("Secondary", palette.get("secondary", "#666")),
                        ("Accent", palette.get("accent", "#999")),
                    ]
                ):
                    with pcols[j]:
                        st.color_picker(name, val, disabled=True, key=f"c_{i}_{j}")

                for feat in opt.get("key_features", []):
                    st.write(f"• {feat}")

            with col2:
                st.metric("Cost", f"${opt.get('estimated_cost', 0):,.0f}")
                st.write("**Pros:**")
                for pro in opt.get("pros", []):
                    st.write(f"✅ {pro}")
                st.write("**Cons:**")
                for con in opt.get("cons", []):
                    st.write(f"⚠️ {con}")

            if st.button(
                f"✓ Select Option {opt.get('option_number', i+1)}",
                key=f"sel_{i}",
                type="primary",
            ):
                st.session_state.project_state["selected_design"] = opt.get(
                    "option_number", i + 1
                )
                st.success(f"Option {opt.get('option_number', i+1)} selected!")

    st.divider()

    selected = st.session_state.project_state.get("selected_design")
    if selected:
        st.success(f"✓ Selected Option {selected}")
        if st.button("📦 Proceed to Procurement", type="primary"):
            st.session_state.project_state["phase"] = "procurement"
            st.rerun()
    else:
        st.info("Select a design option above")

    if st.button("← Back"):
        st.session_state.project_state["phase"] = "valuation"
        st.session_state.project_state["designs"] = None
        st.rerun()


def render_procurement_phase():
    st.header("📦 Bill of Materials")

    if not st.session_state.project_state.get("bom"):
        with st.spinner("📦 Generating BOM..."):
            bom = run_procurement_agent(
                st.session_state.project_state["designs"],
                st.session_state.project_state.get("selected_design", 1),
            )
            st.session_state.project_state["bom"] = bom
            st.rerun()

    bom = st.session_state.project_state["bom"]

    if "error" in bom:
        st.error(f"Error: {bom['error']}")
        if st.button("🔄 Try Again"):
            st.session_state.project_state["bom"] = None
            st.rerun()
        return

    if "raw_response" in bom:
        st.warning("Couldn't parse response")
        if st.button("🔄 Try Again"):
            st.session_state.project_state["bom"] = None
            st.rerun()
        return

    # Summary
    summary = bom.get("total_summary", {})
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total (Low)", f"${summary.get('grand_total_low', 0):,.0f}")
    with col2:
        st.metric("Total (High)", f"${summary.get('grand_total_high', 0):,.0f}")

    st.divider()

    # Categories
    bill = bom.get("bill_of_materials", {})
    for category in bill.get("categories", []):
        with st.expander(
            f"📁 {category.get('category_name', 'Category')} - ${category.get('category_total', 0):,.0f}"
        ):
            for item in category.get("items", []):
                st.write(f"**{item.get('item_name', 'Item')}**")
                st.write(f"Qty: {item.get('quantity', 0)} {item.get('unit', '')}")
                st.write(
                    f"${item.get('total_price_low', 0):,.0f} - ${item.get('total_price_high', 0):,.0f}"
                )
                if item.get("supplier"):
                    st.write(f"Supplier: {item['supplier']}")
                st.divider()

    # Labor
    with st.expander("👷 Labor"):
        for labor in bom.get("labor_estimates", []):
            st.write(
                f"**{labor.get('trade', '')}:** {labor.get('estimated_hours', 0)}hrs @ ${labor.get('hourly_rate_low', 0)}-${labor.get('hourly_rate_high', 0)}/hr"
            )

    st.divider()

    st.subheader("✅ Approval Gate 2")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("👍 Approve", type="primary"):
            st.session_state.project_state["gate_2_approved"] = True
            st.session_state.project_state["phase"] = "complete"
            st.rerun()
    with col2:
        if st.button("← Back"):
            st.session_state.project_state["phase"] = "design"
            st.session_state.project_state["bom"] = None
            st.rerun()
    with col3:
        if st.button("🔄 Start Over"):
            st.session_state.project_state = {
                "phase": "upload",
                "images": [],
                "preferences": {},
                "valuation": None,
                "designs": None,
                "bom": None,
                "gate_1_approved": False,
                "gate_2_approved": False,
                "has_3d_scan": False,
                "scan_metadata": None,
            }
            st.rerun()


def render_complete_phase():
    st.header("✅ Phase 1 Complete!")
    st.success("Congratulations! You've completed Phase 1.")
    st.balloons()

    valuation = st.session_state.project_state.get("valuation", {})
    bom = st.session_state.project_state.get("bom", {})

    col1, col2 = st.columns(2)
    with col1:
        cost = valuation.get("cost_estimate", {})
        st.write(
            f"**Renovation Cost:** ${cost.get('low_estimate', 0):,.0f} - ${cost.get('high_estimate', 0):,.0f}"
        )
        if st.session_state.project_state.get("has_3d_scan"):
            st.write("**3D Scan:** ✅ Used")
    with col2:
        summary = bom.get("total_summary", {})
        st.write(
            f"**Final Budget:** ${summary.get('grand_total_low', 0):,.0f} - ${summary.get('grand_total_high', 0):,.0f}"
        )

    st.divider()

    # Export
    export_data = {
        "timestamp": datetime.now().isoformat(),
        "preferences": st.session_state.project_state.get("preferences", {}),
        "has_3d_scan": st.session_state.project_state.get("has_3d_scan", False),
        "scan_metadata": st.session_state.project_state.get("scan_metadata"),
        "valuation": valuation,
        "selected_design": st.session_state.project_state.get("selected_design"),
        "bill_of_materials": bom,
    }

    st.download_button(
        "📥 Download Project JSON",
        data=json.dumps(export_data, indent=2),
        file_name=f"omnirenovation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
    )

    if st.button("🔄 Start New Project"):
        st.session_state.project_state = {
            "phase": "upload",
            "images": [],
            "preferences": {},
            "valuation": None,
            "designs": None,
            "bom": None,
            "gate_1_approved": False,
            "gate_2_approved": False,
            "has_3d_scan": False,
            "scan_metadata": None,
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
    elif phase == "procurement":
        render_procurement_phase()
    elif phase == "complete":
        render_complete_phase()


if __name__ == "__main__":
    main()
