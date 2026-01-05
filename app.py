"""
OmniRenovation AI - Phase 1 Pilot
With interactive 3D GLB viewer using Three.js
"""

import streamlit as st
import streamlit.components.v1 as components
import anthropic
import json
import base64
from datetime import datetime
from io import BytesIO
from PIL import Image
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
        "glb_base64": None,
    }


def create_3d_viewer_html(glb_base64: str, height: int = 500) -> str:
    """
    Create an HTML/JS viewer for GLB files using Three.js
    """
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
                font-size: 18px;
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
            // Base64 GLB data
            const glbBase64 = "{glb_base64}";
            
            // Convert base64 to ArrayBuffer
            function base64ToArrayBuffer(base64) {{
                const binaryString = atob(base64);
                const bytes = new Uint8Array(binaryString.length);
                for (let i = 0; i < binaryString.length; i++) {{
                    bytes[i] = binaryString.charCodeAt(i);
                }}
                return bytes.buffer;
            }}
            
            // Scene setup
            const container = document.getElementById('container');
            const scene = new THREE.Scene();
            
            // Camera
            const camera = new THREE.PerspectiveCamera(
                60, 
                container.clientWidth / container.clientHeight, 
                0.1, 
                1000
            );
            camera.position.set(5, 5, 5);
            
            // Renderer
            const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
            renderer.setSize(container.clientWidth, container.clientHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            renderer.outputEncoding = THREE.sRGBEncoding;
            renderer.toneMapping = THREE.ACESFilmicToneMapping;
            renderer.toneMappingExposure = 1;
            container.appendChild(renderer.domElement);
            
            // Controls
            const controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            controls.screenSpacePanning = true;
            controls.minDistance = 1;
            controls.maxDistance = 100;
            
            // Lighting
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
            scene.add(ambientLight);
            
            const directionalLight1 = new THREE.DirectionalLight(0xffffff, 0.8);
            directionalLight1.position.set(5, 10, 7);
            scene.add(directionalLight1);
            
            const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.4);
            directionalLight2.position.set(-5, 5, -5);
            scene.add(directionalLight2);
            
            // Grid helper
            const gridHelper = new THREE.GridHelper(20, 20, 0x444444, 0x222222);
            scene.add(gridHelper);
            
            // Load GLB
            const loader = new THREE.GLTFLoader();
            const arrayBuffer = base64ToArrayBuffer(glbBase64);
            
            loader.parse(arrayBuffer, '', 
                function(gltf) {{
                    const model = gltf.scene;
                    
                    // Calculate bounding box
                    const box = new THREE.Box3().setFromObject(model);
                    const center = box.getCenter(new THREE.Vector3());
                    const size = box.getSize(new THREE.Vector3());
                    
                    // Center the model
                    model.position.sub(center);
                    
                    // Scale to fit
                    const maxDim = Math.max(size.x, size.y, size.z);
                    const scale = 5 / maxDim;
                    model.scale.multiplyScalar(scale);
                    
                    scene.add(model);
                    
                    // Update camera
                    const scaledSize = size.multiplyScalar(scale);
                    const maxScaledDim = Math.max(scaledSize.x, scaledSize.y, scaledSize.z);
                    camera.position.set(maxScaledDim * 1.5, maxScaledDim * 1.5, maxScaledDim * 1.5);
                    controls.update();
                    
                    // Update info
                    let vertexCount = 0;
                    let triangleCount = 0;
                    model.traverse(function(child) {{
                        if (child.isMesh) {{
                            const geometry = child.geometry;
                            if (geometry.attributes.position) {{
                                vertexCount += geometry.attributes.position.count;
                            }}
                            if (geometry.index) {{
                                triangleCount += geometry.index.count / 3;
                            }} else if (geometry.attributes.position) {{
                                triangleCount += geometry.attributes.position.count / 3;
                            }}
                        }}
                    }});
                    
                    document.getElementById('info').innerHTML = 
                        `Vertices: ${{vertexCount.toLocaleString()}}<br>` +
                        `Triangles: ${{Math.floor(triangleCount).toLocaleString()}}<br>` +
                        `Size: ${{size.x.toFixed(2)}} x ${{size.y.toFixed(2)}} x ${{size.z.toFixed(2)}}`;
                    
                    // Hide loading
                    document.getElementById('loading').style.display = 'none';
                }},
                function(error) {{
                    console.error('Error loading GLB:', error);
                    document.getElementById('loading').innerHTML = 'Error loading model: ' + error.message;
                }}
            );
            
            // Animation loop
            function animate() {{
                requestAnimationFrame(animate);
                controls.update();
                renderer.render(scene, camera);
            }}
            animate();
            
            // Handle resize
            window.addEventListener('resize', function() {{
                camera.aspect = container.clientWidth / container.clientHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(container.clientWidth, container.clientHeight);
            }});
        </script>
    </body>
    </html>
    """
    return html


def extract_glb_metadata(file_bytes):
    """Extract basic metadata from GLB file."""
    try:
        magic = struct.unpack("<I", file_bytes[0:4])[0]
        if magic != 0x46546C67:
            return None, "Not a valid GLB file"

        version = struct.unpack("<I", file_bytes[4:8])[0]
        total_length = struct.unpack("<I", file_bytes[8:12])[0]

        chunk_length = struct.unpack("<I", file_bytes[12:16])[0]
        chunk_type = struct.unpack("<I", file_bytes[16:20])[0]

        if chunk_type != 0x4E4F534A:
            return None, "Invalid GLB structure"

        json_data = file_bytes[20 : 20 + chunk_length].decode("utf-8")
        gltf = json.loads(json_data)

        metadata = {
            "format": "GLB",
            "version": version,
            "file_size_mb": len(file_bytes) / (1024 * 1024),
            "meshes": len(gltf.get("meshes", [])),
            "materials": len(gltf.get("materials", [])),
            "textures": len(gltf.get("textures", [])),
            "nodes": len(gltf.get("nodes", [])),
            "scenes": len(gltf.get("scenes", [])),
            "animations": len(gltf.get("animations", [])),
        }

        mesh_names = [
            m.get("name", f"Mesh_{i}") for i, m in enumerate(gltf.get("meshes", []))
        ]
        metadata["mesh_names"] = mesh_names[:10]

        return metadata, None

    except Exception as e:
        return None, f"Error parsing GLB: {str(e)}"


def capture_3d_screenshot_html(glb_base64: str) -> str:
    """
    Create HTML that renders GLB and captures a screenshot.
    Returns image data via postMessage.
    """
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ margin: 0; }}
            canvas {{ display: block; }}
        </style>
    </head>
    <body>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
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
            
            // Setup
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0xf0f0f0);
            
            const camera = new THREE.PerspectiveCamera(60, 4/3, 0.1, 1000);
            
            const renderer = new THREE.WebGLRenderer({{ antialias: true, preserveDrawingBuffer: true }});
            renderer.setSize(800, 600);
            renderer.outputEncoding = THREE.sRGBEncoding;
            document.body.appendChild(renderer.domElement);
            
            // Lighting
            scene.add(new THREE.AmbientLight(0xffffff, 0.6));
            const light = new THREE.DirectionalLight(0xffffff, 0.8);
            light.position.set(5, 10, 7);
            scene.add(light);
            
            // Load and capture
            const loader = new THREE.GLTFLoader();
            loader.parse(base64ToArrayBuffer(glbBase64), '',
                function(gltf) {{
                    const model = gltf.scene;
                    
                    const box = new THREE.Box3().setFromObject(model);
                    const center = box.getCenter(new THREE.Vector3());
                    const size = box.getSize(new THREE.Vector3());
                    
                    model.position.sub(center);
                    
                    const maxDim = Math.max(size.x, size.y, size.z);
                    const scale = 5 / maxDim;
                    model.scale.multiplyScalar(scale);
                    
                    scene.add(model);
                    
                    // Position camera
                    camera.position.set(7, 5, 7);
                    camera.lookAt(0, 0, 0);
                    
                    // Render
                    renderer.render(scene, camera);
                    
                    // Capture
                    const imageData = renderer.domElement.toDataURL('image/png');
                    
                    // Send to parent
                    window.parent.postMessage({{
                        type: 'screenshot',
                        data: imageData
                    }}, '*');
                }}
            );
        </script>
    </body>
    </html>
    """
    return html


def process_uploaded_image(uploaded_file):
    """Process uploaded image and return base64 data."""
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
    """VALUATION AGENT"""
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

    metadata_section = ""
    if scan_metadata:
        metadata_section = f"""
3D SCAN METADATA:
- Format: {scan_metadata.get('format', 'GLB')}
- File Size: {scan_metadata.get('file_size_mb', 0):.2f} MB
- Meshes: {scan_metadata.get('meshes', 'N/A')}
- Materials: {scan_metadata.get('materials', 'N/A')}
- Textures: {scan_metadata.get('textures', 'N/A')}

This is a 3D scanned property. Use the rendered views for visual analysis.
"""

    prompt_text = f"""You are an expert renovation valuation agent. Analyze the property and provide assessment.
{metadata_section}
User Preferences:
- Budget: {preferences.get('budget', 'Not specified')}
- Goals: {preferences.get('goals', 'General renovation')}
- Style: {preferences.get('style', 'Modern')}
- Priorities: {preferences.get('priorities', 'Not specified')}

Return JSON:
{{
    "property_assessment": {{
        "room_types_identified": ["rooms"],
        "current_condition": "poor/fair/good/excellent",
        "condition_details": "description",
        "square_footage_estimate": "estimate",
        "age_estimate": "estimate"
    }},
    "renovation_scope": {{
        "recommended_work": [{{"area": "name", "work_needed": "desc", "priority": "high/medium/low"}}],
        "structural_concerns": ["concerns"],
        "quick_wins": ["wins"]
    }},
    "cost_estimate": {{
        "currency": "USD",
        "low_estimate": 0,
        "mid_estimate": 0,
        "high_estimate": 0,
        "breakdown": [{{"category": "name", "low": 0, "high": 0}}],
        "contingency_percentage": 15
    }},
    "roi_analysis": {{
        "estimated_value_increase": 0,
        "roi_percentage": 0,
        "payback_assessment": "description"
    }},
    "risk_assessment": {{
        "overall_risk": "low/medium/high",
        "risks": [{{"risk": "desc", "likelihood": "level", "mitigation": "suggestion"}}]
    }},
    "timeline_estimate": {{
        "minimum_weeks": 0,
        "maximum_weeks": 0,
        "phases": ["phases"]
    }}
}}"""

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

    except Exception as e:
        return {"error": str(e)}


def run_design_agent(images: list, preferences: dict, valuation: dict) -> dict:
    """DESIGN AGENT"""
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

    prompt_text = f"""You are an interior design expert. Create 3 design options based on:

VALUATION: {json.dumps(valuation, indent=2)}

PREFERENCES:
- Style: {preferences.get('style', 'Modern')}
- Budget: {preferences.get('budget', 'Mid-range')}
- Goals: {preferences.get('goals', 'Renovation')}

Return JSON:
{{
    "design_options": [
        {{
            "option_number": 1,
            "name": "Design Name",
            "style": "Style",
            "concept": "Brief description",
            "color_palette": {{
                "primary": "#hex",
                "secondary": "#hex",
                "accent": "#hex",
                "description": "Palette description"
            }},
            "key_features": ["feature1", "feature2"],
            "room_by_room": [
                {{"room": "name", "changes": ["change1"], "furniture_suggestions": ["item1"], "materials": ["material1"]}}
            ],
            "estimated_cost": 0,
            "pros": ["pro1"],
            "cons": ["con1"],
            "best_for": "Target user"
        }}
    ],
    "design_recommendations": {{
        "recommended_option": 1,
        "reasoning": "Why",
        "customization_suggestions": ["suggestion1"]
    }},
    "execution_notes": {{
        "suggested_order": ["phase1"],
        "diy_friendly_tasks": ["task1"],
        "professional_required": ["task1"]
    }}
}}"""

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

    except Exception as e:
        return {"error": str(e)}


def run_procurement_agent(designs: dict, selected_option: int) -> dict:
    """PROCUREMENT AGENT"""
    client = get_claude_client()
    if not client:
        return {"error": "No API key configured"}

    selected_design = None
    for opt in designs.get("design_options", []):
        if opt.get("option_number") == selected_option:
            selected_design = opt
            break

    if not selected_design:
        return {"error": "Design not found"}

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=8192,
            messages=[
                {
                    "role": "user",
                    "content": f"""Create a CONCISE BOM for:

DESIGN: {selected_design.get('name', 'Renovation')}
COST: ${selected_design.get('estimated_cost', 25000)}
FEATURES: {', '.join(selected_design.get('key_features', []))}

Return ONLY JSON (4-5 categories, 2-3 items each):
{{
    "bill_of_materials": {{
        "project_summary": {{"design_name": "name", "total_estimated_cost": 0, "number_of_items": 0, "number_of_categories": 0}},
        "categories": [
            {{
                "category_name": "Name",
                "category_total": 0,
                "items": [{{"item_name": "Item", "quantity": 0, "unit": "unit", "unit_price_low": 0, "unit_price_high": 0, "total_price_low": 0, "total_price_high": 0, "supplier": "Store"}}]
            }}
        ]
    }},
    "labor_estimates": [{{"trade": "Trade", "estimated_hours": 0, "hourly_rate_low": 0, "hourly_rate_high": 0, "total_low": 0, "total_high": 0}}],
    "total_summary": {{"materials_low": 0, "materials_high": 0, "labor_low": 0, "labor_high": 0, "contingency": 0, "grand_total_low": 0, "grand_total_high": 0}},
    "procurement_strategy": {{"recommended_approach": "Strategy", "order_sequence": ["First", "Second"]}}
}}""",
                }
            ],
        )

        response_text = response.content[0].text
        if "```" in response_text:
            response_text = response_text.split("```")[1].replace("json", "").strip()

        try:
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                return json.loads(response_text[json_start:json_end])
        except json.JSONDecodeError:
            pass

        return {"raw_response": response_text}

    except Exception as e:
        return {"error": str(e)}


# ============== UI ==============


def render_header():
    st.title("🏠 OmniRenovation AI")
    st.caption("AI-Native Renovation Platform - Phase 1 Pilot")

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
            help="Get key at console.anthropic.com",
        )
        if api_key:
            st.session_state.api_key = api_key
            st.success("✅ API key configured!")

    st.divider()

    # Upload tabs
    tab1, tab2 = st.tabs(["📷 Photos", "🎯 3D Scan (GLB)"])

    uploaded_images = []
    uploaded_3d = None

    with tab1:
        st.subheader("Property Photos")
        uploaded_images = st.file_uploader(
            "Upload photos of your property",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            key="img_upload",
        )

        if uploaded_images:
            cols = st.columns(4)
            for i, file in enumerate(uploaded_images):
                with cols[i % 4]:
                    file.seek(0)
                    try:
                        img = Image.open(file)
                        st.image(img, caption=file.name, use_column_width=True)
                    except:
                        st.error(f"Could not load {file.name}")

    with tab2:
        st.subheader("3D Scan File")
        st.info(
            "📱 Export from **Polycam**, **Matterport**, or **3D Scanner App** as GLB"
        )

        uploaded_3d = st.file_uploader(
            "Upload GLB/GLTF file",
            type=["glb", "gltf"],
            accept_multiple_files=False,
            key="3d_upload",
        )

        if uploaded_3d:
            st.success(
                f"✅ Loaded: {uploaded_3d.name} ({uploaded_3d.size / 1024 / 1024:.2f} MB)"
            )

            # Read and encode file
            uploaded_3d.seek(0)
            file_bytes = uploaded_3d.read()
            uploaded_3d.seek(0)

            glb_base64 = base64.b64encode(file_bytes).decode("utf-8")

            # Extract metadata
            metadata, error = extract_glb_metadata(file_bytes)
            if metadata:
                with st.expander("📊 File Info", expanded=True):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Meshes", metadata.get("meshes", 0))
                    with col2:
                        st.metric("Materials", metadata.get("materials", 0))
                    with col3:
                        st.metric("Textures", metadata.get("textures", 0))
                    with col4:
                        st.metric("Size", f"{metadata.get('file_size_mb', 0):.1f} MB")

            # 3D Viewer
            st.subheader("🎮 Interactive 3D Preview")
            st.caption("Drag to rotate • Scroll to zoom • Right-click to pan")

            viewer_html = create_3d_viewer_html(glb_base64, height=450)
            components.html(viewer_html, height=480, scrolling=False)

            # Store for later
            st.session_state.project_state["glb_base64"] = glb_base64
            st.session_state.project_state["scan_metadata"] = metadata

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
                "No Preference",
            ],
        )

    with col2:
        goals = st.multiselect(
            "Goals",
            [
                "Increase property value",
                "Improve functionality",
                "Update aesthetics",
                "Fix structural issues",
                "Energy efficiency",
                "Prepare for sale",
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
                "Flooring",
                "Lighting",
                "Paint/Walls",
                "Windows",
                "Exterior",
                "Storage",
            ],
            default=["Kitchen", "Bathroom(s)"],
        )

    notes = st.text_area("Additional Notes", placeholder="Any specific requirements...")

    st.divider()

    has_api_key = bool(
        st.secrets.get("ANTHROPIC_API_KEY") or st.session_state.get("api_key")
    )
    has_content = bool(uploaded_images or uploaded_3d)

    if st.button(
        "🚀 Start Analysis", type="primary", disabled=not has_content or not has_api_key
    ):
        images = []

        # Process images
        for file in uploaded_images or []:
            result = process_uploaded_image(file)
            if result["success"]:
                images.append(result)

        # For 3D, we'll use rendered views later or metadata only for now
        if uploaded_3d:
            st.session_state.project_state["has_3d_scan"] = True
            # Note: In production, you'd capture screenshots from the 3D viewer
            # For now, we proceed with metadata

        st.session_state.project_state["images"] = images
        st.session_state.project_state["preferences"] = {
            "budget": budget,
            "style": style,
            "goals": ", ".join(goals),
            "priorities": ", ".join(priorities),
            "notes": notes,
        }
        st.session_state.project_state["phase"] = "valuation"
        st.rerun()

    if not has_api_key:
        st.warning("⚠️ Configure API key above")
    if not has_content:
        st.info("📤 Upload photos or 3D scan to continue")


def render_valuation_phase():
    st.header("📊 Property Valuation")

    # Show 3D viewer if available
    if st.session_state.project_state.get("glb_base64"):
        with st.expander("🎮 3D Model", expanded=False):
            viewer_html = create_3d_viewer_html(
                st.session_state.project_state["glb_base64"], height=350
            )
            components.html(viewer_html, height=380)

    # Show metadata
    if st.session_state.project_state.get("scan_metadata"):
        meta = st.session_state.project_state["scan_metadata"]
        cols = st.columns(4)
        with cols[0]:
            st.metric("Meshes", meta.get("meshes", "N/A"))
        with cols[1]:
            st.metric("Materials", meta.get("materials", "N/A"))
        with cols[2]:
            st.metric("Textures", meta.get("textures", "N/A"))
        with cols[3]:
            st.metric("File", f"{meta.get('file_size_mb', 0):.1f} MB")
        st.divider()

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
        st.error(f"Error: {valuation['error']}")
        if st.button("🔄 Retry"):
            st.session_state.project_state["valuation"] = None
            st.rerun()
        return

    if "raw_response" in valuation:
        st.warning("Parse error")
        st.text(valuation["raw_response"][:1500])
        if st.button("🔄 Retry"):
            st.session_state.project_state["valuation"] = None
            st.rerun()
        return

    # Cost metrics
    cost = valuation.get("cost_estimate", {})
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Low", f"${cost.get('low_estimate', 0):,.0f}")
    with col2:
        st.metric("Mid", f"${cost.get('mid_estimate', 0):,.0f}")
    with col3:
        st.metric("High", f"${cost.get('high_estimate', 0):,.0f}")

    st.divider()

    # Details
    assessment = valuation.get("property_assessment", {})
    with st.expander("🏠 Assessment", expanded=True):
        st.write(f"**Condition:** {assessment.get('current_condition', 'N/A')}")
        st.write(f"**Rooms:** {', '.join(assessment.get('room_types_identified', []))}")
        st.write(f"**Size:** {assessment.get('square_footage_estimate', 'N/A')}")
        st.write(f"**Details:** {assessment.get('condition_details', 'N/A')}")

    scope = valuation.get("renovation_scope", {})
    with st.expander("🔨 Recommended Work"):
        for work in scope.get("recommended_work", []):
            icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
                work.get("priority", ""), "⚪"
            )
            st.write(f"{icon} **{work.get('area')}:** {work.get('work_needed')}")

    roi = valuation.get("roi_analysis", {})
    with st.expander("📈 ROI"):
        st.write(f"**Value Increase:** ${roi.get('estimated_value_increase', 0):,.0f}")
        st.write(f"**ROI:** {roi.get('roi_percentage', 0)}%")

    timeline = valuation.get("timeline_estimate", {})
    with st.expander("📅 Timeline"):
        st.write(
            f"**Duration:** {timeline.get('minimum_weeks', 0)}-{timeline.get('maximum_weeks', 0)} weeks"
        )

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("👍 Approve & Continue", type="primary"):
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
                "glb_base64": None,
            }
            st.rerun()


def render_design_phase():
    st.header("🎨 Design Options")

    if not st.session_state.project_state["designs"]:
        with st.spinner("🎨 Generating designs..."):
            designs = run_design_agent(
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

    if "raw_response" in designs:
        st.warning("Parse error")
        if st.button("🔄 Retry"):
            st.session_state.project_state["designs"] = None
            st.rerun()
        return

    options = designs.get("design_options", [])
    if not options:
        st.warning("No designs generated")
        if st.button("🔄 Retry"):
            st.session_state.project_state["designs"] = None
            st.rerun()
        return

    tabs = st.tabs(
        [f"Option {o.get('option_number', i+1)}" for i, o in enumerate(options)]
    )

    for i, (tab, opt) in enumerate(zip(tabs, options)):
        with tab:
            st.subheader(opt.get("name", "Design"))

            col1, col2 = st.columns([2, 1])
            with col1:
                st.write(f"**Style:** {opt.get('style')}")
                st.write(f"**Concept:** {opt.get('concept')}")
                st.write("**Features:**")
                for f in opt.get("key_features", []):
                    st.write(f"• {f}")
            with col2:
                st.metric("Cost", f"${opt.get('estimated_cost', 0):,.0f}")
                st.write("**Pros:**")
                for p in opt.get("pros", []):
                    st.write(f"✅ {p}")

            if st.button(
                f"Select Option {opt.get('option_number', i+1)}",
                key=f"s{i}",
                type="primary",
            ):
                st.session_state.project_state["selected_design"] = opt.get(
                    "option_number", i + 1
                )
                st.success("Selected!")

    st.divider()

    if st.session_state.project_state.get("selected_design"):
        st.success(
            f"✓ Option {st.session_state.project_state['selected_design']} selected"
        )
        if st.button("📦 Continue to Procurement", type="primary"):
            st.session_state.project_state["phase"] = "procurement"
            st.rerun()

    if st.button("← Back"):
        st.session_state.project_state["phase"] = "valuation"
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
        st.error(bom["error"])
        if st.button("🔄 Retry"):
            st.session_state.project_state["bom"] = None
            st.rerun()
        return

    if "raw_response" in bom:
        st.warning("Parse error")
        if st.button("🔄 Retry"):
            st.session_state.project_state["bom"] = None
            st.rerun()
        return

    summary = bom.get("total_summary", {})
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total (Low)", f"${summary.get('grand_total_low', 0):,.0f}")
    with col2:
        st.metric("Total (High)", f"${summary.get('grand_total_high', 0):,.0f}")

    st.divider()

    for cat in bom.get("bill_of_materials", {}).get("categories", []):
        with st.expander(
            f"📁 {cat.get('category_name')} - ${cat.get('category_total', 0):,.0f}"
        ):
            for item in cat.get("items", []):
                st.write(
                    f"**{item.get('item_name')}** - {item.get('quantity')} {item.get('unit')}"
                )
                st.write(
                    f"${item.get('total_price_low', 0):,.0f} - ${item.get('total_price_high', 0):,.0f}"
                )
                if item.get("supplier"):
                    st.caption(f"Supplier: {item['supplier']}")
                st.divider()

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("👍 Approve", type="primary"):
            st.session_state.project_state["phase"] = "complete"
            st.rerun()
    with col2:
        if st.button("← Back"):
            st.session_state.project_state["phase"] = "design"
            st.rerun()


def render_complete_phase():
    st.header("✅ Complete!")
    st.success("Phase 1 finished!")
    st.balloons()

    valuation = st.session_state.project_state.get("valuation", {})
    bom = st.session_state.project_state.get("bom", {})

    cost = valuation.get("cost_estimate", {})
    summary = bom.get("total_summary", {})

    col1, col2 = st.columns(2)
    with col1:
        st.write(
            f"**Est. Cost:** ${cost.get('low_estimate', 0):,.0f} - ${cost.get('high_estimate', 0):,.0f}"
        )
        st.write(
            f"**3D Scan:** {'✅' if st.session_state.project_state.get('has_3d_scan') else '❌'}"
        )
    with col2:
        st.write(
            f"**Final Budget:** ${summary.get('grand_total_low', 0):,.0f} - ${summary.get('grand_total_high', 0):,.0f}"
        )

    st.divider()

    export_data = {
        "timestamp": datetime.now().isoformat(),
        "preferences": st.session_state.project_state.get("preferences"),
        "has_3d_scan": st.session_state.project_state.get("has_3d_scan"),
        "scan_metadata": st.session_state.project_state.get("scan_metadata"),
        "valuation": valuation,
        "selected_design": st.session_state.project_state.get("selected_design"),
        "bill_of_materials": bom,
    }

    st.download_button(
        "📥 Download JSON",
        json.dumps(export_data, indent=2),
        f"project_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        "application/json",
    )

    if st.button("🔄 New Project"):
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
    elif phase == "procurement":
        render_procurement_phase()
    elif phase == "complete":
        render_complete_phase()


if __name__ == "__main__":
    main()
