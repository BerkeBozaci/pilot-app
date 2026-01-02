"""
OmniRenovation AI - Phase 1 Pilot
Fast prototype for testing the core agent workflow
"""

import streamlit as st
import anthropic
import json
import base64
from datetime import datetime
from io import BytesIO
from PIL import Image

# Page config
st.set_page_config(page_title="OmniRenovation AI", page_icon="🏠", layout="wide")

# Initialize session state
if "project_state" not in st.session_state:
    st.session_state.project_state = {
        "phase": "upload",  # upload -> valuation -> design -> procurement -> complete
        "images": [],
        "preferences": {},
        "valuation": None,
        "designs": None,
        "bom": None,
        "gate_1_approved": False,
        "gate_2_approved": False,
    }


def process_uploaded_image(uploaded_file):
    """
    Process uploaded image and return base64 data with correct MIME type.
    Uses PIL to detect actual image format regardless of file extension.
    """
    # Reset file pointer
    uploaded_file.seek(0)
    bytes_data = uploaded_file.read()
    uploaded_file.seek(0)

    # Use PIL to detect actual image format
    try:
        img = Image.open(BytesIO(bytes_data))
        actual_format = img.format.lower() if img.format else "jpeg"

        # Map PIL format to MIME type
        format_to_mime = {
            "jpeg": "image/jpeg",
            "jpg": "image/jpeg",
            "png": "image/png",
            "gif": "image/gif",
            "webp": "image/webp",
        }

        mime_type = format_to_mime.get(actual_format, "image/jpeg")

        # If format doesn't match what Claude expects, convert to JPEG
        if actual_format not in ["jpeg", "jpg", "png", "gif", "webp"]:
            # Convert to RGB (in case of RGBA) and save as JPEG
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGB")

            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=95)
            bytes_data = buffer.getvalue()
            mime_type = "image/jpeg"

        # Encode to base64
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


def run_valuation_agent(images: list, preferences: dict) -> dict:
    """
    VALUATION AGENT
    Analyzes images, estimates costs, assesses risks
    """
    client = get_claude_client()
    if not client:
        return {"error": "No API key configured"}

    # Build image content for Claude
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
        return {"error": "No valid images to analyze"}

    image_content.append(
        {
            "type": "text",
            "text": f"""You are an expert renovation valuation agent. Analyze these room/property images and provide a comprehensive assessment.

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

Be realistic and detailed in your estimates. Base costs on current US market rates but note that costs vary by location.""",
        }
    )

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": image_content}],
        )

        # Parse JSON from response
        response_text = response.content[0].text
        try:
            # Try to extract JSON from the response
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
    """
    DESIGN AGENT
    Generates design options based on valuation and preferences
    """
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
        return {"error": "No valid images to analyze"}

    image_content.append(
        {
            "type": "text",
            "text": f"""You are an expert interior design agent. Based on the property images and the valuation below, create 3 distinct design options.

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

Be creative but practical. Ensure designs are achievable within the budget constraints from the valuation.""",
        }
    )

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": image_content}],
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
    """
    PROCUREMENT AGENT
    Creates BOM, finds suppliers, compares prices
    """
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

        # Clean up the response - remove markdown code blocks if present
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

    # Progress indicator
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
    st.header("📤 Upload Property Images")

    # API Key input (for pilot - in production this would be server-side)
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

    # Image upload
    st.subheader("Property Photos")
    uploaded_files = st.file_uploader(
        "Upload photos of your property (multiple allowed)",
        type=["jpg", "jpeg", "png", "webp", "gif"],
        accept_multiple_files=True,
        help="Upload clear photos of each room, any problem areas, and exterior if relevant",
    )

    if uploaded_files:
        cols = st.columns(4)
        for i, file in enumerate(uploaded_files):
            with cols[i % 4]:
                # Reset file pointer and display
                file.seek(0)
                try:
                    img = Image.open(file)
                    st.image(img, caption=file.name, width="stretch")
                except Exception as e:
                    st.error(f"Could not load {file.name}: {e}")

    st.divider()

    # Preferences form
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

    # Check if API key is configured (either from secrets or user input)
    has_api_key = bool(
        st.secrets.get("ANTHROPIC_API_KEY") or st.session_state.get("api_key")
    )

    # Submit button
    if st.button(
        "🚀 Start Analysis",
        type="primary",
        disabled=not uploaded_files or not has_api_key,
    ):
        # Process and store images
        images = []
        failed_images = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, file in enumerate(uploaded_files):
            status_text.text(
                f"Processing image {i+1}/{len(uploaded_files)}: {file.name}"
            )
            progress_bar.progress((i + 1) / len(uploaded_files))

            result = process_uploaded_image(file)
            if result["success"]:
                images.append(result)
            else:
                failed_images.append(
                    f"{file.name}: {result.get('error', 'Unknown error')}"
                )

        progress_bar.empty()
        status_text.empty()

        if failed_images:
            st.warning(
                f"Some images could not be processed: {', '.join(failed_images)}"
            )

        if images:
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
            st.error("No images could be processed. Please try different image files.")

    if not has_api_key:
        st.warning("Please configure your Anthropic API key above to proceed.")
    if not uploaded_files:
        st.info("Please upload at least one property photo to proceed.")


def render_valuation_phase():
    st.header("📊 Property Valuation & Assessment")

    if not st.session_state.project_state["valuation"]:
        with st.spinner("🔍 Analyzing your property... This may take a minute."):
            valuation = run_valuation_agent(
                st.session_state.project_state["images"],
                st.session_state.project_state["preferences"],
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
        st.text(valuation["raw_response"])
        if st.button("🔄 Try Again"):
            st.session_state.project_state["valuation"] = None
            st.rerun()
        return

    # Display valuation results
    col1, col2, col3 = st.columns(3)

    cost = valuation.get("cost_estimate", {})
    with col1:
        st.metric("Estimated Cost (Low)", f"${cost.get('low_estimate', 0):,.0f}")
    with col2:
        st.metric("Estimated Cost (Mid)", f"${cost.get('mid_estimate', 0):,.0f}")
    with col3:
        st.metric("Estimated Cost (High)", f"${cost.get('high_estimate', 0):,.0f}")

    st.divider()

    # Property Assessment
    assessment = valuation.get("property_assessment", {})
    with st.expander("🏠 Property Assessment", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Condition:** {assessment.get('current_condition', 'N/A')}")
            st.write(
                f"**Rooms Identified:** {', '.join(assessment.get('room_types_identified', []))}"
            )
        with col2:
            st.write(
                f"**Est. Size:** {assessment.get('square_footage_estimate', 'N/A')}"
            )
            st.write(f"**Est. Age:** {assessment.get('age_estimate', 'N/A')}")
        st.write(f"**Details:** {assessment.get('condition_details', 'N/A')}")

    # Renovation Scope
    scope = valuation.get("renovation_scope", {})
    with st.expander("🔨 Recommended Renovation Scope", expanded=True):
        for work in scope.get("recommended_work", []):
            priority_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
                work.get("priority", ""), "⚪"
            )
            st.write(
                f"{priority_color} **{work.get('area', 'N/A')}:** {work.get('work_needed', 'N/A')}"
            )

        if scope.get("quick_wins"):
            st.write("**Quick Wins:**", ", ".join(scope.get("quick_wins", [])))

    # Cost Breakdown
    with st.expander("💰 Cost Breakdown"):
        for item in cost.get("breakdown", []):
            st.write(
                f"• **{item.get('category', 'N/A')}:** ${item.get('low', 0):,.0f} - ${item.get('high', 0):,.0f}"
            )

    # ROI Analysis
    roi = valuation.get("roi_analysis", {})
    with st.expander("📈 ROI Analysis"):
        st.write(
            f"**Estimated Value Increase:** ${roi.get('estimated_value_increase', 0):,.0f}"
        )
        st.write(f"**ROI:** {roi.get('roi_percentage', 0)}%")
        st.write(f"**Assessment:** {roi.get('payback_assessment', 'N/A')}")

    # Risk Assessment
    risk = valuation.get("risk_assessment", {})
    with st.expander("⚠️ Risk Assessment"):
        st.write(f"**Overall Risk Level:** {risk.get('overall_risk', 'N/A')}")
        for r in risk.get("risks", []):
            st.write(
                f"• **{r.get('risk', 'N/A')}** (Likelihood: {r.get('likelihood', 'N/A')})"
            )
            st.write(f"  Mitigation: {r.get('mitigation', 'N/A')}")

    # Timeline
    timeline = valuation.get("timeline_estimate", {})
    with st.expander("📅 Timeline Estimate"):
        st.write(
            f"**Duration:** {timeline.get('minimum_weeks', 0)} - {timeline.get('maximum_weeks', 0)} weeks"
        )
        st.write(f"**Phases:** {', '.join(timeline.get('phases', []))}")

    st.divider()

    # Approval Gate 1
    st.subheader("✅ Approval Gate 1: Valuation Review")
    st.write(
        "Review the valuation above. If you're satisfied, approve to proceed to design phase."
    )

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
            }
            st.rerun()


def render_design_phase():
    st.header("🎨 Design Options")

    if not st.session_state.project_state["designs"]:
        with st.spinner("🎨 Generating design options... This may take a minute."):
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
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Try Again"):
                st.session_state.project_state["designs"] = None
                st.rerun()
        with col2:
            if st.button("← Back to Valuation"):
                st.session_state.project_state["phase"] = "valuation"
                st.rerun()
        return

    if "raw_response" in designs:
        st.warning("Couldn't parse structured response. Raw output:")
        st.text(designs["raw_response"])
        if st.button("🔄 Try Again"):
            st.session_state.project_state["designs"] = None
            st.rerun()
        return

    # Display design options
    design_options = designs.get("design_options", [])

    if not design_options:
        st.warning("No design options were generated. Please try again.")
        if st.button("🔄 Regenerate Designs"):
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
                st.subheader(opt.get("name", "Design Option"))
                st.write(f"**Style:** {opt.get('style', 'N/A')}")
                st.write(f"**Concept:** {opt.get('concept', 'N/A')}")

                # Color palette
                palette = opt.get("color_palette", {})
                st.write("**Color Palette:**")
                pcols = st.columns(3)
                for j, (color_name, color_val) in enumerate(
                    [
                        ("Primary", palette.get("primary", "#000000")),
                        ("Secondary", palette.get("secondary", "#666666")),
                        ("Accent", palette.get("accent", "#999999")),
                    ]
                ):
                    with pcols[j]:
                        st.color_picker(
                            color_name, color_val, disabled=True, key=f"color_{i}_{j}"
                        )

                st.write(f"**Key Features:**")
                for feat in opt.get("key_features", []):
                    st.write(f"• {feat}")

            with col2:
                st.metric("Estimated Cost", f"${opt.get('estimated_cost', 0):,.0f}")

                st.write("**Pros:**")
                for pro in opt.get("pros", []):
                    st.write(f"✅ {pro}")

                st.write("**Cons:**")
                for con in opt.get("cons", []):
                    st.write(f"⚠️ {con}")

                st.write(f"**Best For:** {opt.get('best_for', 'N/A')}")

            # Room by room details
            with st.expander("Room-by-Room Details"):
                for room in opt.get("room_by_room", []):
                    st.write(f"**{room.get('room', 'Room')}**")
                    st.write(f"Changes: {', '.join(room.get('changes', []))}")
                    st.write(f"Materials: {', '.join(room.get('materials', []))}")
                    st.divider()

            if st.button(
                f"✓ Select Option {opt.get('option_number', i+1)}",
                key=f"select_{i}",
                type="primary",
            ):
                st.session_state.project_state["selected_design"] = opt.get(
                    "option_number", i + 1
                )
                st.success(f"Option {opt.get('option_number', i+1)} selected!")

    # Recommendations
    recs = designs.get("design_recommendations", {})
    with st.expander("💡 AI Recommendation"):
        st.write(f"**Recommended Option:** Option {recs.get('recommended_option', 1)}")
        st.write(f"**Reasoning:** {recs.get('reasoning', 'N/A')}")

    st.divider()

    # Show selected design and proceed button
    selected = st.session_state.project_state.get("selected_design")
    if selected:
        st.success(f"✓ You have selected Option {selected}")
        if st.button("📦 Proceed to Procurement", type="primary"):
            st.session_state.project_state["phase"] = "procurement"
            st.rerun()
    else:
        st.info("Please select one of the design options above to proceed.")

    # Option to go back
    if st.button("← Back to Valuation"):
        st.session_state.project_state["phase"] = "valuation"
        st.session_state.project_state["designs"] = None
        st.rerun()


def render_procurement_phase():
    st.header("📦 Bill of Materials & Procurement")

    if not st.session_state.project_state.get("bom"):
        with st.spinner("📦 Generating Bill of Materials... This may take a minute."):
            bom = run_procurement_agent(
                st.session_state.project_state["designs"],
                st.session_state.project_state.get("selected_design", 1),
            )
            st.session_state.project_state["bom"] = bom
            st.rerun()

    bom = st.session_state.project_state["bom"]

    if "error" in bom:
        st.error(f"Error: {bom['error']}")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Try Again"):
                st.session_state.project_state["bom"] = None
                st.rerun()
        with col2:
            if st.button("← Back to Design"):
                st.session_state.project_state["phase"] = "design"
                st.rerun()
        return

    if "raw_response" in bom:
        st.warning("Couldn't parse structured response. Raw output:")
        st.text(bom["raw_response"])
        if st.button("🔄 Try Again"):
            st.session_state.project_state["bom"] = None
            st.rerun()
        return

    # Summary
    summary = bom.get("total_summary", {})
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Materials (Low)", f"${summary.get('materials_low', 0):,.0f}")
    with col2:
        st.metric("Materials (High)", f"${summary.get('materials_high', 0):,.0f}")
    with col3:
        st.metric("Labor (Low)", f"${summary.get('labor_low', 0):,.0f}")
    with col4:
        st.metric("Labor (High)", f"${summary.get('labor_high', 0):,.0f}")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Grand Total (Low)", f"${summary.get('grand_total_low', 0):,.0f}")
    with col2:
        st.metric("Grand Total (High)", f"${summary.get('grand_total_high', 0):,.0f}")

    st.divider()

    # Categories
    bill = bom.get("bill_of_materials", {})
    for category in bill.get("categories", []):
        with st.expander(
            f"📁 {category.get('category_name', 'Category')} - ${category.get('category_total', 0):,.0f}"
        ):
            for item in category.get("items", []):
                st.write(f"**{item.get('item_name', 'Item')}**")
                st.write(
                    f"Quantity: {item.get('quantity', 0)} {item.get('unit', 'units')}"
                )
                st.write(
                    f"Price Range: ${item.get('unit_price_low', 0):,.2f} - ${item.get('unit_price_high', 0):,.2f} per {item.get('unit', 'unit')}"
                )
                st.write(
                    f"Total: ${item.get('total_price_low', 0):,.0f} - ${item.get('total_price_high', 0):,.0f}"
                )

                # Handle both old and new format for suppliers
                if item.get("suggested_suppliers"):
                    st.write("**Suggested Suppliers:**")
                    for supplier in item.get("suggested_suppliers", []):
                        if isinstance(supplier, dict):
                            st.write(
                                f"• {supplier.get('name', 'N/A')} ({supplier.get('type', 'N/A')}) - {supplier.get('price_range', 'N/A')}"
                            )
                        else:
                            st.write(f"• {supplier}")
                elif item.get("supplier"):
                    st.write(f"**Suggested Supplier:** {item.get('supplier')}")

                st.divider()

    # Labor estimates
    with st.expander("👷 Labor Estimates"):
        labor_list = bom.get("labor_estimates", [])
        if labor_list:
            for labor in labor_list:
                st.write(
                    f"**{labor.get('trade', 'Trade')}:** {labor.get('estimated_hours', 0)} hours"
                )
                st.write(
                    f"Rate: ${labor.get('hourly_rate_low', 0)}-${labor.get('hourly_rate_high', 0)}/hr"
                )
                st.write(
                    f"Total: ${labor.get('total_low', 0):,.0f} - ${labor.get('total_high', 0):,.0f}"
                )
                if labor.get("notes"):
                    st.write(f"Notes: {labor.get('notes')}")
                st.divider()
        else:
            st.write("No labor estimates provided.")

    # Procurement strategy
    strategy = bom.get("procurement_strategy", {})
    with st.expander("📋 Procurement Strategy"):
        st.write(
            f"**Recommended Approach:** {strategy.get('recommended_approach', 'N/A')}"
        )
        if strategy.get("seasonal_considerations"):
            st.write(f"**Best Time to Buy:** {strategy.get('seasonal_considerations')}")
        if strategy.get("order_sequence"):
            st.write("**Order Sequence:**")
            for seq in strategy.get("order_sequence", []):
                st.write(f"• {seq}")

    st.divider()

    # Approval Gate 2
    st.subheader("✅ Approval Gate 2: Budget Approval")
    st.write("Review the Bill of Materials above. This completes Phase 1 of the pilot.")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("👍 Approve Budget", type="primary"):
            st.session_state.project_state["gate_2_approved"] = True
            st.session_state.project_state["phase"] = "complete"
            st.rerun()
    with col2:
        if st.button("← Back to Design"):
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
            }
            st.rerun()


def render_complete_phase():
    st.header("✅ Phase 1 Complete!")

    st.success("Congratulations! You've completed Phase 1 of OmniRenovation AI.")

    st.balloons()

    st.subheader("📊 Project Summary")

    valuation = st.session_state.project_state.get("valuation", {})
    designs = st.session_state.project_state.get("designs", {})
    bom = st.session_state.project_state.get("bom", {})

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Valuation Summary:**")
        cost = valuation.get("cost_estimate", {})
        st.write(
            f"• Estimated Renovation Cost: ${cost.get('low_estimate', 0):,.0f} - ${cost.get('high_estimate', 0):,.0f}"
        )

        roi = valuation.get("roi_analysis", {})
        st.write(f"• Expected ROI: {roi.get('roi_percentage', 0)}%")

        timeline = valuation.get("timeline_estimate", {})
        st.write(
            f"• Timeline: {timeline.get('minimum_weeks', 0)} - {timeline.get('maximum_weeks', 0)} weeks"
        )

    with col2:
        st.write("**Selected Design:**")
        selected = st.session_state.project_state.get("selected_design", 1)
        for opt in designs.get("design_options", []):
            if opt.get("option_number") == selected:
                st.write(f"• {opt.get('name', 'N/A')} ({opt.get('style', 'N/A')})")
                st.write(f"• Estimated Cost: ${opt.get('estimated_cost', 0):,.0f}")

        summary = bom.get("total_summary", {})
        st.write(
            f"• Final Budget Range: ${summary.get('grand_total_low', 0):,.0f} - ${summary.get('grand_total_high', 0):,.0f}"
        )

    st.divider()

    st.subheader("🚀 What's Next? (Phase 2)")
    st.write(
        """
    Phase 2 will include:
    - **Contractor Outreach Agent**: Finding and vetting contractors
    - **Scheduling Agent**: Coordinating timelines
    - **Monitoring Agent**: Tracking progress during renovation
    - **Audit Agent**: Final documentation and handoff
    """
    )

    st.divider()

    # Export data
    st.subheader("📥 Export Project Data")

    # Remove base64 image data from export (too large)
    export_images = [
        {"name": img["name"], "type": img["type"]}
        for img in st.session_state.project_state.get("images", [])
    ]

    export_data = {
        "timestamp": datetime.now().isoformat(),
        "preferences": st.session_state.project_state.get("preferences", {}),
        "images_uploaded": export_images,
        "valuation": valuation,
        "selected_design": st.session_state.project_state.get("selected_design"),
        "designs": designs,
        "bill_of_materials": bom,
    }

    st.download_button(
        "Download Project JSON",
        data=json.dumps(export_data, indent=2),
        file_name=f"omnirenovation_project_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
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
        }
        st.rerun()


# ============== MAIN APP ==============


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
