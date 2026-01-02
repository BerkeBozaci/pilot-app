"""
OmniRenovation AI - Phase 1 Pilot
Fast prototype for testing the core agent workflow
"""

import streamlit as st
import anthropic
import json
import base64
from datetime import datetime
from pathlib import Path

# Page config
st.set_page_config(
    page_title="OmniRenovation AI",
    page_icon="🏠",
    layout="wide"
)

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
        "gate_2_approved": False
    }

def encode_image(uploaded_file):
    """Convert uploaded file to base64"""
    bytes_data = uploaded_file.getvalue()
    return base64.standard_b64encode(bytes_data).decode("utf-8")

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
        image_content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": img["type"],
                "data": img["data"]
            }
        })
    
    image_content.append({
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

Be realistic and detailed in your estimates. Base costs on current US market rates but note that costs vary by location."""
    })
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": image_content}]
    )
    
    # Parse JSON from response
    response_text = response.content[0].text
    try:
        # Try to extract JSON from the response
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            return json.loads(response_text[json_start:json_end])
    except json.JSONDecodeError:
        pass
    
    return {"raw_response": response_text}

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
        image_content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": img["type"],
                "data": img["data"]
            }
        })
    
    image_content.append({
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

Be creative but practical. Ensure designs are achievable within the budget constraints from the valuation."""
    })
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": image_content}]
    )
    
    response_text = response.content[0].text
    try:
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            return json.loads(response_text[json_start:json_end])
    except json.JSONDecodeError:
        pass
    
    return {"raw_response": response_text}

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
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": f"""You are a procurement specialist agent. Create a detailed Bill of Materials (BOM) for the following design:

SELECTED DESIGN:
{json.dumps(selected_design, indent=2)}

Create a comprehensive BOM in this JSON format:
{{
    "bill_of_materials": {{
        "project_summary": {{
            "design_name": "name",
            "total_estimated_cost": 0,
            "number_of_items": 0,
            "number_of_categories": 0
        }},
        "categories": [
            {{
                "category_name": "e.g., Flooring",
                "category_total": 0,
                "items": [
                    {{
                        "item_name": "specific product name",
                        "description": "details",
                        "specifications": {{
                            "dimensions": "if applicable",
                            "material": "material type",
                            "color": "color/finish"
                        }},
                        "quantity": 0,
                        "unit": "sq ft / pieces / etc",
                        "unit_price_low": 0,
                        "unit_price_high": 0,
                        "total_price_low": 0,
                        "total_price_high": 0,
                        "suggested_suppliers": [
                            {{
                                "name": "Store/Supplier name",
                                "type": "big-box / specialty / online",
                                "price_range": "$ / $$ / $$$",
                                "notes": "any relevant notes"
                            }}
                        ],
                        "alternatives": [
                            {{
                                "name": "alternative product",
                                "price_difference": "+/- amount",
                                "trade_off": "what you gain/lose"
                            }}
                        ],
                        "lead_time_days": 0,
                        "installation_notes": "any special requirements"
                    }}
                ]
            }}
        ]
    }},
    "labor_estimates": [
        {{
            "trade": "e.g., Electrician",
            "estimated_hours": 0,
            "hourly_rate_low": 0,
            "hourly_rate_high": 0,
            "total_low": 0,
            "total_high": 0,
            "notes": "scope of work"
        }}
    ],
    "procurement_strategy": {{
        "recommended_approach": "description",
        "bulk_discount_opportunities": ["opportunity 1"],
        "seasonal_considerations": "best time to buy",
        "order_sequence": ["what to order first", "what can wait"]
    }},
    "total_summary": {{
        "materials_low": 0,
        "materials_high": 0,
        "labor_low": 0,
        "labor_high": 0,
        "contingency": 0,
        "grand_total_low": 0,
        "grand_total_high": 0
    }}
}}

Use realistic US market prices. Include both budget and premium options where relevant."""
        }]
    )
    
    response_text = response.content[0].text
    try:
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            return json.loads(response_text[json_start:json_end])
    except json.JSONDecodeError:
        pass
    
    return {"raw_response": response_text}


# ============== UI COMPONENTS ==============

def render_header():
    st.title("🏠 OmniRenovation AI")
    st.caption("AI-Native Renovation & Asset Management Platform - Phase 1 Pilot")
    
    # Progress indicator
    phases = ["📤 Upload", "📊 Valuation", "🎨 Design", "📦 Procurement", "✅ Complete"]
    phase_map = {"upload": 0, "valuation": 1, "design": 2, "procurement": 3, "complete": 4}
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
            help="Enter your Claude API key. Get one at console.anthropic.com"
        )
        if api_key:
            st.session_state.api_key = api_key
            st.success("API key configured!")
    
    st.divider()
    
    # Image upload
    st.subheader("Property Photos")
    uploaded_files = st.file_uploader(
        "Upload photos of your property (multiple allowed)",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        help="Upload clear photos of each room, any problem areas, and exterior if relevant"
    )
    
    if uploaded_files:
        cols = st.columns(4)
        for i, file in enumerate(uploaded_files):
            with cols[i % 4]:
                st.image(file, caption=file.name, use_container_width=True)
    
    st.divider()
    
    # Preferences form
    st.subheader("Your Preferences")
    
    col1, col2 = st.columns(2)
    
    with col1:
        budget = st.select_slider(
            "Budget Range",
            options=["< $5,000", "$5,000 - $15,000", "$15,000 - $30,000", "$30,000 - $50,000", "$50,000 - $100,000", "> $100,000"],
            value="$15,000 - $30,000"
        )
        
        style = st.selectbox(
            "Design Style Preference",
            ["Modern Minimalist", "Contemporary", "Scandinavian", "Industrial", "Mid-Century Modern", 
             "Traditional", "Transitional", "Bohemian", "Coastal", "Farmhouse", "No Preference"]
        )
    
    with col2:
        goals = st.multiselect(
            "Renovation Goals",
            ["Increase property value", "Improve functionality", "Update aesthetics", 
             "Fix structural issues", "Increase energy efficiency", "Prepare for sale",
             "Personal enjoyment", "Accommodate family changes"],
            default=["Update aesthetics", "Improve functionality"]
        )
        
        priorities = st.multiselect(
            "Priority Areas",
            ["Kitchen", "Bathroom(s)", "Living Room", "Bedroom(s)", "Flooring Throughout",
             "Lighting", "Paint/Walls", "Windows", "Exterior", "Storage"],
            default=["Kitchen", "Bathroom(s)"]
        )
    
    additional_notes = st.text_area(
        "Additional Notes",
        placeholder="Any specific requirements, constraints, or wishes..."
    )
    
    st.divider()
    
    # Submit button
    if st.button("🚀 Start Analysis", type="primary", disabled=not uploaded_files or not st.session_state.api_key):
        # Store images and preferences
        images = []
        for file in uploaded_files:
            images.append({
                "name": file.name,
                "type": file.type,
                "data": encode_image(file)
            })
        
        st.session_state.project_state["images"] = images
        st.session_state.project_state["preferences"] = {
            "budget": budget,
            "style": style,
            "goals": ", ".join(goals),
            "priorities": ", ".join(priorities),
            "notes": additional_notes
        }
        st.session_state.project_state["phase"] = "valuation"
        st.rerun()

def render_valuation_phase():
    st.header("📊 Property Valuation & Assessment")
    
    if not st.session_state.project_state["valuation"]:
        with st.spinner("🔍 Analyzing your property... This may take a minute."):
            valuation = run_valuation_agent(
                st.session_state.project_state["images"],
                st.session_state.project_state["preferences"]
            )
            st.session_state.project_state["valuation"] = valuation
            st.rerun()
    
    valuation = st.session_state.project_state["valuation"]
    
    if "error" in valuation:
        st.error(f"Error: {valuation['error']}")
        return
    
    if "raw_response" in valuation:
        st.warning("Couldn't parse structured response. Raw output:")
        st.text(valuation["raw_response"])
        return
    
    # Display valuation results
    col1, col2, col3 = st.columns(3)
    
    cost = valuation.get("cost_estimate", {})
    with col1:
        st.metric(
            "Estimated Cost (Low)",
            f"${cost.get('low_estimate', 0):,.0f}"
        )
    with col2:
        st.metric(
            "Estimated Cost (Mid)",
            f"${cost.get('mid_estimate', 0):,.0f}"
        )
    with col3:
        st.metric(
            "Estimated Cost (High)",
            f"${cost.get('high_estimate', 0):,.0f}"
        )
    
    st.divider()
    
    # Property Assessment
    assessment = valuation.get("property_assessment", {})
    with st.expander("🏠 Property Assessment", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Condition:** {assessment.get('current_condition', 'N/A')}")
            st.write(f"**Rooms Identified:** {', '.join(assessment.get('room_types_identified', []))}")
        with col2:
            st.write(f"**Est. Size:** {assessment.get('square_footage_estimate', 'N/A')}")
            st.write(f"**Est. Age:** {assessment.get('age_estimate', 'N/A')}")
        st.write(f"**Details:** {assessment.get('condition_details', 'N/A')}")
    
    # Renovation Scope
    scope = valuation.get("renovation_scope", {})
    with st.expander("🔨 Recommended Renovation Scope", expanded=True):
        for work in scope.get("recommended_work", []):
            priority_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(work.get("priority", ""), "⚪")
            st.write(f"{priority_color} **{work.get('area', 'N/A')}:** {work.get('work_needed', 'N/A')}")
        
        if scope.get("quick_wins"):
            st.write("**Quick Wins:**", ", ".join(scope.get("quick_wins", [])))
    
    # Cost Breakdown
    with st.expander("💰 Cost Breakdown"):
        for item in cost.get("breakdown", []):
            st.write(f"• **{item.get('category', 'N/A')}:** ${item.get('low', 0):,.0f} - ${item.get('high', 0):,.0f}")
    
    # ROI Analysis
    roi = valuation.get("roi_analysis", {})
    with st.expander("📈 ROI Analysis"):
        st.write(f"**Estimated Value Increase:** ${roi.get('estimated_value_increase', 0):,.0f}")
        st.write(f"**ROI:** {roi.get('roi_percentage', 0)}%")
        st.write(f"**Assessment:** {roi.get('payback_assessment', 'N/A')}")
    
    # Risk Assessment
    risk = valuation.get("risk_assessment", {})
    with st.expander("⚠️ Risk Assessment"):
        st.write(f"**Overall Risk Level:** {risk.get('overall_risk', 'N/A')}")
        for r in risk.get("risks", []):
            st.write(f"• **{r.get('risk', 'N/A')}** (Likelihood: {r.get('likelihood', 'N/A')})")
            st.write(f"  Mitigation: {r.get('mitigation', 'N/A')}")
    
    # Timeline
    timeline = valuation.get("timeline_estimate", {})
    with st.expander("📅 Timeline Estimate"):
        st.write(f"**Duration:** {timeline.get('minimum_weeks', 0)} - {timeline.get('maximum_weeks', 0)} weeks")
        st.write(f"**Phases:** {', '.join(timeline.get('phases', []))}")
    
    st.divider()
    
    # Approval Gate 1
    st.subheader("✅ Approval Gate 1: Valuation Review")
    st.write("Review the valuation above. If you're satisfied, approve to proceed to design phase.")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👍 Approve & Continue to Design", type="primary"):
            st.session_state.project_state["gate_1_approved"] = True
            st.session_state.project_state["phase"] = "design"
            st.rerun()
    with col2:
        if st.button("🔄 Start Over"):
            st.session_state.project_state = {
                "phase": "upload", "images": [], "preferences": {},
                "valuation": None, "designs": None, "bom": None,
                "gate_1_approved": False, "gate_2_approved": False
            }
            st.rerun()

def render_design_phase():
    st.header("🎨 Design Options")
    
    if not st.session_state.project_state["designs"]:
        with st.spinner("🎨 Generating design options... This may take a minute."):
            designs = run_design_agent(
                st.session_state.project_state["images"],
                st.session_state.project_state["preferences"],
                st.session_state.project_state["valuation"]
            )
            st.session_state.project_state["designs"] = designs
            st.rerun()
    
    designs = st.session_state.project_state["designs"]
    
    if "error" in designs:
        st.error(f"Error: {designs['error']}")
        return
    
    if "raw_response" in designs:
        st.warning("Couldn't parse structured response. Raw output:")
        st.text(designs["raw_response"])
        return
    
    # Display design options
    design_options = designs.get("design_options", [])
    
    tabs = st.tabs([f"Option {opt.get('option_number', i+1)}: {opt.get('name', 'Design')}" for i, opt in enumerate(design_options)])
    
    selected_option = None
    
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
                for j, (color_name, color_val) in enumerate([
                    ("Primary", palette.get("primary", "#000")),
                    ("Secondary", palette.get("secondary", "#666")),
                    ("Accent", palette.get("accent", "#999"))
                ]):
                    with pcols[j]:
                        st.color_picker(color_name, color_val, disabled=True, key=f"color_{i}_{j}")
                
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
            
            if st.button(f"Select Option {opt.get('option_number', i+1)}", key=f"select_{i}", type="primary"):
                selected_option = opt.get("option_number", i+1)
    
    # Recommendations
    recs = designs.get("design_recommendations", {})
    with st.expander("💡 AI Recommendation"):
        st.write(f"**Recommended Option:** Option {recs.get('recommended_option', 1)}")
        st.write(f"**Reasoning:** {recs.get('reasoning', 'N/A')}")
    
    st.divider()
    
    if selected_option:
        st.success(f"You selected Option {selected_option}!")
        st.session_state.project_state["selected_design"] = selected_option
        
        if st.button("📦 Proceed to Procurement", type="primary"):
            st.session_state.project_state["phase"] = "procurement"
            st.rerun()

def render_procurement_phase():
    st.header("📦 Bill of Materials & Procurement")
    
    if not st.session_state.project_state.get("bom"):
        with st.spinner("📦 Generating Bill of Materials... This may take a minute."):
            bom = run_procurement_agent(
                st.session_state.project_state["designs"],
                st.session_state.project_state.get("selected_design", 1)
            )
            st.session_state.project_state["bom"] = bom
            st.rerun()
    
    bom = st.session_state.project_state["bom"]
    
    if "error" in bom:
        st.error(f"Error: {bom['error']}")
        return
    
    if "raw_response" in bom:
        st.warning("Couldn't parse structured response. Raw output:")
        st.text(bom["raw_response"])
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
        with st.expander(f"📁 {category.get('category_name', 'Category')} - ${category.get('category_total', 0):,.0f}"):
            for item in category.get("items", []):
                st.write(f"**{item.get('item_name', 'Item')}**")
                st.write(f"Quantity: {item.get('quantity', 0)} {item.get('unit', 'units')}")
                st.write(f"Price Range: ${item.get('unit_price_low', 0):,.2f} - ${item.get('unit_price_high', 0):,.2f} per {item.get('unit', 'unit')}")
                st.write(f"Total: ${item.get('total_price_low', 0):,.0f} - ${item.get('total_price_high', 0):,.0f}")
                
                if item.get("suggested_suppliers"):
                    st.write("**Suggested Suppliers:**")
                    for supplier in item.get("suggested_suppliers", []):
                        st.write(f"• {supplier.get('name', 'N/A')} ({supplier.get('type', 'N/A')}) - {supplier.get('price_range', 'N/A')}")
                
                st.divider()
    
    # Labor estimates
    with st.expander("👷 Labor Estimates"):
        for labor in bom.get("labor_estimates", []):
            st.write(f"**{labor.get('trade', 'Trade')}:** {labor.get('estimated_hours', 0)} hours")
            st.write(f"Rate: ${labor.get('hourly_rate_low', 0)}-${labor.get('hourly_rate_high', 0)}/hr")
            st.write(f"Total: ${labor.get('total_low', 0):,.0f} - ${labor.get('total_high', 0):,.0f}")
            st.write(f"Notes: {labor.get('notes', 'N/A')}")
            st.divider()
    
    # Procurement strategy
    strategy = bom.get("procurement_strategy", {})
    with st.expander("📋 Procurement Strategy"):
        st.write(f"**Recommended Approach:** {strategy.get('recommended_approach', 'N/A')}")
        st.write(f"**Best Time to Buy:** {strategy.get('seasonal_considerations', 'N/A')}")
        st.write("**Order Sequence:**")
        for seq in strategy.get("order_sequence", []):
            st.write(f"• {seq}")
    
    st.divider()
    
    # Approval Gate 2
    st.subheader("✅ Approval Gate 2: Budget Approval")
    st.write("Review the Bill of Materials above. This completes Phase 1 of the pilot.")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👍 Approve Budget", type="primary"):
            st.session_state.project_state["gate_2_approved"] = True
            st.session_state.project_state["phase"] = "complete"
            st.rerun()
    with col2:
        if st.button("🔄 Start Over"):
            st.session_state.project_state = {
                "phase": "upload", "images": [], "preferences": {},
                "valuation": None, "designs": None, "bom": None,
                "gate_1_approved": False, "gate_2_approved": False
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
        st.write(f"• Estimated Renovation Cost: ${cost.get('low_estimate', 0):,.0f} - ${cost.get('high_estimate', 0):,.0f}")
        
        roi = valuation.get("roi_analysis", {})
        st.write(f"• Expected ROI: {roi.get('roi_percentage', 0)}%")
        
        timeline = valuation.get("timeline_estimate", {})
        st.write(f"• Timeline: {timeline.get('minimum_weeks', 0)} - {timeline.get('maximum_weeks', 0)} weeks")
    
    with col2:
        st.write("**Selected Design:**")
        selected = st.session_state.project_state.get("selected_design", 1)
        for opt in designs.get("design_options", []):
            if opt.get("option_number") == selected:
                st.write(f"• {opt.get('name', 'N/A')} ({opt.get('style', 'N/A')})")
                st.write(f"• Estimated Cost: ${opt.get('estimated_cost', 0):,.0f}")
        
        summary = bom.get("total_summary", {})
        st.write(f"• Final Budget Range: ${summary.get('grand_total_low', 0):,.0f} - ${summary.get('grand_total_high', 0):,.0f}")
    
    st.divider()
    
    st.subheader("🚀 What's Next? (Phase 2)")
    st.write("""
    Phase 2 will include:
    - **Contractor Outreach Agent**: Finding and vetting contractors
    - **Scheduling Agent**: Coordinating timelines
    - **Monitoring Agent**: Tracking progress during renovation
    - **Audit Agent**: Final documentation and handoff
    """)
    
    st.divider()
    
    # Export data
    st.subheader("📥 Export Project Data")
    
    export_data = {
        "timestamp": datetime.now().isoformat(),
        "preferences": st.session_state.project_state.get("preferences", {}),
        "valuation": valuation,
        "selected_design": st.session_state.project_state.get("selected_design"),
        "designs": designs,
        "bill_of_materials": bom
    }
    
    st.download_button(
        "Download Project JSON",
        data=json.dumps(export_data, indent=2),
        file_name=f"omnirenovation_project_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )
    
    if st.button("🔄 Start New Project"):
        st.session_state.project_state = {
            "phase": "upload", "images": [], "preferences": {},
            "valuation": None, "designs": None, "bom": None,
            "gate_1_approved": False, "gate_2_approved": False
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
