import streamlit as st
import google.generativeai as genai
from PIL import Image
import time

# 1. Page Config
st.set_page_config(page_title="AI Room Designer", layout="wide")
st.title("🏠 AI Interior Designer")

# 2. Sidebar - Setup
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("Gemini API Key", type="password")
style = st.sidebar.selectbox(
    "Choose a Style", ["Modern", "Boho", "Industrial", "Minimalist"]
)

# 3. Model Setup & Quota Handling
if api_key:
    genai.configure(api_key=api_key)
    # Using the stable 2026 ID
    model = genai.GenerativeModel("gemini-2.5-flash-image")

    uploaded_file = st.file_uploader(
        "Upload an empty room photo", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        img = Image.open(uploaded_file)
        col1, col2 = st.columns(2)

        with col1:
            st.image(img, caption="Original Room", use_container_width=True)

        if st.button("Generate Design"):
            with st.spinner("AI is painting your room..."):
                try:
                    prompt = f"Maintain the exact walls, floor, and windows. Interior design this empty room in {style} style with real furniture."

                    # Generation Call
                    response = model.generate_content([prompt, img])

                    # 2026 SDK result extraction
                    generated_img = (
                        response.candidates[0].content.parts[0].inline_data.data
                    )

                    with col2:
                        st.image(
                            generated_img,
                            caption=f"New {style} Design",
                            use_container_width=True,
                        )
                        st.success("Done!")

                except Exception as e:
                    if "429" in str(e):
                        st.error(
                            "🚨 QUOTA ERROR: Your billing account is linked, but your 'Images Per Minute' limit is currently 0. This happens during the 24-hour verification period for new Tier 1 accounts."
                        )
                    else:
                        st.error(f"Error: {e}")
else:
    st.info("Enter your API Key in the sidebar to begin.")
