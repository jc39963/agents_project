"""
1. Capture frame from webcam.

2. Perceive color/category using perception.py.

3. Pass that data to style agent 1 and style agent 2.

4. Display their competing Zara picks on the screen."""
import streamlit as st
from src.dl.agent import Agent
from PIL import Image 
import numpy as np 
import base64
import io
import time
import pandas as pd
import json

# --- UI CONFIG ---
st.set_page_config(page_title="👜 Fitting Room: CompleteTheLook", page_icon="", layout="wide")


st.markdown("""
    <style>
    /* Center and Style the Tabs */
    .stTabs [data-baseweb="tab-list"] {
        display: flex;
        justify-content: center;
        gap: 50px;
        margin-bottom: 30px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 28px !important; 
        font-weight: 800 !important;
        color: #31333F;
    }
    .stTabs [aria-selected="true"] {
        color: #FF4B4B !important;
    }
    
    /* Ensure the camera widget stays inside its column */
    div[data-testid="stCameraInput"] {
        width: 100% !important;
        max-width: 450px;
        margin: 0 auto;
    }
    
    /* Global container constraint */
    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# convert to format base64 that gpt can take in AND save to jpg for CLIP embedding 
def pil_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# --- SESSION STATE ---
if "agent" not in st.session_state:
    st.session_state.agent = Agent()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "logs" not in st.session_state:
    st.session_state.logs = []

# --- APP LAYOUT ---
st.title(" Fitting Room: Complete the Look")

# tabs for Chat vs Evaluation
tab_chat, tab_eval = st.tabs(["👜 Style Assistant", "🛠️ Evaluation"])

with tab_chat:
    col_input, col_output = st.columns([1, 1.5])

    with col_input:
        st.subheader("Step 1: Capture")
        
        # maybe add prep timer or cropping??
        img_file = st.camera_input("Fashion Capture", label_visibility="hidden")

    with col_output:
        st.subheader("Step 2: Recommendation")
        
        if img_file:
            # 1. Perception
            raw_img = Image.open(img_file).convert("RGB")
            
            # save the image locally so CLIP / tools can access it
            image_path = "data/images/user_image.jpg"
            raw_img.save(image_path)
            
            
            
            # 2. Planning + Agent Execution
            with st.chat_message("assistant"):
                st.write("✨ Analyzing your piece...")
                
                # Convert image to base64 so gpt can read it
                img_b64 = pil_to_base64(raw_img)
                
                output_area = st.empty()
                full_response = ""
                
                # call agent, passing the b64 string to the chat and image path
                for chunk in st.session_state.agent.chat(img_b64, image_path):
                    full_response += chunk
                    output_area.markdown(full_response + "▌")
                
                output_area.markdown(full_response)
                

                # --- 3. Visual Results ---
        # we look for the most recent search results to display
        search_results = None
        for log in reversed(st.session_state.logs):
            if log['action'] == 'find_similar_items':
                try:
                    search_results = json.loads(log['result'])
                    break # Get the latest one
                except:
                    continue

        if search_results:
            st.divider()
            st.header("🛍️ Shop the Look")
            st.caption("Agent's top picks from the Zara Catalog")
            
            # Create a clean grid
            cols = st.columns(3)
            # We show the top 9 matches
            for idx, match in enumerate(search_results.get('matches', [])[:9]):
                with cols[idx]:
                    meta = match.get('metadata', {})
                    # Perception check: ensure the URL exists
                    if 'image_url' in meta:
                        st.image(meta['image_url'], width=True)
                        st.write(f"**{meta.get('name', 'Zara Item')}**")
                        st.write(f"Match: {int(match.get('score', 0)*100)}%")
                        st.link_button("Buy on Zara", meta.get('product_url', '#'))

# IMPLEMENT EVAL
# with tab_eval 

# Sidebar Logs to see thinking
with st.sidebar:
    st.header("🕵️ Agent Logic")
    if st.button("🗑 Reset Session"):
        st.session_state.agent.reset()
        st.session_state.messages = []
        st.session_state.logs = []
        st.session_state.clear()
        st.rerun()
    
    for log in st.session_state.logs:
        with st.expander(f"Action: {log['action']}"):
            st.write(log['result'])