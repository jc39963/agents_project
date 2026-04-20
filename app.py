"""
1. Capture frame from webcam.

2. Perceive color/category using perception.py.

3. Pass that data to style agent 1 and style agent 2.

4. Display their competing Zara picks on the screen."""

import streamlit as st
from src.dl.agent import Agent
from PIL import Image 
import numpy as np 
import cv2 
import os
import pandas as pd
import json
#from eval import run_benchmark

# streamlit app for UI
# functionality: logs, ask agent, check evaluation

st.set_page_config(page_title="CompleteTheLook", page_icon="👗", layout="centered")

st.title("👗 CompleteTheLook")
st.caption("Upload an image of clothes and I'll suggest how to complete the look")

#  Session state 
if "agent" not in st.session_state:
    st.session_state.agent = Agent()

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of roles and content
if "logs" not in st.session_state:
    st.session_state.logs = [] # to store tool call history / logs for eval and checking

# ---- making tabs so can do eval 
tab_chat, tab_eval = st.tabs(["💬 User Chat", "📊 Evaluation Dashboard"])
# ---- creating chat tab

with tab_chat:
    #  Chat history 
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 1. Webcam Input Component
    img_file = st.camera_input("Show me your outfit!")



    if img_file:
        # Show user message immediately
        st.session_state.messages.append({"role": "user", "content": img_file})
        with st.chat_message("user"):
            st.markdown(img_file)

        # Stream agent response
        with st.chat_message("assistant"):
            output_area = st.empty()
            full_response = ""

            for chunk in st.session_state.agent.chat(img_file):
                full_response += chunk
                output_area.markdown(full_response + "▌")   # blinking cursor effect

            output_area.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

"""# evaluation tab
with tab_eval:
    st.header(" Agent Evaluation Benchmarks")
    st.markdown("Run the standard test suite to measure agent performance quantitatively.")

    if st.button(" Run Full Evaluation Benchmark"):
        # call it from a eval.py
        results = run_benchmark()
        df = pd.DataFrame(results)
        
        st.success("Evaluation Complete!")
        
        # Display high-level metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Success Rate", f"{df['success'].mean()*100:.1f}%")
        col2.metric("Avg Grounding", f"{df['grounding_score'].mean():.2f}/5")
        col3.metric("Avg Steps", f"{df['steps'].mean():.1f}")
        
        # show  detailed table
        st.dataframe(df)
#  Sidebar for logging and available functions
with st.sidebar:
    st.header("PantryChef")
    st.markdown(
        "An LLM agent that searches real recipes and adapts them to exactly "
        "what you have in your kitchen.\n\n"
        "**Tools available:**\n"
        "- `get_recipes` — find candidates\n"
        "- `get_missing_ingredients` — pantry diff\n"
        "- `get_nutrition_summary` — macros\n"
        "- `get_recipe_details` — full instructions\n"
    )
    st.divider()
    if st.button("🗑 Clear conversation"):
        st.session_state.agent.reset()
        st.session_state.messages = []
        st.rerun()

    st.subheader("🕵️ Internal Agent Logs")
    st.caption("Real-time 'Chain of Thought' and Tool Outputs")
    
    if not st.session_state.logs:
        st.info("No tool calls yet. Ask the agent a question!")
    
    for log in st.session_state.logs:
        with st.expander(f"Action: {log['action']}", expanded=True):
            if "Thought" in log['action']:
                st.write(log['result'])
            else:
                st.json(log['result']) 

"""

