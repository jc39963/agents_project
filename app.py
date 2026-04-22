"""
1. Capture frame from webcam.

2. Perceive color/category using perception.py.

3. Pass that data to style agent 1 and style agent 2.

4. Display their competing Zara picks on the screen."""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import torch  # Import torch first to initialize its OpenMP backend safely!

torch.set_num_threads(1)  # Prevent PyTorch from spawning clashing thread pools

import streamlit as st
from src.dl.agent import Agent
from PIL import Image
import numpy as np
import base64
import io
import time
import pandas as pd
import json
import threading
from src.non_dl.agent import llm_agent_with_function_calling
from src.eval import evaluate_aesthetic, evaluate_robustness
from streamlit.runtime.scriptrunner import add_script_run_ctx

# --- UI CONFIG ---
st.set_page_config(
    page_title="👜 Fitting Room: CompleteTheLook",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)


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
if "latency" not in st.session_state:
    st.session_state.latency = None
if "last_agent" not in st.session_state:
    st.session_state.last_agent = None
if "non_dl_recs" not in st.session_state:
    st.session_state.non_dl_recs = None
if "trigger_agent" not in st.session_state:
    st.session_state.trigger_agent = None
if "eval_state" not in st.session_state:
    st.session_state.eval_state = {
        "running": False,
        "done": False,
        "results": None,
        "robust_results": None,
    }

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

            st.success("Image captured! Choose an agent to generate recommendations:")
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                run_dl = st.button(
                    "🤖 DL Agent", use_container_width=True, type="primary"
                )
                st.caption("Uses Deep Learning (CLIP, GPT) for visual matches")
            with btn_col2:
                run_non_dl = st.button(
                    "📐 Non-DL Agent", use_container_width=True, type="primary"
                )
                st.caption("Uses traditional CV (SVM, Color Theory)")

            if run_dl:
                # Clear previous session logs so UI doesn't duplicate
                st.session_state.logs = []
                st.session_state.latency = None
                st.session_state.eval_state = {
                    "running": False,
                    "done": False,
                    "results": None,
                    "robust_results": None,
                }
                st.session_state.last_agent = "DL Agent"
                st.session_state.non_dl_recs = None
                st.session_state.trigger_agent = "DL Agent"
                st.rerun()

            elif run_non_dl:
                # Clear previous session logs so UI doesn't duplicate
                st.session_state.logs = []
                st.session_state.latency = None
                st.session_state.eval_state = {
                    "running": False,
                    "done": False,
                    "results": None,
                    "robust_results": None,
                }
                st.session_state.last_agent = "Non-DL Agent"
                st.session_state.non_dl_recs = None
                st.session_state.trigger_agent = "Non-DL Agent"
                st.rerun()

            if st.session_state.trigger_agent == "DL Agent":
                st.session_state.trigger_agent = None
                # 2. Planning + Agent Execution
                with st.chat_message("assistant"):
                    st.write("✨ Analyzing your piece with DL Agent...")

                    # Convert image to base64 so gpt can read it
                    img_b64 = pil_to_base64(raw_img)

                    output_area = st.empty()
                    full_response = ""

                    start_time = time.time()
                    # call agent, passing the b64 string to the chat and image path
                    search_successful = False
                    for chunk in st.session_state.agent.chat(img_b64, image_path):
                        # Check if a successful search happened by looking at the logs
                        search_successful = any(
                            log.get("action") == "find_similar_items"
                            and "Error" not in str(log.get("result", ""))
                            for log in st.session_state.logs
                        )
                        if not search_successful:
                            full_response += chunk
                            output_area.markdown(full_response + "▌")

                    if not search_successful:
                        output_area.markdown(full_response)
                    else:
                        output_area.empty()
                    st.session_state.latency = time.time() - start_time

            elif st.session_state.trigger_agent == "Non-DL Agent":
                st.session_state.trigger_agent = None
                with st.chat_message("assistant"):
                    st.write("✨ Analyzing your piece with Non-DL techniques...")
                    start_time = time.time()
                    final_data = llm_agent_with_function_calling(image_path)
                    st.session_state.latency = time.time() - start_time

                    st.session_state.non_dl_recs = final_data.get("find_recs", {}).get(
                        "recommendations", []
                    )

        # --- 3. Visual Results ---
        if st.session_state.last_agent == "DL Agent":
            # we look for the most recent search results to display
            search_results = None
            for log in reversed(st.session_state.logs):
                if log["action"] == "find_similar_items":
                    try:
                        parsed = log["result"]
                        # Parse from string if necessary
                        if isinstance(parsed, str):
                            parsed = json.loads(parsed)
                        # Handle cases where the JSON is double-encoded
                        if isinstance(parsed, str):
                            parsed = json.loads(parsed)
                        if isinstance(parsed, dict):
                            search_results = parsed
                            break  # Get the latest one
                    except:
                        continue

            if search_results:
                matches = search_results.get("matches", [])[:9]

                if (
                    not st.session_state.eval_state["running"]
                    and not st.session_state.eval_state["done"]
                ):
                    st.session_state.eval_state["running"] = True
                    top_4_ids = [str(match.get("id")) for match in matches[:4]]

                    def run_eval_bg(img_path, ids, state):
                        res = evaluate_aesthetic(img_path, ids)
                        robust_res = evaluate_robustness(img_path, ids, agent_type="dl")
                        state["results"] = res
                        state["robust_results"] = robust_res
                        state["running"] = False
                        state["done"] = True

                    t = threading.Thread(
                        target=run_eval_bg,
                        args=(image_path, top_4_ids, st.session_state.eval_state),
                    )
                    add_script_run_ctx(t)
                    t.start()

                st.write(
                    f"Done! Here are {len(matches)} item recommendations based on visual similarity, style, and the article of clothing captured in your image."
                )

                for idx, match in enumerate(matches):
                    meta = match.get("metadata", {})
                    # Perception check: ensure the URL exists
                    if "image_url" in meta:
                        st.write(f"**{idx + 1}. {meta.get('name', 'Zara Item')}**")
                        st.image(meta["image_url"], width=250)
                        st.link_button("Buy on Zara", meta.get("product_url", "#"))
                        st.write("")  # Extra spacing between items

        elif (
            st.session_state.last_agent == "Non-DL Agent"
            and getattr(st.session_state, "non_dl_recs", None) is not None
        ):
            recs = st.session_state.non_dl_recs
            if not recs:
                st.write(
                    "No exact recommendations returned. Check agent logs for general style advice."
                )
            else:
                if (
                    not st.session_state.eval_state["running"]
                    and not st.session_state.eval_state["done"]
                ):
                    st.session_state.eval_state["running"] = True
                    top_4_ids = [str(r) for r in recs[:4]]

                    def run_eval_bg(img_path, ids, state):
                        res = evaluate_aesthetic(img_path, ids)
                        robust_res = evaluate_robustness(
                            img_path, ids, agent_type="non-dl"
                        )
                        state["results"] = res
                        state["robust_results"] = robust_res
                        state["running"] = False
                        state["done"] = True

                    t = threading.Thread(
                        target=run_eval_bg,
                        args=(
                            image_path,
                            top_4_ids,
                            st.session_state.eval_state,
                        ),
                    )
                    add_script_run_ctx(t)
                    t.start()

                st.write(
                    f"Done! Here are {len(recs)} item recommendations based on color theory, style, and the article of clothing captured in your image."
                )
                try:
                    catalog_df = pd.read_csv(
                        "data/zara_combined.csv", index_col="reference"
                    )
                    for idx, rid in enumerate(recs):
                        try:
                            item_info = catalog_df.loc[int(rid)]
                            if isinstance(item_info, pd.DataFrame):
                                item_info = item_info.iloc[0]

                            item_name = item_info.get("name", f"Zara Item {rid}")
                            st.write(f"**{idx + 1}. {item_name}**")
                            img_url = item_info.get("image_url", "")
                            if img_url:
                                st.image(img_url, width=250)
                            prod_url = item_info.get(
                                "url", item_info.get("product_url", "#")
                            )
                            st.link_button("Buy on Zara", prod_url)
                            st.write("")  # Extra spacing between items
                        except:
                            continue
                except Exception as e:
                    st.error(f"Could not load catalog to display items: {e}")

# IMPLEMENT EVAL
with tab_eval:
    st.header("Agent Performance")
    if st.session_state.latency is not None:
        st.subheader(f"Results for: {st.session_state.last_agent}")
        st.metric(
            label="⏱️ Latency (Execution Time) of Generating Recommendations",
            value=f"{st.session_state.latency:.2f} seconds",
        )
        if st.session_state.eval_state["running"]:
            st.info(
                "🔄 Robustness & aesthetic evaluations are running in the background. Results will appear here shortly."
            )
            time.sleep(2)
            st.rerun()
        elif st.session_state.eval_state["done"]:
            results = st.session_state.eval_state["results"]
            robust_results = st.session_state.eval_state.get("robust_results")
            if robust_results:
                col1, col2 = st.columns(2)
                col1.metric(
                    label="Robustness: Blurred Image Overlap",
                    value=f"{robust_results['blur_overlap']:.2f}%",
                )
                col2.metric(
                    label="Robustness: Darkened Image Overlap",
                    value=f"{robust_results['dark_overlap']:.2f}%",
                )
            if results:
                st.metric(
                    label=" Average Aesthetic Score (Generated by a Fashion Expert LLM-as-Judge)",
                    value=f"{results['average_score']:.2f} / 5",
                )
                st.write(
                    "### The Fashion Expert's Reasoning on Why These Recs Work (Or Don't)"
                )
                for eval_data in results.get("evaluations", []):
                    st.write(
                        f"- **{eval_data.get('item_name', 'Unknown Item')} (Score: {eval_data.get('score', 'N/A')}/5):** {eval_data.get('reasoning', 'No reasoning provided.')}"
                    )
            else:
                st.error("Evaluation failed to return all results.")
    else:
        st.info("Run an agent in the Style Assistant tab to see evaluation metrics.")

# Sidebar Logs to see thinking
with st.sidebar:
    st.header("🕵️ Agent Logic")
    if st.button("🗑 Reset Session"):
        st.session_state.agent.reset()
        st.session_state.messages = []
        st.session_state.logs = []
        st.session_state.latency = None
        st.session_state.last_agent = None
        st.session_state.trigger_agent = None
        st.session_state.eval_state = {
            "running": False,
            "done": False,
            "results": None,
            "robust_results": None,
        }
        st.session_state.non_dl_recs = None
        st.session_state.clear()
        st.rerun()

    for log in st.session_state.logs:
        with st.expander(f"Action: {log['action']}"):
            st.write(log["result"])
