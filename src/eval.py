import cv2
import os
import json
import time
import base64
import pandas as pd
import openai
from src.non_dl.agent import llm_agent_with_function_calling
from src.dl.agent import Agent

# from non_dl.agent import llm_agent_with_function_calling
from dotenv import load_dotenv

# eval tab should show latency (returned by original agent call), 2 overlaps (w/ images), and aesthetic eval


def calculate_overlap(original_list: list[str], new_list: list[str]) -> float:
    if not original_list:
        return 0.0

    set_orig = set(original_list)
    set_new = set(new_list)

    # How many items from new list are present in original one
    overlap_count = len(set_orig.intersection(set_new))

    # Calculate percentage overlap based on the size of original list
    overlap_percentage = (overlap_count / len(set_orig)) * 100
    return overlap_percentage


def encode_image(image_path: str) -> str:
    """Encodes a local image file into a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def evaluate_robustness(
    image_path: str, original_item_ids: list[str], agent_type: str = "non-dl"
):
    """Tests agent robustness by blurring and darkening the image, then calculating result overlap."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image: {image_path}")
        return

    # File paths for the perturbed images
    base_dir = os.path.dirname(image_path)
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)

    blurred_path = os.path.join(base_dir, f"{name}_blurred{ext}")
    dark_path = os.path.join(base_dir, f"{name}_dark{ext}")

    # Apply a strong Gaussian blur
    blurred_img = cv2.GaussianBlur(img, (25, 25), 0)
    cv2.imwrite(blurred_path, blurred_img)

    # Wait to avoid rate limits
    time.sleep(3)

    print(f"\n[Test 1] Running {agent_type} agent on blurred image: {blurred_path}")
    blurred_recs = []
    if agent_type == "non-dl":
        blurred_data = llm_agent_with_function_calling(
            blurred_path, goal="find items for blurred image"
        )
        blurred_recs = [
            str(r) for r in blurred_data.get("find_recs", {}).get("recommendations", [])
        ]
    elif agent_type == "dl":
        dl_agent = Agent(log_to_ui=False)
        blurred_b64 = encode_image(blurred_path)
        for _ in dl_agent.chat(blurred_b64, blurred_path):
            pass
        for log in reversed(dl_agent.current_run_logs):
            if log["action"] == "find_similar_items":
                res = log["result"]
                if isinstance(res, str):
                    try:
                        res = json.loads(res)
                    except:
                        pass
                if isinstance(res, str):
                    try:
                        res = json.loads(res)
                    except:
                        pass
                if isinstance(res, dict):
                    matches = res.get("matches", [])
                    blurred_recs = [str(m.get("id")) for m in matches[:4]]
                    break

    blur_overlap = calculate_overlap(original_item_ids, blurred_recs)
    print(f"Blur Test Recommendations: {blurred_recs}")
    print(f"Blur Test Overlap: {blur_overlap:.2f}%")

    # Reduce brightness by 50% (alpha=0.5)
    dark_img = cv2.convertScaleAbs(img, alpha=0.5, beta=0)
    cv2.imwrite(dark_path, dark_img)

    # Wait to avoid rate limits
    time.sleep(3)

    print(f"\n[Test 2] Running {agent_type} agent on darkened image: {dark_path}")
    dark_recs = []
    if agent_type == "non-dl":
        dark_data = llm_agent_with_function_calling(
            dark_path, goal="find items for dark image"
        )
        dark_recs = [
            str(r) for r in dark_data.get("find_recs", {}).get("recommendations", [])
        ]
    elif agent_type == "dl":
        dl_agent = Agent(log_to_ui=False)
        dark_b64 = encode_image(dark_path)
        for _ in dl_agent.chat(dark_b64, dark_path):
            pass
        for log in reversed(dl_agent.current_run_logs):
            if log["action"] == "find_similar_items":
                res = log["result"]
                if isinstance(res, str):
                    try:
                        res = json.loads(res)
                    except:
                        pass
                if isinstance(res, str):
                    try:
                        res = json.loads(res)
                    except:
                        pass
                if isinstance(res, dict):
                    matches = res.get("matches", [])
                    dark_recs = [str(m.get("id")) for m in matches[:4]]
                    break

    dark_overlap = calculate_overlap(original_item_ids, dark_recs)
    print(f"Dark Test Recommendations: {dark_recs}")
    print(f"Dark Test Overlap: {dark_overlap:.2f}%")

    return {"blur_overlap": blur_overlap, "dark_overlap": dark_overlap}


def evaluate_aesthetic(image_path: str, original_item_ids: list[str]) -> dict:
    """Uses a multimodal LLM as a 'judge' to evaluate the aesthetic match of recommended items."""
    print("\n=== Starting Aesthetic Evaluation (LLM as Judge) ===")

    try:
        # Assuming the CSV has 'id' and 'image_url' columns
        catalog_df = pd.read_csv("data/zara_combined.csv", index_col="reference")
    except FileNotFoundError:
        print("Error: Catalog file 'data/zara_combined.csv' not found.")
        return None
    except (KeyError, ValueError):
        print("Error: Catalog CSV must have a 'reference' column to be used as index.")
        return None

    # Setup OpenAI client for the judge
    load_dotenv()
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    base64_original_image = encode_image(image_path)

    judge_prompt = """You are a world-class fashion expert and stylist. Your task is to evaluate how well a recommended clothing item pairs with an original item provided by a user.

You will be given multiple images:
1. The user's original clothing item.
2. A series of recommended clothing items from a catalog, each preceded by its Item ID.

Based on these images, evaluate how well EACH recommended item pairs with the ORIGINAL item. Consider the following criteria:
- **Color Harmony:** Do the colors complement each other? 
- **Style Cohesion:** Do the styles of the items match? (e.g., casual with casual, formal with formal).
- **Proportion and Silhouette:** Would the shapes and lengths of these items create a balanced and aesthetically pleasing outfit?

Your response MUST be a JSON object with the following structure:
{
  "evaluations": [
    {
      "item_id": "<the ID of the recommended item>",
      "score": <an integer score from 1 (poor match) to 5 (excellent match)>,
      "reasoning": "<a string explaining your score based on color, style, and proportion>"
    }
  ]
}"""

    # Build the message content with the original image first
    content_array = [
        {"type": "text", "text": "Here is the user's original clothing item:"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_original_image}"},
        },
        {
            "type": "text",
            "text": "Here are the recommended items from the catalog. Please evaluate each one against the original item above.",
        },
    ]

    valid_items_found = False
    item_id_to_name = {}
    for item_id in original_item_ids:
        try:
            # Get recommended item's image URL from the catalog
            item_info = catalog_df.loc[int(item_id)]
            if isinstance(item_info, pd.DataFrame):
                item_info = item_info.iloc[0]
            rec_image_url = item_info["image_url"]  # This column should exist!

            item_name = item_info.get("name", f"Item {item_id}")
            item_id_to_name[str(item_id)] = item_name

            content_array.append({"type": "text", "text": f"Item ID: {item_id}"})
            content_array.append(
                {"type": "image_url", "image_url": {"url": rec_image_url}}
            )
            valid_items_found = True

        except (KeyError, ValueError, TypeError):
            print(f"Warning: Item ID '{item_id}' not found in catalog. Skipping.")

    if not valid_items_found:
        print("No valid images found in the catalog to evaluate.")
        return None

    print(f"\nSending batch evaluation request for {len(original_item_ids)} items...")
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": judge_prompt},
                {"role": "user", "content": content_array},
            ],
        )

        judge_result = json.loads(response.choices[0].message.content)
        aesthetic_scores = judge_result.get("evaluations", [])

        for eval_data in aesthetic_scores:
            rec_id = str(eval_data.get("item_id"))
            item_name = item_id_to_name.get(rec_id, f"Item {rec_id}")
            eval_data["item_name"] = item_name
            print(
                f"Item: {item_name} | Score: {eval_data.get('score', 'N/A')}/5 | Reasoning: {eval_data.get('reasoning', 'N/A')}"
            )

        total_score = sum(item.get("score", 0) for item in aesthetic_scores)
        average_score = total_score / len(aesthetic_scores)
        print(f"\n--- Average Aesthetic Score: {average_score:.2f}/5 ---")

        return {"average_score": average_score, "evaluations": aesthetic_scores}

    except Exception as e:
        print(f"An error occurred during aesthetic evaluation: {e}")
        return None


if __name__ == "__main__":
    # Example Usage Flow
    original_image = "data/images/captured.jpg"

    # For testing, you would normally run the baseline first to get its IDs, like this:
    print("Getting baseline recommendations...")
    baseline_data = llm_agent_with_function_calling(original_image)
    baseline_recs = baseline_data.get("find_recs", {}).get("recommendations", [])
    print(f"Baseline Recommendations: {baseline_recs}")

    if baseline_recs:
        evaluate_robustness(original_image, baseline_recs)
        evaluate_aesthetic(original_image, baseline_recs)
    else:
        print(
            "No baseline recommendations found to compare against. Did the agent hit an error?"
        )
