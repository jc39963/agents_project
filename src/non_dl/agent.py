import json
import openai
from dotenv import load_dotenv
import os
import time
from .identify_type import identify_type
from .identify_color import get_dominant_rgb
from .color_match import color_theory_test
from .item_combos import create_combo
from .search_catalog import find_recs

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def llm_agent_with_function_calling(
    image_path,
    goal="find items from catalog that can be paired with the given clothing",
):

    state = {"image": image_path, "goal": goal, "data": {}}

    # Defining the tools available to the LLM
    tools_definition = [
        {
            "type": "function",
            "function": {
                "name": "identify_type",
                "description": "Analyzes an image to determine the clothing item's type (e.g., 'shirt', 'pants') and extracts its feature vector for style comparison.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "img_path": {
                            "type": "string",
                            "description": "The file path to the image to analyze.",
                        }
                    },
                    "required": ["img_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_dominant_rgb",
                "description": "Analyzes an image to find the dominant color and returns it as an RGB tuple.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "img_path": {
                            "type": "string",
                            "description": "The file path to the image to analyze.",
                        }
                    },
                    "required": ["img_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "create_combo",
                "description": "Given a type of clothing item, suggests two other types of items that pair well with it.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "item_type": {
                            "type": "string",
                            "description": "The type of clothing item (e.g., 'shirt', 'pants') obtained from 'identify_type'.",
                        }
                    },
                    "required": ["item_type"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "color_theory_test",
                "description": "Given a source RGB color, finds two other colors that form a triadic color harmony.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "rgb_color": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "The source RGB color as a list of three numbers (e.g., [255, 0, 0]) obtained from 'get_dominant_rgb'.",
                        }
                    },
                    "required": ["rgb_color"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "find_recs",
                "description": "Searches the product catalog for recommended items. This is the final step and requires all prior information: the list of item types, and the list of target colors.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "rgb_list": {
                            "type": "array",
                            "description": "A list of two target RGB color lists for the recommended items, from 'color_theory_test'.",
                            "items": {"type": "array", "items": {"type": "number"}},
                        },
                        "item_list": {
                            "type": "array",
                            "description": "A list of two target item types to search for (e.g., ['pants', 'jackets']), from 'create_combo'.",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["rgb_list", "item_list"],
                },
            },
        },
    ]

    messages = [
        {
            "role": "system",
            "content": f"""You are a fashion advisor agent. Your goal: {goal}

You have access to tools to identify items and colors in images, find complementary items and colors, and search a product catalog.

Work step by step:
1. First identify what item is in the image
2. Identify its color
3. Find what items pair well with it
4. Find what colors work well
5. Search the catalog for recommendations
6. Once you have search results, provide final recommendations. You should return a list of specific item IDs from the catalog search as your final output. If you are unable to find specific items or encounter an error, clearly state: 'Error, unable to find specific items, but here are general recommendations based on your item:' and provide general style recommendations based on the item type and colors.""",
        },
        {
            "role": "user",
            "content": f"Help me find items that I can wear with the item in this image: {image_path}",
        },
    ]

    max_iterations = 5
    total_start_time = time.time()

    for iteration in range(max_iterations):
        iter_start_time = time.time()
        print(f"\n--- Iteration {iteration + 1} ---")
        iter_log = f"**Iteration {iteration + 1}**\n\n"

        # Call OpenAI with function calling
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools_definition,
            tool_choice="auto",
        )

        response_message = response.choices[0].message
        if response_message.content:
            iter_log += f"**Agent Thought:**\n{response_message.content}\n\n"

        if response_message.tool_calls:
            # Add assistant's response to messages
            messages.append(response_message)

            # Execute tool call that the LLM thinks should be executed
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                iter_log += f"🛠 **Calling Tool:** `{function_name}`\n**Args:** {function_args}\n"
                result = execute_function(function_name, function_args, state)

                if "error" in result:
                    print(f"Tool Call Error: {result['error']}")
                    iter_log += f"❌ **Error:** {result['error']}\n\n"
                else:
                    iter_log += f"✅ **Result:** {result}\n\n"

                # Add tool call result to state so agent can track it for next round
                state["data"][function_name] = result

                # Add result to messages
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": json.dumps(result),
                    }
                )

        else:
            print("Agent finished reasoning")
            print(response_message.content)
            iter_log += "Agent finished reasoning.\n"
            try:
                import streamlit as st

                if "logs" in st.session_state:
                    st.session_state.logs.append(
                        {
                            "action": f"Non-DL Iteration {iteration + 1}",
                            "result": iter_log,
                        }
                    )
            except ImportError:
                pass
            break

        try:
            import streamlit as st

            if "logs" in st.session_state:
                st.session_state.logs.append(
                    {"action": f"Non-DL Iteration {iteration + 1}", "result": iter_log}
                )
        except ImportError:
            pass

        iter_end_time = time.time()
        print(
            f"--- Iteration {iteration + 1} took {iter_end_time - iter_start_time:.2f} seconds ---"
        )

    total_end_time = time.time()
    print(
        f"\n--- Total execution time: {total_end_time - total_start_time:.2f} seconds ---"
    )
    return state["data"]


def execute_function(function_name, arguments, state=None):
    print(f"Calling: {function_name} with {arguments}")

    try:
        if function_name == "identify_type":
            item_type, feat_vec = identify_type(img_path=arguments["img_path"])
            if state is not None:
                state["data"]["hidden_feature_vector"] = feat_vec
            return {
                "item_type": item_type,
                "status": "Feature vector extracted and saved internally.",
            }

        elif function_name == "get_dominant_rgb":
            rgb_tuple = get_dominant_rgb(img_path=arguments["img_path"])
            return {"dominant_rgb": list(rgb_tuple)}

        elif function_name == "create_combo":
            combo_list = create_combo(item_type=arguments["item_type"])
            return {"item_combos": combo_list}

        elif function_name == "color_theory_test":
            matches = color_theory_test(rgb=arguments["rgb_color"])
            return {"triadic_colors": matches}

        elif function_name == "find_recs":
            if state and "hidden_feature_vector" in state["data"]:
                arguments["feat_vec"] = state["data"]["hidden_feature_vector"]
            recommendations = find_recs(**arguments)
            return {"recommendations": recommendations}

        else:
            return {"error": f"Unknown function: {function_name}"}
    except Exception as e:
        return {"error": f"Error executing {function_name}: {e}"}


if __name__ == "__main__":
    image_to_process = "data/images/captured.jpg"

    if not os.path.exists(image_to_process):
        print(f"Error: Image not found at '{image_to_process}'")
        print("Please capture an image first.")
    else:
        print(f"Found image: '{image_to_process}'. Starting agent...")
        final_data = llm_agent_with_function_calling(
            image_path=image_to_process,
        )
        print("\n--- Agent's Final Data ---")
        print(json.dumps(final_data, indent=2))
