import json 
from openai import OpenAI
from typing import Dict, List, Any, Generator
import os 
from dotenv import load_dotenv
from matching import TOOL_FUNCTIONS, TOOL_SCHEMAS
import streamlit as st

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
SYSTEM_PROMPT = """
    You are an expert Fashion Stylist, you will receive an image file which is a user upload of an image. Your task is to generate 1-5 items of clothes that would match and go with the user uploaded piece of clothing. 
    You will output several options for things to go with the clothing item for an outfit and search for similar clothing items in a database of Zara clothes, finding the top k best picks based on your expert style judgment for current popular fashion. 
    For EVERY request this is your process:
    1. ALWAYS call get_image_embedding on the image taken 
    2. From the image embedding returned, generate 1-5 clothing items to go with the uploaded item to complete the outfit, think of multiple aesthetics that users might like to give variet of suggestions. 
    3. Based on the similar Zara items to the ones you suggested, present the top 10 for the user
    KEY RULES:
    - ALWAYS use the `thought` parameter in your tool calls to explain your reasoning before executing them. 
    - ALWAYS explain your reasoning before executing tool calls or if you have to resort to generating a CLIP image.
    - If nothing in the Zara database is similar or a good match, consider having CLIP generate an image of what you wanted to display
    - If there is no inputted aesthetic, go with what you think as an expert is most fashionable currently, OR provide a variety of options covering different aesthetics or crowdpleasers (e.g. preppy, bohemian, casual, basics, clean girl)

"""
class Agent():
    def __init__(self):
        #self.name = name 
        #self.role = role
        #self.tools = {tool.__name__: tool for tool in tools}
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.last_run_metadata = {"steps": 0, "tool_names": []}
        self.current_run_logs = []
    # incase reset needed
    def reset(self):
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.last_run_metadata = {"steps": 0, "tool_names": []}
        self.current_run_logs = []
    # actual agent loop
    def chat(self, message: str) -> Generator[str, None, None]:
        self.messages.append({"role": "user", "content": message})
        steps = 0 
        # track metadata for logging and evaluation
        self.last_run_metadata['steps'] = 0
        # enter agent tool loop
        while True:
            steps += 1
            self.last_run_metadata['steps'] += 1
            # generate actual agent response
            response = client.chat.completions.create(model = "gpt-4.1-mini", messages = self.messages, 
                            tools = TOOL_SCHEMAS, 
                            tool_choice = "auto")
            # check if llm wants to talk or act
            if not response.choices[0].message.tool_calls:
                content = response.choices[0].message.content
                # add to history
                self.messages.append({"role": "assistant", "content": content})
                # Yield in chunks to simulate a stream for the UI
                chunk_size = 20
                for i in range(0, len(content), chunk_size):
                    yield content[i:i+chunk_size]
                break
            else:
                # append and print action
                self.messages.append(response.choices[0].message)
                print(response.choices[0].message)
                # if tool call
                for tool_call in response.choices[0].message.tool_calls:
                    # basically run or execute this tool 
                    function_name = tool_call.function.name
                    # parse JSON string into dict
                    args = json.loads(tool_call.function.arguments)
                    
                    # Extract the thought process and log it so we can see how it's deciding
                    thought = args.pop("thought", None)
                    if thought:
                        st.session_state.logs.append({
                            "action": f"Thought ({function_name})",
                            "result": thought
                        })
                    # call tool
                    try:
                        fn = TOOL_FUNCTIONS[function_name]
                        result = fn(**args)
                    except Exception as e:
                        result = f"Error: {str(e)}"
                    # log it and display in streamlit 
                    log_entry = {"action": function_name, "result": result}
                    self.current_run_logs.append(log_entry)
                    self.last_run_metadata["tool_names"].append(function_name)
                    st.session_state.logs.append({
                    "action": function_name,
                    "result": json.dumps(result, indent=2)
                    })
                    self.messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)})
    
