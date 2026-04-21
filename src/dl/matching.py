from PIL import Image
import pandas as pd 

from urllib.parse import urljoin
import requests 
#import cv2 
import numpy as np 
from PIL import Image
import requests
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel 
from data.database import get_db
import torch

# yolo detect image from webcam
# use get_embeddings from generate_embeddings
# agent read this and give aesthetic, gives some matching recs based on vibe query too? search pinecone db for similar clothes 
# iterates based on clothes given?
# # clip also generates what that agent match is and look at that example vs zara one?

# Load the model and processor directly from Hugging Face
model_name = "patrickjohncyh/fashion-clip"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

# Connect to Pinecone index
index = get_db()
def find_similar_items(query, top_k: int = 5):
    """Search Pinecone using llm terms."""


    print(f"\n🔎 Searching for matching clothes")

    # Generate embedding
    query_embedding = encode_texts([query])
    query_vector = query_embedding[0].tolist()

    # Query Pinecone index
    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)

    print("\nTop similar products:")
    output = {"matches": []}
    for match in results.matches:
        print(f"ID: {match.id} | Score: {match.score:.4f} | URL: {match.metadata.get('product_url')}")
        output["matches"].append({
            "id": match.id,
            "score": float(match.score),
            "metadata": match.metadata
        })

    return output

# using image path / jpg
def get_image_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    return encode_images([image])



# encode images to embeddings with CLIP
def encode_images(images):
    processed_images = []
    for img in images:
        if isinstance(img, str):
            processed_images.append(Image.open(img).convert("RGB"))
        else:
            processed_images.append(img)
            
    inputs = processor(images=processed_images, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
        if hasattr(outputs, 'pooler_output'):
            image_features = outputs.pooler_output
        elif isinstance(outputs, (list, tuple)):
            image_features = outputs[0]
        else:
            image_features = outputs
    return image_features / image_features.norm(p=2, dim=-1, keepdim=True) # Normalize

# use CLIP to encode text to embedding
def encode_texts(texts):
    inputs = processor(text=texts, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model.get_text_features(**inputs)
        if hasattr(outputs, 'pooler_output'):
            text_features = outputs.pooler_output
        elif isinstance(outputs, (list, tuple)):
            text_features = outputs[0]
        else:
            text_features = outputs
    return text_features / text_features.norm(p=2, dim=-1, keepdim=True) # Normalize




TOOL_FUNCTIONS = {
    "get_image_embedding": get_image_embedding, 
    "find_similar_items": find_similar_items,
    "encode_images": encode_images, 
    "encode_texts": encode_texts, 
}

TOOL_SCHEMAS = [
    {
    "type": "function",
    "function": {
        "name": "get_image_embedding", 
        "description": ("returns embedding from fashion-CLIP model of inputted image, always call this first and then use output embedding to decide what clothes to pair with."
        ),
        "parameters": {
            "type": "object", 
            "properties": {
                "thought": {
                    "type": "string", 
                    "description": "Explain why you are embedding this image."
                }, 
                "image_path": {
                    "type": "string", 
                    "description": "Path to image to embed."
                },
            },
            "required": ["thought", "image_path"]
        
        },

    },
},
{
    "type": "function",
    "function": {
        "name": "find_similar_items", 
        "description": ("Takes input text or query, encodes the text into an embedding, and searches Pinecone vector database for "
        "similar items of clothing and returns top k results."
        ),
        "parameters": {
            "type": "object", 
            "properties": {
                "thought": {
                    "type": "string", 
                    "description": "Explain why you are searching for similar items and why you picked those items."
                
                },
                "query": {
                    "type": "string", 
                    "description": "Item of clothing query to search for items similar to."
                }, 
                "top_k":{
                    "type": "integer",
                    "description": "number of items to return." 

                },
            },
            "required": ["thought", "query"]
        

        },

    },
},
{
    "type": "function",
    "function": {
        "name": "encode_images", 
        "description": ("Takes input list of image paths, uses CLIP model to turn images into embeddings and returns"
        "as a list of floats"),
        "parameters": {
            "type": "object", 
            "properties": {
                    "thought": {
                        "type": "string",
                        "description": "Explain why you are encoding these images and not just doing a single image with get_image_embedding"
                    },
                    "images": {
                        "type": "array", 
                        "items": {
                            "type": "string",
                            "description": "path to image to embed"
                        },
                    },
                },
                "required": ["thought", "images"]
            },
        
    },

},
{
    "type": "function",
    "function": {
        "name": "encode_texts", 
        "description": ("Takes text input, uses CLIP model to turn text into embeddings and returns embedding "
        "as a list of floats"),
        "parameters": {
            "type": "object", 
            "properties": {
                    "thought": {
                        "type": "string",
                        "description": "Explain why you are encoding this text"
                    },
                    "texts": {
                        "type": "string", 
                        "description": "text to embed"
                    },
                },
                "required": ["texts"]
            },
        
    },

},


]