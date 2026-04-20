from PIL import Image
import pandas as pd 

from urllib.parse import urljoin
import requests 
#import cv2 
import numpy as np 
from PIL import Image
import requests
from io import BytesIO
from generate_embeddings import encode_images, encode_texts
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
#from sentence_transformers import SentenceTransformer, util
from data.database import get_db

# yolo detect image from webcam
# use get_embeddings from generate_embeddings
# agent read this and give aesthetic, gives some matching recs based on vibe query too? search pinecone db for similar clothes 
# iterates based on clothes given?
# # clip also generates what that agent match is and look at that example vs zara one?



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
    final_id = []
    for match in results.matches:
        print(f"ID: {match.id} | Score: {match.score:.4f} | URL: {match.metadata.get('product_url')}")
        final_id.append(match.id)

    return final_id

def get_image_embedding(image_path):
    return encode_images([image_path])



if __name__ == '__main__':

    #user_embedding = encode_images(['data/images/fur_coat.jpg'])
    find_similar_items("brown pants")