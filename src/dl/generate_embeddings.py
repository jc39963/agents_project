#from sentence_transformers import SentenceTransformer, util
from PIL import Image
import pandas as pd 
#from bs4 import BeautifulSoup
from urllib.parse import urljoin
import requests 
import cv2 
import numpy as np 
from PIL import Image
import requests
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel 
import torch
from data.database import get_db
import zipfile
from matching import encode_images, encode_texts

# Load the model and processor directly from Hugging Face
model_name = "patrickjohncyh/fashion-clip"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)
BATCH_SIZE = 50
# Get Pinecone index
index = get_db()
# get image to format for clip
def read_images(zip_path):
  all_images = []
  with zipfile.ZipFile(zip_path, 'r') as z:
    for filename in z.namelist():
        # Only process image files
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            with z.open(filename) as f:
                # Read bytes and convert to PIL (required for CLIP preprocess)
                image = Image.open(io.BytesIO(f.read())).convert("RGB")
                all_images.append(image)
  return all_images


def ingest_to_pinecone(embeddings):
        # Process in batches
    for i in range(0, len(embeddings), BATCH_SIZE):
        batch = embeddings[i : i + BATCH_SIZE]



        # Upsert into Pinecone
        vectors_to_upsert = [(pid, emb) for (pid, _), emb in zip(batch, embeddings)]
        index.upsert(vectors_to_upsert)

        print(
            f"Upserted batch {i // BATCH_SIZE + 1} ({len(vectors_to_upsert)} products)"
        )

    print("🎉 All embeddings ingested into Pinecone successfully.")

if __name__ == '__main__':
    
    

    list_of_zara_images = read_images('zarapath.zip')
    zara_embeddings = encode_images(list_of_zara_images)
    ingest_to_pinecone(zara_embeddings)
    # push to pinecone db






