from pinecone import Pinecone, ServerlessSpec
import csv
import os
import cv2
import urllib.request
import numpy as np
import requests
import time
import joblib
import pandas as pd
from src.utils import get_db
from src.non_dl.utils import *
from dotenv import load_dotenv


def download_imgs(csv_path: str):
    """
    Reads products CSV, generates embeddings in batches,
    and upserts them into Pinecone.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Read CSV and download image as jpg from url column
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            product_id = row.get("reference")
            name = row.get("name", "")
            brand = row.get("brand", "")
            category = row.get("category_hint", "")
            color = row.get("color", "")
            image_url = row.get("image_url", "")
            session = requests.Session()
            session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
    'Referer': 'https://www.zara.net/' 
})
            for attempt in range(3): 
              try:
                response = session.get(image_url, timeout=15)
                if response.status_code == 200:
                    with open(f'data/dataset/clothes/{product_id}.jpg', 'wb') as f: # Download image jpgs to data/dataset/clothes folder
                        f.write(response.content)
                        break
                elif response.status_code == 403:
                # If we hit a 403, wait longer
                    time.sleep(2 ** attempt) 
                    if attempt == 2:
                      print(f'Hit 403 after 3 attempts, skipping {product_id}.')
              except:
                pass

def color_and_category(img_id):
  item = df.loc[df['reference'] == int(img_id)]
  category = item['category'].values[0]
  color = item['color'].values[0]
  cleaned_color_rgb = fashion_to_rgb(color)
  return category, cleaned_color_rgb

def upsert_images_batched(f_index, c_index, image_dir, pca, scaler, batch_size=100):
    shape_batch = []
    color_batch = []
    
    hog = cv2.HOGDescriptor(_winSize=(128, 128), _blockSize=(16, 16), 
                            _blockStride=(8, 8), _cellSize=(8, 8), _nbins=9)
    
    file_list = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for i, img_name in enumerate(tqdm(file_list, desc="Processing Batches")):
        img_id = img_name.split('.')[0]
        img_path = os.path.join(image_dir, img_name)
        
        # --- Preprocessing ---
        img = cv2.imread(img_path)
        if img is None: continue
        
        height, width = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        normalized = clahe_grayscale(gray)
        padded = add_padding(normalized, height, width)
        final = cv2.resize(padded, (128, 128))
        
        hog_features = hog.compute(final).reshape(1, -1)   
        post_pca = pca.transform(hog_features)
        post_scaling = scaler.transform(post_pca)
        final_hog_vec = post_scaling.flatten().tolist()

        # Skip any all 0 vectors for now
        if np.all(final_hog_vec==0):
          print(f'Skipping a feature vectors ith all 0s for {img_id}')
          continue
        
        category, cleaned_color_rgb = color_and_category(img_id)
        metadata = {"item_type": str(category)}
        epsilon = 1e-6
        safe_color_vec = [float(x) if x > 0 else epsilon for x in cleaned_color_rgb]
        
        # --- Add to Buffers ---
        shape_batch.append({
            "id": img_id, 
            "values": final_hog_vec, 
            "metadata": metadata
        })
        
        color_batch.append({
            "id": img_id, 
            "values": safe_color_vec, 
            "metadata": metadata
        })
        
        # --- Upsert ---
        if len(shape_batch) >= batch_size:
            f_index.upsert(vectors=shape_batch)
            c_index.upsert(vectors=color_batch)
            # Clear buffers
            shape_batch = []
            color_batch = []

    # --- Final Cleanup ---
    if shape_batch:
        f_index.upsert(vectors=shape_batch)
        c_index.upsert(vectors=color_batch)
        
    print(f"\nSuccessfully indexed {len(file_list)} items.")
    

if __name__ == "__main__":
    load_dotenv()
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    
    # Download JPG images for all catalog items to data folder
    download_imgs('data/zara_combined.csv')
    
    # Load PCA & scaler objects
    pca = joblib.load('/content/pca_model.joblib')
    scaler = joblib.load('/content/scaler_model.joblib')

    # Create Pinecone indices for RGB & HOG feature vectors
    features_index = get_db("product-non-dl-features", 1655, metric='cosine')
    color_index = get_db("product-non-dl-colors", 3, metric='euclidean')
    
    img_dir = 'data/dataset/clothes'
    upsert_images_batched(features_index, color_index, img_dir, pca, scaler)