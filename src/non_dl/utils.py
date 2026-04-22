import joblib
import cv2
import numpy as np
import webcolors
import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv


def load_artifacts():
    base_dir = os.path.dirname(__file__)
    svm = joblib.load(os.path.join(base_dir, "artifacts", "svm_model.joblib"))
    pca = joblib.load(os.path.join(base_dir, "artifacts", "pca_model.joblib"))
    scaler = joblib.load(os.path.join(base_dir, "artifacts", "scaler_model.joblib"))
    return svm, pca, scaler


def clahe_grayscale(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    return img


def add_padding(cropped, height, width):
    diff = np.abs(height - width)
    p1 = diff // 2
    p2 = diff - p1
    if height > width:
        padded = cv2.copyMakeBorder(
            cropped, 0, 0, p1, p2, cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
    else:
        padded = cv2.copyMakeBorder(
            cropped, p1, p2, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
    return padded


def fashion_to_rgb(name):
    if not isinstance(name, str) or name.isdigit():
        return (128, 128, 128)  # Gray for invalid/numeric data

    name = name.lower().strip()

    # 1. Direct Overrides for "Fashion Only" Terms
    fashion_dictionary = {
        "vigore": (93, 64, 55),  # Heathered Brown
        "ecru": (250, 249, 246),  # Off-white/Cream
        "tobacco": (110, 66, 30),  # Warm Earthy Brown
        "whiskey": (167, 85, 2),  # Amber Brown
        "anthracite": (56, 56, 56),  # Dark Charcoal
        "mink": (136, 119, 105),  # Muted Taupe
        "taupe": (139, 133, 137),  # Gray-Brown
        "burgundy": (128, 0, 32),  # Deep Red
        "khaki": (195, 176, 145),  # Tan-Khaki
        "bottle green": (0, 106, 78),
    }

    # 2. Check for multi-colors (e.g., 'Ecru / Black')
    # Take the first dominant color mentioned
    if "/" in name:
        name = name.split("/")[0].strip()

    # 3. Keyword Match
    for fashion_key, rgb in fashion_dictionary.items():
        if fashion_key in name:
            return rgb

    # 4. Standard Color Match (Using webcolors)
    standard_colors = [
        "red",
        "blue",
        "green",
        "yellow",
        "black",
        "white",
        "pink",
        "purple",
        "gray",
        "orange",
        "brown",
        "indigo",
        "maroon",
        "silver",
        "tan",
    ]
    for base in standard_colors:
        if base in name:
            try:
                return webcolors.name_to_rgb(base)
            except:
                return (128, 128, 128)

    return (128, 128, 128)  # Fallback to gray


def get_db(name, embed_dim=None, metric=None):
    load_dotenv()
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Only list indexes and create if we are in setup mode (dimensions provided)
    if (embed_dim is not None) and (metric is not None):
        existing_indexes = pc.list_indexes().names()
        if name not in existing_indexes:
            spec = ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            )
            pc.create_index(
                name=name,
                dimension=embed_dim,
                metric=metric,
                spec=spec,
            )
            print(f"Created Pinecone index: {name}")
        else:
            print(f"Pinecone index ready: {name}")

    # Return the index object
    index = pc.Index(name=name)
    return index


def euclidean_similarity_score(vec1, vec2):
    euc = np.linalg.norm(vec1 - vec2)
    similarity_score = 1 / (1 + euc)
    return similarity_score
    # return np.sqrt(np.sum((vec1 - vec2)**2))
