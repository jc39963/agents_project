# use Yolo or something to find clothing in frame
import numpy as np 
#import matplotlib.pyplot as plt 
import random
import os 
import cv2 
import shutil 
import tqdm
import glob 
import torch 
#from ultralytics import YOLO



#print(f"PyTorch version: {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")

#detection_model = YOLO("yolov8n.pt")

# finetuned model from huggingface
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch

CLOTHING_CATEGORIES = {
    1: "Blazer",
    2: "Blouse",
    3: "Cardigan",
    4: "Dress",
    5: "Hoodie",
    6: "Jacket",
    7: "Jeans",
    8: "Nightgown",
    9: "Outerwear",
    10: "Pajamas",
    11: "Rain jacket",
    12: "Rain trousers",
    13: "Robe",
    14: "Shirt",
    15: "Shorts",
    16: "Skirt",
    17: "Sweater",
    18: "T-shirt",
    19: "Tank top",
    20: "Tights",
    21: "Top",
    22: "Training top",
    23: "Trousers",
    24: "Tunic",
    25: "Vest",
    26: "Winter jacket",
    27: "Winter trousers"
}


def load_model():
    # Load model and processor
    processor = AutoImageProcessor.from_pretrained("wargoninnovation/wargon-clothing-classifier")
    model = AutoModelForImageClassification.from_pretrained("wargoninnovation/wargon-clothing-classifier")
    return processor, model


def predict(image_path, processor, model):
    # Load and preprocess image
    image = Image.open(image_path)
    inputs = processor(image, return_tensors="pt")

    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # Get top prediction
    predicted_class_id = predictions.argmax().item()
    predicted_class = CLOTHING_CATEGORIES[predicted_class_id]

    print(predicted_class, predicted_class_id)
    return predicted_class

if __name__ == '__main__':
    processor, model = load_model()

    predict('data/images/fur_coat.jpg', processor, model)