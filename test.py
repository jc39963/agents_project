from PIL import Image

from urllib.parse import urljoin

# import cv2
import torch
import numpy as np
import requests
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel
from src.utils import get_db
