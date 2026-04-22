import cv2
import numpy as np
import colorsys

def get_dominant_rgb(img_path: str ='data/images/captured.jpg') -> tuple: 
    """Uses K-means clustering to find the dominant color in an image.

    Args:
        img_path (str, optional): Path to image of clothing item whose color we want to identify. Defaults to 'data/images/captured.jpg'.

    Returns:
        tuple: RGB code for the dominant color.
    """
    img = cv2.imread(img_path)
    img_small = cv2.resize(img, (50, 50))
    pixels = img_small.reshape(-1, 3).astype(np.float32)

    # Find the dominant color (k=1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, _, centers = cv2.kmeans(pixels, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert BGR (OpenCV) to RGB
    b, g, r = centers[0]
    return (float(r), float(g), float(b))

# Test
# source_rgb = get_dominant_rgb('data/images/captured.jpg')
# print(source_rgb)