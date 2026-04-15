# read in image of clothing
# detect clothes, pull out color etc? 
# will call on webcam.py? or just read from data/images

# non dl version, just draw a rectangle or box and get median or mean of colors?
# or when doing webcam, tell them to center in frame?

import cv2
import numpy as np 

img = cv2.imread("image.jpg")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# define range of blue color in HSV 
lower_blue = np.array([110,50,50])
upper_blue = np.array

