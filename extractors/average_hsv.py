"""
Returns the average hue, saturation, and value of the imagel
"""
import cv2
import imutils
import numpy as np

def getFeatureName():
    return ["Average_Hue", "Average_Saturation", "Average_Value"]

def extractFeature(img):
    # Get dominant color scheme via k-means
    # convert to hsv        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # reshape the image to be a list of pixels
    img = img.reshape((img.shape[0] * img.shape[1], 3))
    avg_h = np.mean(img[:, 0])
    avg_s = np.mean(img[:, 1])
    avg_v = np.mean(img[:, 2])

    return [avg_h, avg_s, avg_v]

cv_image = cv2.imread("test_data/wonder_woman.jpg")
cv_image = imutils.resize(cv_image, width=200)

print extractFeature(cv_image)