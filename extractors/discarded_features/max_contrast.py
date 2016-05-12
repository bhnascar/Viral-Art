"""
Returns the maximum difference of values in the image
TODO: decide whether this is relevant if we have palette contrast
"""
import cv2
import imutils
import numpy as np


def getFeatureName():
    return ["Max Hue Contrast",
            "Max Saturation Contrast",
            "Max Value Contrast"]


def extractFeature(img):
    # Get dominant color scheme via k-means
    # convert to hsv
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # get the gradients
    h_grad = np.diff(img[:, :, 0])
    s_grad = np.diff(img[:, :, 1])
    v_grad = np.diff(img[:, :, 2])

    return [np.amax(h_grad), np.amax(s_grad), np.amax(v_grad)]

cv_image = cv2.imread("test_data/lizbeth.jpg")
cv_image = imutils.resize(cv_image, width=200)

print extractFeature(cv_image)
