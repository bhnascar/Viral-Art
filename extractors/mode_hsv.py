"""
Returns the most common hue, saturation, and value of the image
"""
import cv2
import imutils
import numpy as np
from scipy import stats

def getFeatureName():
    return ["Mode_Hue", "Mode_Saturation", "Mode_Value"]

def extractFeature(img):
    # Get dominant color scheme via k-means
    # convert to hsv        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # reshape the image to be a list of pixels
    img = img.reshape((img.shape[0] * img.shape[1], 3))

    h = np.asscalar(stats.mode(img[:, 0])[0])
    s = np.asscalar(stats.mode(img[:, 1])[0])
    v = np.asscalar(stats.mode(img[:, 2])[0])

    return [h, s, v]

def main():
    cv_image = cv2.imread("test_data/wonder_woman.jpg")
    cv_image = imutils.resize(cv_image, width=200)
    print extractFeature(cv_image)

if __name__ == "__main__":
    main()