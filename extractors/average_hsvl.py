"""
Returns the average hue, saturation, and value of the imagel
"""
import cv2
import imutils
import numpy as np
import util

def getFeatureName():
    return ["Average_Hue", "Average_Saturation", "Average_Value",
            "Average_Light"]

def extractFeature(img):
    # Get dominant color scheme via k-means
    # convert to hsv        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    # reshape the image to be a list of pixels
    img = img.reshape((img.shape[0] * img.shape[1], 3))
    avg_h = np.mean(img[:, 0])
    avg_s = np.mean(img[:, 1])
    avg_v = np.mean(img[:, 2])
    avg_l = np.mean(hls_img[:, 1])

    # normalize
    avg_h = float(avg_h) / util.MAX_HUE
    avg_s = float(avg_s) / util.MAX_SAT
    avg_v = float(avg_v) / util.MAX_VAL
    avg_l = float(avg_l) / util.MAX_LIGHT

    return [avg_h, avg_s, avg_v, avg_l]

def main():
    cv_image = cv2.imread("test_data/wonder_woman.jpg")
    cv_image = imutils.resize(cv_image, width=200)
    print extractFeature(cv_image)

if __name__ == "__main__":
    main()