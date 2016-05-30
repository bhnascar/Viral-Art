"""
Returns the most common hue, saturation, and value of the image
"""
import cv2
import imutils
import numpy as np
from scipy import stats
import util


def getFeatureName():
    return ["Mode_Hue", "Mode_Saturation", "Mode_Value",
            "Mode_Light"]


def extractFeature(img):
    # Get dominant color scheme via k-means
    # convert to hsv
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    # reshape the image to be a list of pixels
    img = img.reshape((img.shape[0] * img.shape[1], 3))
    hls_img = hls_img.reshape((hls_img.shape[0] * hls_img.shape[1], 3))

    h = np.asscalar(stats.mode(img[:, 0])[0])
    s = np.asscalar(stats.mode(img[:, 1])[0])
    v = np.asscalar(stats.mode(img[:, 2])[0])
    l = np.asscalar(stats.mode(hls_img[:, 1])[0])

    return [util.normalize(h, util.MIN_HUE, util.MAX_HUE),
            util.normalize(s, util.MIN_SAT, util.MAX_SAT),
            util.normalize(v, util.MIN_VAL, util.MAX_VAL),
            util.normalize(l, util.MIN_LIGHT, util.MAX_LIGHT)]


def main():
    cv_image = cv2.imread("test_data/wonder_woman.jpg")
    cv_image = imutils.resize(cv_image, width=200)
    print extractFeature(cv_image)

if __name__ == "__main__":
    main()