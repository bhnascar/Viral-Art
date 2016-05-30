"""
Returns the roughness of the image, measured in hsv
"""
import cv2
import imutils
import numpy as np
import os
import util


def getFeatureName():
    return ["Average_Hue_Roughness",
            "Average_Saturation_Roughness",
            "Average_Value_Roughness",
            "Average_Light_Roughness"]


def extractFeature(img):
    img = imutils.resize(img, width=200)
    # convert to hsv
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    # get the gradients
    h_grad = np.diff(img[:, :, 0])
    s_grad = np.diff(img[:, :, 1])
    v_grad = np.diff(img[:, :, 2])
    l_grad = np.diff(hls_img[:, :, 1])

    # range found via manual inspection (aka it's shit)
    mini = 20
    maxi = 150
    return [util.normalize(np.mean(h_grad), mini, maxi),
            util.normalize(np.mean(s_grad), mini, maxi),
            util.normalize(np.mean(v_grad), mini, maxi),
            util.normalize(np.mean(l_grad), mini, maxi)]


def main():
    for filename in os.listdir('test_data/'):
        path = 'test_data/' + filename
        if os.path.isdir(path):
            continue
        print filename
        cv_image = cv2.imread(path)
        print extractFeature(cv_image)

    # cv_image = cv2.imread("test_data/wonder_woman.jpg")
    # print extractFeature(cv_image)

if __name__ == "__main__":
    main()
