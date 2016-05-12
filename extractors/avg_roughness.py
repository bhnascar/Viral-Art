"""
Returns the roughness of the image, measured in hsv
TODO: should we combine the measures into one somehow?
"""
import cv2
import imutils
import numpy as np


def getFeatureName():
    return ["Average_Hue_Roughness",
            "Average_Saturation_Roughness",
            "Average_Value_Roughness"]


def extractFeature(img):
    # convert to hsv
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # get the gradients
    h_grad = np.diff(img[:, :, 0])
    s_grad = np.diff(img[:, :, 1])
    v_grad = np.diff(img[:, :, 2])

    return [np.mean(h_grad), np.mean(s_grad), np.mean(v_grad)]

def main():
    cv_image = cv2.imread("test_data/strider.jpg")
    cv_image = imutils.resize(cv_image, width=200)
    print extractFeature(cv_image)

if __name__ == "__main__":
    main()
