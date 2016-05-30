"""
Returns
    - fullbody exists
    - where it is located
    - how big it is w.r.t. the entire image

Only works for realistic drawings right now.
TODO: look into training a Haar cascade for more cartoony styles
"""
import cv2
import util
import imutils

IS_DEBUG = False
NUM_LOC_BINS = 5
MAX_VAL = 100


def getFeatureName():
    return ["fullbody_exists", "fullbody_size"] + \
        util.binFeatureNames("fullbody_x", NUM_LOC_BINS, MAX_VAL) + \
        util.binFeatureNames("fullbody_y", NUM_LOC_BINS, MAX_VAL)


def extractFeature(img):
    img_h, img_w = img.shape[:2]

    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # find fullbody
    fullbody_cascade = cv2.CascadeClassifier('extractors/cascades/haarcascade_fullbody.xml')

    # for debugging
    if IS_DEBUG:
        fullbody_cascade = cv2.CascadeClassifier('cascades/haarcascade_fullbody.xml')

    # Check if it's a frontal face first
    fullbody = fullbody_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30),
        flags = 0
        )

    # return none if there are no faces
    if len(fullbody) == 0:
        return [None] * len(getFeatureName())

    # we only want one per image, so take the first one always
    (x, y, w, h) = fullbody[0]

    # fullbody size, as percentage of image
    fullbody_size = float(w * h) / (img_h * img_w)

    # fullbody location, as percentage
    fullbody_loc_x = [0] * NUM_LOC_BINS
    fullbody_loc_y = [0] * NUM_LOC_BINS
    fullbody_loc_x[util.getBinIndex(100*(float(x + float(w) / 2) / img_w),
                                    NUM_LOC_BINS, MAX_VAL)] = 1
    fullbody_loc_y[util.getBinIndex(100*(float(y + float(h) / 2) / img_h),
                                    NUM_LOC_BINS, MAX_VAL)] = 1

    # ready the features for returning
    features = [1, fullbody_size] + fullbody_loc_x + fullbody_loc_y

    assert len(features) == len(getFeatureName()), \
        "length of features matches feature names"

    '''Display and debug'''
    if IS_DEBUG:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        cv2.imshow('img', img)
        cv2.waitKey(0)

    return features


def main():
    cv_image = cv2.imread("test_data/rey.png")
    # cv_image = imutils.resize(cv_image, width=200)

    print extractFeature(cv_image)

if __name__ == "__main__":
    main()
