"""
Returns
    - torso exists
    - where it is located
    - how big the torso is w.r.t. the entire image

Only works for realistic drawings right now.
TODO: look into training a Haar cascade for more cartoony styles
"""
import cv2
import util
import imutils

IS_DEBUG = True
NUM_LOC_BINS = 5
MAX_VAL = 100


def getFeatureName():
    return ["torso_exists", "torso_size"] + \
        util.binFeatureNames("torso_x", NUM_LOC_BINS, MAX_VAL) + \
        util.binFeatureNames("torso_y", NUM_LOC_BINS, MAX_VAL)


def extractFeature(img):
    img_h, img_w = img.shape[:2]

    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # find torso
    torso_cascade = cv2.CascadeClassifier('extractors/cascades/haarcascade_headshoulders.xml')

    # for debugging
    if IS_DEBUG:
        torso_cascade = cv2.CascadeClassifier('cascades/haarcascade_headshoulders.xml')

    # Check if it's a frontal face first
    torso = torso_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30),
        flags = 0
        )

    # return none if there are no torso
    if len(torso) == 0:
        return [0] * len(getFeatureName())

    # we only want one per image, so take the first one always
    (x, y, w, h) = torso[0]

    # torso size, as percentage of image
    torso_size = float(w * h) / (img_h * img_w)

    # torso location, as percentage
    torso_loc_x = [0] * NUM_LOC_BINS
    torso_loc_y = [0] * NUM_LOC_BINS
    torso_loc_x[util.getBinIndex(100*(float(x + float(w) / 2) / img_w),
                                 NUM_LOC_BINS, MAX_VAL)] = 1
    torso_loc_y[util.getBinIndex(100*(float(y + float(h) / 2) / img_h),
                                 NUM_LOC_BINS, MAX_VAL)] = 1

    # ready the features for returning
    features = [1, torso_size] + torso_loc_x + torso_loc_y

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
