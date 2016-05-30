"""
Returns
    - Face exists
    - how the face is
    - where it is located
    - how big the face is w.r.t. the entire image

    - number of eyes found
    - where the eyes are located
    - how big the eyes are in comparison to the face

    - if they are smiling

Only works for realistic faces right now.
TODO: look into training a Haar cascade for more cartoony faces,
or the other Haar Cascades
"""
import cv2
import util
import imutils
import numpy as np
import os

IS_DEBUG = False
NUM_LOC_BINS = 5
MAX_LOC_VAL = 1


def getEyeFeatureNames():
    return ["eye_size", "number_of_visible_eyes"] + \
        util.binFeatureNames("eye_x_in_img", NUM_LOC_BINS, MAX_LOC_VAL*100) + \
        util.binFeatureNames("eye_y_in_img", NUM_LOC_BINS, MAX_LOC_VAL*100)


def getFaceFeatureNames():
    return ["is_frontal_face", "is_profile_face", "face_size"] + \
        util.binFeatureNames("face_x", NUM_LOC_BINS, MAX_LOC_VAL*100) + \
        util.binFeatureNames("face_y", NUM_LOC_BINS, MAX_LOC_VAL*100)


def getSmileFeatureNames():
    return []
    # return ["there_is_a_smile"]


def getFeatureName():
    return getFaceFeatureNames() + getEyeFeatureNames() + \
        getSmileFeatureNames()


def getSmileFeatures(fx, fy, fw, fh, img, gray):
    '''doesn't work very well'''

    # smile_cascade = cv2.CascadeClassifier('extractors/cascades/haarcascade_smile.xml')

    # # for debugging
    # if IS_DEBUG:
    #     smile_cascade = cv2.CascadeClassifier('cascades/haarcascade_smile.xml')

    # # smile detection
    # smile = smile_cascade.detectMultiScale(
    #     gray,
    #     scaleFactor=1.1,
    #     minNeighbors=5,
    #     minSize=(30, 30),
    #     flags = 0
    # )
    return []


def removeExtraneousEyes(eyes, fx, fy, fw, fh):
    ''' remove any eyes not in the face '''
    to_remove = []
    for index, val in enumerate(eyes):
        (ex, ey, ew, eh) = val
        if ((ex < fx) or (ey < fy) or
           (ex + ew > fx + fw) or (ey + eh > fy + fh)):
            to_remove.append(index)

    eyes = np.delete(eyes, to_remove, axis=0)
    return eyes[:2]


def getEyeAverage(eyes):
    '''If there are 2 eyes in the face, get the average
    (center between the eyes)
    '''
    number_of_visible_eyes = len(eyes)

    avg_x = 0
    avg_y = 0
    avg_w = 0
    avg_h = 0
    for (ex, ey, ew, eh) in eyes:
        avg_x = avg_x + float(ex + float(ew) / 2)
        avg_y = avg_y + float(ey + float(eh) / 2)
        avg_w = avg_w + ew
        avg_h = avg_h + eh
    avg_x = float(avg_x) / number_of_visible_eyes
    avg_y = float(avg_y) / number_of_visible_eyes
    avg_w = float(avg_w) / number_of_visible_eyes
    avg_h = float(avg_h) / number_of_visible_eyes

    return (avg_x, avg_y, avg_w, avg_h)


def getEyeFeatures(fx, fy, fw, fh, img, gray):
    img_h, img_w = img.shape[:2]
    # find eyes
    open_eyes_cascade = cv2.CascadeClassifier('extractors/cascades/haarcascade_eye.xml')

    # for debugging
    if IS_DEBUG:
        open_eyes_cascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')
    # print 'hi'
    # eye detection
    # eyes = open_eyes_cascade.detectMultiScale3(
    eyes = open_eyes_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minSize=(30, 30),
        minNeighbors=5
        # outputRejectLevels = True
    )
    # weights = eyes[2]
    # eyes = eyes[0]
    # print weights

    eyes = removeExtraneousEyes(eyes, fx, fy, fw, fh)

    # Don't do eye features if there are no eyes!!
    if not len(eyes) > 0:
        return [0] * len(getEyeFeatureNames())

    # get the center of the eyes
    (avg_x, avg_y, avg_w, avg_h) = getEyeAverage(eyes)

    # size of the eye in the face
    eye_size = float(avg_w * avg_h) / (fw * fh)

    # placement of eye in the image
    eye_loc_x = [0]*NUM_LOC_BINS
    eye_loc_y = [0]*NUM_LOC_BINS
    eye_loc_x[util.getBinIndex(float(avg_x) / img_w,
                               NUM_LOC_BINS, MAX_LOC_VAL)] = 1
    eye_loc_y[util.getBinIndex(float(avg_y) / img_h,
                               NUM_LOC_BINS, MAX_LOC_VAL)] = 1

    if IS_DEBUG:
        '''Display for debugging'''
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(img, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    features = [eye_size, util.normalize(len(eyes), 0, 2)] + \
        eye_loc_x + eye_loc_y

    assert len(features) == len(getEyeFeatureNames()), \
        "length of eye features matches feature names"

    return features


def extractFeature(img):
    is_frontal_face = 0
    is_profile_face = 0

    img_h, img_w = img.shape[:2]
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # find faces
    front_face_cascade = cv2.CascadeClassifier('extractors/cascades/haarcascade_frontalface_default.xml')
    profile_face_cascade = cv2.CascadeClassifier('extractors/cascades/haarcascade_profileface.xml')

    # for debugging
    if IS_DEBUG:
        front_face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
        profile_face_cascade = cv2.CascadeClassifier('cascades/haarcascade_profileface.xml')

    # Check if it's a frontal face first
    faces = front_face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30),
        flags = 0
    )

    # if not, check for a profile view
    if len(faces) > 0:
        is_frontal_face = 1
    else:
        faces = profile_face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30),
            flags = 0
        )

    # return none if there are no faces
    if len(faces) == 0:
        return [0] * len(getFeatureName())
    elif not is_frontal_face:
        is_profile_face = 1

    # we only want one face per image, so take the first one always
    (x, y, w, h) = faces[0]

    # face size, as percentage of image
    face_size = float(w * h) / (img_h * img_w)

    # face location, as percentage
    face_loc_x = [0] * NUM_LOC_BINS
    face_loc_y = [0] * NUM_LOC_BINS
    face_loc_x[util.getBinIndex((float(x + float(w) / 2) / img_w),
                                NUM_LOC_BINS, MAX_LOC_VAL)] = 1
    face_loc_y[util.getBinIndex((float(y + float(h) / 2) / img_h),
                                NUM_LOC_BINS, MAX_LOC_VAL)] = 1

    # ready the features for returning
    features = [is_frontal_face, is_profile_face, face_size] + \
        face_loc_x + face_loc_y
    features += getEyeFeatures(x, y, w, h, img, gray)
    features += getSmileFeatures(x, y, w, h, img, gray)

    assert len(features) == len(getFeatureName()), \
        "length of face features matches feature names"

    '''Display and debug'''
    if IS_DEBUG:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.imshow('img', img)
        cv2.waitKey(0)
        return (features, img, gray)

    return features


def main():
    correct_face_list = []
    correct_eye_list = []
    for filename in os.listdir('test_data/'):
        path = 'test_data/' + filename
        if os.path.isdir(path):
            continue

        print filename
        cv_image = cv2.imread(path)
        blah = extractFeature(cv_image)
        if len(blah) == 3:
            (features, img, filtered) = blah
            print features
            cv2.imwrite('gray_output/' + filename, img)
            cv2.imwrite('gray_output/filter/' + filename, filtered)

    #     print 'correct face or no? [1/0]'
    #     correct_face_list.append(int(raw_input()))
    #     print 'correct eyes or no? [2/1/0]'
    #     correct_eye_list.append(int(raw_input()))

    # print "face accuracy: %f" % (float(sum(correct_face_list)) / len(correct_face_list))
    # print "eye accuracy: %f" % (float(sum(correct_eye_list)) / (2*len(correct_eye_list)))


if __name__ == "__main__":
    main()
