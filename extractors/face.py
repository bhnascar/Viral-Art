"""
Returns
    - Face exists
    - how the face is
    - where it is located
    - how big the face is w.r.t. the entire image

    - number of eyes found
    - where the eyes are located
    - how big the eyes are in comparison to the face

Only works for realistic faces right now.
TODO: look into training a Haar cascade for more cartoony faces,
or the other Haar Cascades
"""
import cv2
import imutils

IS_DEBUG = False


def getFeatureName():
    return ["frontal_face", "profile_face", "face_x", "face_y", "face_size",
            "eye_size", "number_of_visible_eyes",
            "eye_x_in_face", "eye_y_in_face", "eye_x_in_img", "eye_y_in_img"]

# def getSmileFeatures(fx, fy, fw, fh, img, gray_img):


def removeExtraneousEyes(eyes, fx, fy, fw, fh):    
    # remove any eyes not in the face
    to_remove = []
    for index, val in enumerate(eyes):
        (ex, ey, ew, eh) = val
        if ((ex < fx) or (ey < fy) or
           (ex + ew > fx + fw) or (ey + eh > fy + fh)):
            to_remove.append(index)

    for i in to_remove:
        eyes.pop(i)

    return eyes[:2]


def getEyeFeatures(fx, fy, fw, fh, img, gray):
    img_h, img_w = img.shape[:2]
    # find eyes
    open_eyes_cascade = cv2.CascadeClassifier('extractors/cascades/haarcascade_eye.xml')

    # for debugging
    if IS_DEBUG:
        open_eyes_cascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')

    eyes = open_eyes_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = 0
    )

    if not len(eyes) > 0:
        return [None] * 6

    eyes = removeExtraneousEyes(eyes, fx, fy, fw, fh)

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

    eye_size = float(avg_x * avg_h) / (fw * fh)

    eye_x_in_img = avg_x / img_w
    eye_y_in_img = avg_y / img_h
    eye_x_in_face = float(avg_x - fx) / fw
    eye_y_in_face = float(avg_y - fy) / fh

    if IS_DEBUG:
        '''Display for debugging'''
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(img, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    return [eye_size, number_of_visible_eyes,
            eye_x_in_face, eye_y_in_face, eye_x_in_img, eye_y_in_img]


def extractFeature(img):
    is_frontal_face = False
    is_profile_face = False

    img_h, img_w = img.shape[:2]
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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
        is_frontal_face = True
    else:
        faces = front_face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30),
            flags = 0
        )

    # return none if there are no faces
    if len(faces) == 0:
        return [None] * len(getFeatureName())
    elif not is_frontal_face:
        is_profile_face = True

    # we only want one face per image, so take the first one always
    (x, y, w, h) = faces[0]
    face_x = float(x + float(w) / 2) / img_w
    face_y = float(y + float(w) / 2) / img_h
    face_size = float(w * h) / (img_h * img_w)

    features = [is_frontal_face, is_profile_face, face_x, face_y, face_size]
    features += getEyeFeatures(x, y, w, h, img, gray)


    '''Display and debug'''
    # cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # roi_gray = gray[y:y+h, x:x+w]
    # roi_color = img[y:y+h, x:x+w]
    # cv2.imshow('img', img)
    # cv2.waitKey(0)

    return features


def main():
    cv_image = cv2.imread("test_data/wonder_woman.jpg")
    # cv_image = imutils.resize(cv_image, width=200)

    print extractFeature(cv_image)

if __name__ == "__main__":
    main()
