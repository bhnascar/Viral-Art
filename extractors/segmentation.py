import numpy as np
import cv2
from skimage import data, io, segmentation, color
from skimage.future import graph
from matplotlib import pyplot as plt
import imutils
import util
import os

IS_DEBUG = False
NUM_LOC_BINS = 5
MAX_LOC_VAL = 1


def getFeatureName():
    return util.binFeatureNames("max_sat_x", NUM_LOC_BINS, MAX_LOC_VAL*100) + \
        util.binFeatureNames("max_sat_y", NUM_LOC_BINS, MAX_LOC_VAL*100) + \
        util.binFeatureNames("max_light_x", NUM_LOC_BINS, MAX_LOC_VAL*100) + \
        util.binFeatureNames("max_light_y", NUM_LOC_BINS, MAX_LOC_VAL*100) + \
        util.binFeatureNames("max_val_x", NUM_LOC_BINS, MAX_LOC_VAL*100) + \
        util.binFeatureNames("max_val_y", NUM_LOC_BINS, MAX_LOC_VAL*100) + \
        ["highlight_sat_size", "highlight_light_size", "highlight_val_size"] + \
        ["seg_sat_contrast", "seg_light_contrast", "seg_val_contrast"]


class Cluster(object):
    x = 0
    y = 0
    num_pixels = 0
    color = (0, 0, 0)

    def __init__(self, color):
        self.color = color

    def add_pixel(self, x, y):
        self.x += x
        self.y += y
        self.num_pixels += 1

    def finalize(self):
        self.x = round(float(self.x) / self.num_pixels)
        self.y = round(float(self.y) / self.num_pixels)


def plot_marker(loc_x, loc_y, color, img):
    h, w, color_depth = img.shape
    for y in range(-5, 5):
        for x in range(-5, 5):
            py = loc_y + y
            px = loc_x + x

            if 0 <= py < h and 0 <= px < w:
                img[py, px, :] = color


def extractFeature(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = imutils.resize(img, width=200)

    # segment the image
    # labels = segmentation.felzenszwalb(img, scale=40.0)
    labels = segmentation.quickshift(img, ratio=1.0, kernel_size=4, max_dist=8)
    # labels1 = segmentation.slic(img, compactness=15, n_segments=300)

    # get the segmented image
    seg_img = color.label2rgb(labels, img, kind='avg')

    # Convert to hls space
    seg_bgr_img = cv2.cvtColor(seg_img, cv2.COLOR_RGB2BGR)
    hls = cv2.cvtColor(seg_bgr_img, cv2.COLOR_BGR2HLS)
    hsv = cv2.cvtColor(seg_bgr_img, cv2.COLOR_BGR2HSV)

    # get cluster information
    cluster_dict = {}
    max_light = 0
    max_sat = 0
    max_val = 0
    min_light = 1000
    min_sat = 1000
    min_val = 1000
    max_light_label = 0
    max_sat_label = 0
    max_val_label = 0
    height, width, color_depth = img.shape
    for y in range(0, height - 1):
        for x in range(0, width - 1):
            hls_pix = hls[y, x, :]
            hsv_pix = hsv[y, x, :]
            label = labels[y, x]

            if label not in cluster_dict:
                cluster_dict[label] = Cluster(hls_pix)
            cluster_dict[label].add_pixel(x, y)

            min_sat = min(min_sat, hls_pix[2])
            if hls_pix[2] > max_sat:
                max_sat = hls_pix[2]
                max_sat_label = label

            min_light = min(min_light, hls_pix[1])
            if hls_pix[1] > max_light:
                max_light = hls_pix[1]
                max_light_label = label

            min_val = min(min_val, hsv_pix[1])
            if hsv_pix[1] > max_val:
                max_val = hsv_pix[1]
                max_val_label = label

    # calculate cluster locations
    for c in cluster_dict:
        cluster_dict[c].finalize()

    # calculate the actual features
    # get location features
    max_sat_locx_bin = NUM_LOC_BINS*[0]
    max_sat_locy_bin = NUM_LOC_BINS*[0]
    max_light_locx_bin = NUM_LOC_BINS*[0]
    max_light_locy_bin = NUM_LOC_BINS*[0]
    max_val_locx_bin = NUM_LOC_BINS*[0]
    max_val_locy_bin = NUM_LOC_BINS*[0]

    max_sat_locx_bin[util.getBinIndex(float(cluster_dict[max_sat_label].x) / width,
                                      NUM_LOC_BINS, MAX_LOC_VAL)] = 1
    max_sat_locy_bin[util.getBinIndex(float(cluster_dict[max_sat_label].y) /  height,
                                      NUM_LOC_BINS, MAX_LOC_VAL)] = 1
    max_light_locx_bin[util.getBinIndex(float(cluster_dict[max_light_label].x) / width,
                                        NUM_LOC_BINS, MAX_LOC_VAL)] = 1
    max_light_locy_bin[util.getBinIndex(float(cluster_dict[max_light_label].y) / height,
                                        NUM_LOC_BINS, MAX_LOC_VAL)] = 1
    max_val_locx_bin[util.getBinIndex(float(cluster_dict[max_val_label].x) / width,
                                      NUM_LOC_BINS, MAX_LOC_VAL)] = 1
    max_val_locy_bin[util.getBinIndex(float(cluster_dict[max_val_label].y) / height,
                                      NUM_LOC_BINS, MAX_LOC_VAL)] = 1

    # get size features
    total_pixels = height*width
    sat_size = float(cluster_dict[max_sat_label].num_pixels) / total_pixels
    light_size = float(cluster_dict[max_light_label].num_pixels) / total_pixels
    val_size = float(cluster_dict[max_val_label].num_pixels) / total_pixels

    # get contrast features
    mini = 20   # from manual inspection
    max_sat_diff = util.normalize(max_sat - min_sat, mini, util.MAX_SAT)
    max_light_diff = util.normalize(max_light - min_light, mini, util.MAX_LIGHT)
    max_val_diff = util.normalize(max_val - min_val, mini, util.MAX_VAL)

    features = max_sat_locx_bin + max_sat_locy_bin + \
        max_light_locx_bin + max_light_locy_bin + \
        max_val_locx_bin + max_val_locy_bin + \
        [sat_size, light_size, val_size] + \
        [max_sat_diff, max_light_diff, max_val_diff]

    assert len(features) == len(getFeatureName()), \
        "length of segmentation features matches feature names"

    # plot for debugging
    if IS_DEBUG:
        plot_marker(cluster_dict[max_sat_label].x,
                    cluster_dict[max_sat_label].y,
                    (255, 0, 0), seg_bgr_img)

        plot_marker(cluster_dict[max_light_label].x,
                    cluster_dict[max_light_label].y,
                    (0, 255, 0), seg_bgr_img)

        plot_marker(cluster_dict[max_val_label].x,
                    cluster_dict[max_val_label].y,
                    (0, 0, 255), seg_bgr_img)

        plt.figure()
        cv2.imshow('img', seg_bgr_img)
        cv2.waitKey(0)

        return [features, seg_bgr_img]

    return features


def main():
    for filename in os.listdir('test_data/'):
        path = 'test_data/' + filename
        if os.path.isdir(path):
            continue

        print filename
        cv_image = cv2.imread(path)
        if IS_DEBUG:
            (features, img) = extractFeature(cv_image)
            print features
            cv2.imwrite('output/highlights/' + filename, img)
        else:
            print extractFeature(cv_image)


if __name__ == "__main__":
    main()