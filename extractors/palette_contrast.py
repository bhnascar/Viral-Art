"""
Returns the maximum value and hue contrast given the dominant color scheme
"""
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import cv2
import imutils
import util
import os

IS_DEBUG = False
NUM_BINS = 5
MAX_HUE_DIFF_VAL = 180
MAX_SAT_DIFF_VAL = 255
MAX_VAL_DIFF_VAL = 255
MAX_LIGHT_DIFF_VAL = 255


def getFeatureName():
    return ["Color_Contrast", "Saturation_contrast", "Value_Contrast", "Light_Contrast"] + \
        util.binFeatureNames("hue_palette_diff", NUM_BINS, MAX_HUE_DIFF_VAL) + \
        util.binFeatureNames("sat_palette_diff", NUM_BINS, MAX_SAT_DIFF_VAL) + \
        util.binFeatureNames("val_palette_diff", NUM_BINS, MAX_VAL_DIFF_VAL) + \
        util.binFeatureNames("light_palette_diff", NUM_BINS, MAX_LIGHT_DIFF_VAL)


def extractFeature(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = imutils.resize(img, width=200)

    # Get dominant color scheme via k-means
    # reshape the image to be a list of pixels
    img = img.reshape((img.shape[0] * img.shape[1], 3))

    # run kmeans
    clt = KMeans(n_clusters=8)
    clt.fit(img)

    # convert to hsv
    hsv = cv2.cvtColor(np.float32([clt.cluster_centers_]), cv2.COLOR_RGB2HSV)
    # get hls
    hls = cv2.cvtColor(np.float32([clt.cluster_centers_]), cv2.COLOR_RGB2BGR)
    hls = cv2.cvtColor(hls, cv2.COLOR_BGR2HLS)

    # Get max value, saturation, light contrast
    min_val = 255
    max_val = 0
    min_sat = 255
    max_sat = 0
    min_light = 255
    max_light = 0
    hue_set = set()
    for ind, val in enumerate(hsv[0]):
        h, s, v = val
        max_val = max(max_val, v)
        min_val = min(min_val, v)
        max_sat = max(min_sat, s)
        min_sat = min(max_sat, s)
        hue_set.add(int(round(h)))

    for val in hls[0]:
        h, l, s = val
        max_light = max(max_light, l)
        min_light = min(min_light, l)

    max_sat_diff = max_sat - min_sat
    max_val_diff = max_val - min_val
    max_light_diff = max_light - min_light

    hues = list(hue_set)
    max_hue_diff = 0
    num_hues = len(hues)
    for i in range(num_hues):
        for j in range(i+1, num_hues):
            max_hue_diff = max(max_hue_diff,
                               util.opencv_hue_diff(hues[i], hues[j]))

    # normalize the value
    max_hue_diff = util.normalize(max_hue_diff, 0, MAX_HUE_DIFF_VAL)
    max_sat_diff = util.normalize(max_sat_diff, 0, MAX_SAT_DIFF_VAL)
    max_val_diff = util.normalize(max_val_diff, 0, MAX_VAL_DIFF_VAL)
    max_light_diff = util.normalize(max_light_diff, 0, MAX_LIGHT_DIFF_VAL)

    if IS_DEBUG:
        # build a histogram of clusters and then create a figure
        # representing the number of pixels labeled to each color
        hist = centroid_histogram(clt)
        bar = plot_colors(hist, clt.cluster_centers_)

        # show our color bar
        plt.figure()
        plt.axis("off")
        plt.imshow(bar)
        plt.show()

    hue_bins = [0]*NUM_BINS
    sat_bins = [0]*NUM_BINS
    val_bins = [0]*NUM_BINS
    light_bins = [0]*NUM_BINS
    hue_bins[util.getBinIndex(max_hue_diff, NUM_BINS, MAX_HUE_DIFF_VAL)] = 1
    sat_bins[util.getBinIndex(max_sat_diff, NUM_BINS, MAX_SAT_DIFF_VAL)] = 1
    val_bins[util.getBinIndex(max_val_diff, NUM_BINS, MAX_VAL_DIFF_VAL)] = 1
    light_bins[util.getBinIndex(max_light_diff, NUM_BINS, MAX_LIGHT_DIFF_VAL)] = 1

    features = [max_hue_diff, max_sat_diff, max_val_diff, max_light_diff] + \
        hue_bins + sat_bins + val_bins + light_bins

    assert len(features) == len(getFeatureName()), \
        "length of palette contrast features matches feature names"

    return features


'''debugging functions'''
def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins = numLabels)

    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()

    # return the histogram
    return hist


def plot_colors(hist, centroids):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype = "uint8")
    startX = 0

    # loop over the percentage of each cluster and the color of
    # each cluster
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
            color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar


def main():
    for filename in os.listdir('test_data/'):
        path = 'test_data/' + filename
        if os.path.isdir(path):
            continue
        print filename
        cv_image = cv2.imread(path)
        print extractFeature(cv_image)

    # cv_image = cv2.imread("test_data/wonder_woman.jpg")

    # if IS_DEBUG:
    #     cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    #     plt.figure()
    #     plt.axis("off")
    #     plt.imshow(cv_image)
    #     cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

    # print extractFeature(cv_image)

if __name__ == "__main__":
    main()
