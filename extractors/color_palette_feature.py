"""
Returns the hue, saturation, and value palettes
"""
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import cv2
import imutils
import util

IS_DEBUG = False
NUM_BINS = 6
MAX_HUE_VAL = 360  # 0-360
MAX_SAT_VAL = 255  # 0-255
MAX_VAL_VAL = 255  # 0-255


def getFeatureName():
    return util.binFeatureNames("HUE_PALETTE", NUM_BINS, MAX_HUE_VAL) + \
        util.binFeatureNames("SAT_PALETTE", NUM_BINS, MAX_SAT_VAL) + \
        util.binFeatureNames("VAL_PALETTE", NUM_BINS, MAX_VAL_VAL) + \
        ["num_unique_hues", "num_unique_sat", "num_unique_vals"]


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
    hsv = cv2.cvtColor(np.float32([clt.cluster_centers_]),
                           cv2.COLOR_RGB2HSV)

    # create the palette histograms
    hue_palette = [0]*NUM_BINS
    sat_palette = [0]*NUM_BINS
    val_palette = [0]*NUM_BINS

    for c in hsv[0]:
        h, s, v = c
        ''' BINARY OR HISTOGRAM? '''
        hue_palette[util.getBinIndex(h, NUM_BINS, MAX_HUE_VAL)] = 1
        sat_palette[util.getBinIndex(s, NUM_BINS, MAX_SAT_VAL)] = 1
        val_palette[util.getBinIndex(v, NUM_BINS, MAX_VAL_VAL)] = 1

    num_unique_hues = sum(hue_palette)
    num_unique_sat = sum(sat_palette)
    num_unique_vals = sum(val_palette)

    # return the features
    features = hue_palette + sat_palette + val_palette + \
        [num_unique_hues, num_unique_sat, num_unique_vals]

    assert len(features) == len(getFeatureName()), \
        "length of color palette features matches feature names"

    if IS_DEBUG:
        # build a histogram of clusters and then create a figuo each color
        hist = centroid_histogram(clt)
        # representing the number of pixels labeled t
        bar = plot_colors(hist, clt.cluster_centers_)

        # show our color bar
        plt.figure()
        plt.axis("off")
        plt.imshow(bar)
        plt.show()

    # value contrast, color contrast
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
    cv_image = cv2.imread("test_data/yuumei_profile.jpg")

    if IS_DEBUG:
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        plt.figure()
        plt.axis("off")
        plt.imshow(cv_image)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

    print extractFeature(cv_image)

if __name__ == "__main__":
    main()
