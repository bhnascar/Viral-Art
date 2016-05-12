"""
Returns the maximum value contrast given the dominant color scheme
"""
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import cv2
import imutils

def getFeatureName():
    return ["Value_Contrast", "Color_Contrast"]


def extractFeature(img):
    # Get dominant color scheme via k-means
    # reshape the image to be a list of pixels
    img = img.reshape((img.shape[0] * img.shape[1], 3))

    # run kmeans
    clt = KMeans(n_clusters=5)
    clt.fit(img)

    # convert to hsv
    hsv = cv2.cvtColor(np.float32([clt.cluster_centers_]), cv2.COLOR_BGR2HSV)

    # initialize values
    # value runs from 0-255 in opencv
    # hue runs from 0-360 in opencv
    min_val = 255
    max_val = 0
    min_hue = 360
    max_hue = 0
    for ind, val in enumerate(hsv[0]):
        h, s, v = val
        max_val = max(max_val, v)
        min_val = min(min_val, v)
        max_hue = max(max_hue, h)
        min_hue = min(min_hue, h)

    # # build a histogram of clusters and then create a figure
    # # representing the number of pixels labeled to each color
    # hist = centroid_histogram(clt)
    # bar = plot_colors(hist, clt.cluster_centers_)

    # # show our color bar
    # plt.figure()
    # plt.axis("off")
    # plt.imshow(bar)
    # plt.show()

    # value contrast, color contrast
    return [max_val - min_val, max_hue - min_hue]

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
    cv_image = cv2.imread("test_data/wlop.jpg")
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    cv_image = imutils.resize(cv_image, width=200)

    # plt.figure()
    # plt.axis("off")
    # plt.imshow(cv_image)

    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
    print extractFeature(cv_image)

if __name__ == "__main__":
    main()