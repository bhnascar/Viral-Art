import numpy as np

from skimage import data, io, segmentation, color
from skimage.future import graph
from matplotlib import pyplot as plt

from util import *

def compute_diff(c1, c2):
    """
    Computes some diff between c1 and c2 for the
    purpose of comparison.
    """
    return compute_sat_diff(c1, c2)# + compute_lightness_diff(c1, c2)

def main():
    #img = data.coffee()
    img = io.imread("test_data/wonder.jpg")

    labels1 = segmentation.slic(img, compactness=30, n_segments=100)
    out1 = color.label2rgb(labels1, img, kind='avg')

    # Convert to HSV space
    out2 = color.rgb2hsv(out1)

    max_color_diff = 0
    max_pos = (-1, -1)

    height, width, color_depth = out1.shape
    for y in range(0, height - 1):
        for x in range(0, width - 1):
            self = out2[y, x, :]
            bottom_neighbor = out2[y + 1, x, :]
            right_neighbor = out2[y, x + 1, :]

            color_diff = compute_diff(self, bottom_neighbor)
            if (color_diff > max_color_diff):
                max_color_diff = color_diff
                max_pos = (x, y)

            color_diff = compute_diff(self, right_neighbor)
            if (color_diff > max_color_diff):
                max_color_diff = color_diff
                max_pos = (x, y)

    print "Max color diff ({}) at pos {}".format(max_color_diff, max_pos)

    # Highlight max color diff position
    for y in range(-10, 10):
        for x in range(-10, 10):
            py = max_pos[1] + y
            px = max_pos[0] + x
            if 0 <= py <= height and 0 <= px <= width:
                #print "Marked pix"
                out2[py, px, 0] = 1
                out2[py, px, 1] = 0
                out2[py, px, 2] = 0
                out1[py, px, 0] = 255
                out1[py, px, 1] = 0
                out1[py, px, 2] = 0

    #g = graph.rag_mean_color(img, labels1, mode='similarity')
    #labels2 = graph.cut_normalized(labels1, g)
    #out2 = color.label2rgb(labels2, img, kind='avg')

    plt.figure()
    io.imshow(out1)
    plt.figure()
    io.imshow(out2)
    io.show()

if __name__ == "__main__":
    main()