MIN_HUE = 0
MIN_SAT = 0
MIN_VAL = 0
MIN_LIGHT = 0
MAX_HUE = 360
MAX_SAT = 255
MAX_VAL = 255
MAX_LIGHT = 255


# returns the value normalize to 0-1
def normalize(value, minimum, maximum):
    return float(value - minimum) / (maximum - minimum)


def binFeatureNames(prefix, numBins, max_val):
    """
    Returns a list of names of the bucketed features.
    """
    multiple = int(float(max_val) / numBins)
    return [prefix + ('_%d' % (i*multiple)) for i in range(1, numBins+1)]


def getBinIndex(value, numBins, max_val):
    """
    Returns the bucket index of the bin the value should go in.
    """
    if value == max_val:
        return numBins - 1
    else:
        return int((value) / (float(max_val) / numBins))


def opencv_hue_diff(h1, h2):
    """
    Computes HSV color diff with wraparound.
    """
    h_max = max(h1, h2)
    h_min = min(h1, h2)
    d1 = h_max - h_min
    d2 = (360 - h_max) + h_min
    return min(d1, d2)


def compute_hue_diff(c1, c2):
    """
    Computes HSV color diff with wraparound.
    """
    c_max = max(c1[0], c2[0])
    c_min = min(c1[0], c2[0])
    d1 = c_max - c_min
    d2 = (1 - c_max) + c_min
    return min(d1, d2)

def compute_sat_diff(c1, c2):
    """
    Computes HSV saturation diff with wraparound.
    Don't ask why...
    """
    return abs(c1[1] - c2[1])

def compute_lightness_diff(c1, c2):
    """
    Computes HSV value/lightness diff.
    """
    return abs(c1[2] - c2[2])
