def binFeatureNames(prefix, numBins, max_val):
    '''returns a list of names of the bucketed features'''
    multiple = int(max_val / numBins)
    return [prefix + ('_%d' % (i*multiple)) for i in range(1, numBins+1)]


def getBinIndex(value, numBins, max_val):
    '''
        Returns the bucket index of the bin the value should go in.
        Intended for use for the location features.
    '''
    return int((value) / (float(max_val) / numBins))


def compute_hue_diff(c1, c2):
    """
    Computes HSV color diff with wraparound.
    """
    c_max = max(c1[0], c2[0])
    c_min = min(c1[0], c2[0])
    d1 = c_max - c_min
    d2 = (1 - c_max) + c_min
    return min(d1, d2)
