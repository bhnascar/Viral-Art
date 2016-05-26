def binFeatureNames(prefix, numBins, max_val):
    '''returns a list of names of the bucketed features'''
    multiple = int(max_val / numBins)
    return [prefix + ('_%d' % (i*multiple)) for i in range(1, numBins+1)]


def getBinIndex(value, numBins, max_val):
    '''
        Returns the bucket index of the bin the value should go in.
        Intended for use for the location features.
    '''
    return int((value) / (max_val / numBins))
