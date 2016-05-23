def binLocFeatureNames(prefix, numBins):
    '''returns a list of names of the bucketed features'''
    multiple = int(100 / numBins)
    return [prefix + ('_%d' % i*multiple) for i in range(1, numBins+1)]


def getLocBinIndex(value, numBins):
    '''
        Returns the bucket index of the bin the value should go in.
        Intended for use for the location features.
        Values should be fractions (0 - 1)
    '''
    return int((100*value) / (100 / numBins))
