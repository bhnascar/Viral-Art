#!/usr/bin/env python
"""
Predicts the author of a sketch (yes/no) by training on a given
author's gallery.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn import linear_model
from sklearn.cross_validation import cross_val_score

def partition_data_for_artist(artist, features, labels):
    """
    Partitions the input data into a training and testing dataset.
    Returns a tuple of tuples:
    ((train_labels, train_features), (test_labels, test_features))
    """
    (_, cols) = features.shape
    artist_labels = []
    other_labels = []
    artist_data = np.empty([0, cols])
    other_data = np.empty([0, cols])

    for (username, feature) in zip(labels, features):
        if (artist == username):
            artist_labels.append(1)
            artist_data = np.vstack([artist_data, feature])
        else:
            other_labels.append(0)
            other_data = np.vstack([other_data, feature])

    (artist_rows, _) = artist_data.shape
    (other_rows, _) = other_data.shape;

    # Take half of the artist's works and half of others' to train
    train_labels = artist_labels[:artist_rows/2] + other_labels[:other_rows/2]
    train_features = np.vstack([artist_data[:artist_rows/2,:], other_data[:other_rows/2,:]])

    # Take the remaining halves to test
    test_labels = artist_labels[artist_rows/2:] + other_labels[other_rows/2:]
    test_features = np.vstack([artist_data[artist_rows/2:,:], other_data[other_rows/2:,:]])

    return (train_labels, train_features), (test_labels, test_features)

def load_data(datafile):
    """
    Loads feature and classification data from the given file.
    Returns a tuple of (features, labels) where both are
    features is an NP array and labels is a list.
    """
    # Read data
    dataframe = pd.read_csv('output/features.txt')
    artists = dataframe["artist"].tolist()
    features = dataframe.iloc[:, 5:-1]
    features = np.array(features)
    return features, artists

def main(args):
    if len(args) == 2 and args[1] == "help":
        print """
              Usage: ./predictor.py [deviantART username] [features file]
              """
        return
    elif len(args) < 3:
        print "Insufficient or incorrect arguments. Try 'help' for more information";
        return

    features, labels = load_data(args[2])
    (tr_la, tr_ft), (te_la, te_ft) = partition_data_for_artist(args[1], features, labels)

    # Run SVM classifier
    clf = svm.SVC()
    clf.fit(tr_ft, tr_la);
    print cross_val_score(clf, te_ft, te_la, scoring='accuracy')

if __name__ == "__main__":
    main(sys.argv)