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
from sklearn import metrics
from sklearn import grid_search
from sklearn import linear_model
from sklearn.feature_selection import RFE
from sklearn.cross_validation import cross_val_predict

def partition_data_for_artist(artist, features, labels, urls):
    """
    Partitions the input data into a training and testing dataset.
    Returns a tuple of tuples:
    ((train_urls, train_labels, train_features), 
     (test_urls, test_labels, test_features))
    """
    (_, cols) = features.shape
    artist_urls = []
    other_urls = []
    artist_labels = []
    other_labels = []
    artist_data = np.empty([0, cols])
    other_data = np.empty([0, cols])

    for ((url, username), feature) in zip(zip(urls, labels), features):
        if (artist == username):
            artist_labels.append(1)
            artist_urls.append(url)
            artist_data = np.vstack([artist_data, feature])
        elif (username == "Namecchan"):
            other_labels.append(0)
            other_urls.append(url)
            other_data = np.vstack([other_data, feature])

    (artist_rows, _) = artist_data.shape
    (other_rows, _) = other_data.shape;

    train_urls = artist_urls + other_urls
    train_labels = artist_labels + other_labels
    train_features = np.vstack([artist_data, other_data])

    test_urls = train_urls
    test_labels = train_labels
    test_features = train_features

    # # Take half of the artist's works and half of others' to train
    # train_urls = artist_urls[:artist_rows/2] + other_urls[:other_rows/2]
    # train_labels = artist_labels[:artist_rows/2] + other_labels[:other_rows/2]
    # train_features = np.vstack([artist_data[:artist_rows/2,:], other_data[:other_rows/2,:]])

    # # Take the remaining halves to test
    # test_urls = artist_urls[artist_rows/2:] + other_urls[other_rows/2:]
    # test_labels = artist_labels[artist_rows/2:] + other_labels[other_rows/2:]
    # test_features = np.vstack([artist_data[artist_rows/2:,:], other_data[other_rows/2:,:]])

    return (train_urls, train_labels, train_features), (test_urls, test_labels, test_features)

def load_data(datafile):
    """
    Loads feature and classification data from the given file.
    Returns a tuple of (features, labels) where both are
    features is an NP array and labels is a list.
    """
    # Read data
    dataframe = pd.read_csv(datafile)
    urls = dataframe["base_url"].tolist()
    artists = dataframe["artist"].tolist()

    selected_features = range(7, 61) + range(62, 138)
    # selected_features = [36, 39, 40, 41, 42, 43]
    # selected_features = [7, 8, 9, 10, 11, 12, 31, 32, 33, 34, 35, 78, 79, 80, 81, 82, 83, 84]

    features = dataframe.iloc[:, selected_features]
    features = np.array(features)
    return features, artists, urls

def main(args):
    if len(args) == 2 and args[1] == "help":
        print """
              Usage: ./predictor.py [deviantART username] [features file]\n
              Currently you can try 'yuumei' and 'Namecchan' since these
              users have ~30 pictures each in our database of a 1000.
              """
        return
    elif len(args) < 3:
        print "Insufficient or incorrect arguments. Try 'help' for more information";
        return

    features, labels, urls = load_data(args[2])
    (tr_urls, tr_la, tr_ft), (te_urls, te_la, te_ft) = partition_data_for_artist(args[1], features, labels, urls)

    # Perform grid search for best parameters
    # raw_svm = svm.SVC(decision_function_shape='ovr', class_weight='balanced', probability=True) 
    # search_params = {'kernel':['rbf'], 'C':[1, 10, 100, 1000, 10000], 'gamma': [0.00001, 0.0001, 0.01, 1, 10, 100]} 
    # svm_cv = grid_search.GridSearchCV(raw_svm, param_grid=search_params, scoring='recall', cv=5, n_jobs=-1, verbose=5) 
    # svm_cv.fit(tr_ft, tr_la) 
    # print(svm_cv.best_params_)

    # Train SVM classifier
    clf = svm.SVC(kernel = "linear", C = .001, gamma = 1e3)
    #clf = linear_model.LinearRegression()
    clf.fit(tr_ft, tr_la);
    print tr_ft

    # Run classifier
    predictions = cross_val_predict(clf, tr_ft, tr_la, cv=5)
    print predictions

    # Use RFE model to estimate top 5 most useful attributes
    # rfe = RFE(clf, 5)
    # rfe = rfe.fit(tr_ft, tr_la)
    # print(rfe.support_)
    # print(zip(rfe.ranking_, range(1, len(rfe.ranking_))))

    # Compare prediction and real label
    for i in range(0, len(predictions)):
        pred = predictions[i]
        real = tr_la[i];

        # Report any mistakes
        if pred == 1 and real == 0:
            print "False positive: {}".format(tr_urls[i])
        elif real == 1 and pred == 0:
            print "False negative: {}".format(tr_urls[i])

    # Print overall accuracy
    print "Overall accuracy: {}".format(metrics.accuracy_score(tr_la, predictions))
    print """
          Note, this is a misleading accuracy report because the SVC just keeps
          predicting 0 (not a match), which means there's only like ~15/1000 mistakes.
          """

if __name__ == "__main__":
    main(sys.argv)