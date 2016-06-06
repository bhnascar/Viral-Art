#!/usr/bin/env python
"""
Predicts the author of a sketch (yes/no) by training on a given
author's gallery.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import grid_search, ensemble
from sklearn import svm
from sklearn import metrics
from sklearn import linear_model
from sklearn.feature_selection import RFE
from sklearn.cross_validation import cross_val_predict
from sklearn.preprocessing import PolynomialFeatures
from sklearn.naive_bayes import GaussianNB

DEFAULT_DATA_FILE = "output/features.txt"


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
        else:
            other_labels.append(0)
            other_urls.append(url)
            other_data = np.vstack([other_data, feature])

    (artist_rows, _) = artist_data.shape
    (other_rows, _) = other_data.shape

    # Take half of the artist's works and half of others' to train
    train_urls = artist_urls[:artist_rows/2] + other_urls[:other_rows/2]
    train_labels = artist_labels[:artist_rows/2] + other_labels[:other_rows/2]
    train_features = np.vstack([artist_data[:artist_rows/2,:], other_data[:other_rows/2,:]])

    # Take the remaining halves to test
    test_urls = artist_urls[artist_rows/2:] + other_urls[other_rows/2:]
    test_labels = artist_labels[artist_rows/2:] + other_labels[other_rows/2:]
    test_features = np.vstack([artist_data[artist_rows/2:,:], other_data[other_rows/2:,:]])

    return (train_urls, train_labels, train_features), (test_urls, test_labels, test_features)


def forest(train_features, train_labels, test_features, test_labels):
    model = ensemble.RandomForestClassifier()
    model.fit(train_features, train_labels)
    test_results = model.predict(test_features)
    train_results = model.predict(train_features)

    temp = []
    for index, val in enumerate(model.feature_importances_):
        if val > 0.001:
            temp.append((index, val))
    print sorted(temp, key=lambda x: x[1])

    return (test_results, train_results)


def naive_bayes(train_features, train_labels, test_features, test_labels):
    # Train SVM classifier
    model = GaussianNB()
    model.fit(train_features, train_labels)
    test_results = model.predict(test_features)
    train_results = model.predict(train_features)

    return (test_results, train_results)


def logistic(train_features, train_labels, test_features, test_labels):
    # Train SVM classifier
    model = linear_model.LogisticRegressionCV(penalty='l1', solver='liblinear')
    model.fit(train_features, train_labels)
    test_results = model.predict(test_features)
    train_results = model.predict(train_features)

    print "coefficients"
    temp = []
    for index, val in enumerate(model.coef_[0]):
        if val > 0.001:
            temp.append((index, val))
    print sorted(temp, key=lambda x: x[1])

    # print model.coef_

    return (test_results, train_results)


def svc(train_features, train_labels, test_features, test_labels):
    """
    Trains a support vector mahcine on the given training data
    and then runs it against the test data. Returns the result.
    """
    # Train SVM classifier
    rbf_svc = svm.SVC(class_weight='balanced')
    search_params = {'kernel':['poly'], 'C':[1, 10, 100, 1000], 'gamma': [0.000001, 0.0001, 0.01, 1, 10, 100]}
    svm_cv = grid_search.GridSearchCV(rbf_svc, param_grid=search_params, cv=5, n_jobs=1, verbose=5)
    svm_cv.fit(train_features, train_labels)
    # print(svm_cv.best_params_)
    test_results = svm_cv.predict(test_features)
    train_results = svm_cv.predict(train_features)

    return (test_results, train_results)


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

    dataframe = dataframe.drop("artist", axis=1)
    dataframe = dataframe.iloc[:, 5:]
    features = dataframe.values

    # interaction terms
    poly = PolynomialFeatures(interaction_only=True)
    features = poly.fit_transform(features)
    print "interaction"

    return features, artists, urls


def main(args):
    if len(args) == 2 and args[1] == "help":
        print """
              Usage: ./classifier.py [deviantART username] [features file]\n
              Currently you can try 'yuumei' and 'Namecchan' since these
              users have ~30 pictures each in our database of a 1000.
              """
        return
    elif len(args) == 2:
        features_file = DEFAULT_DATA_FILE
    elif len(args) < 3:
        print "Insufficient or incorrect arguments. Try 'help' for more information";
        return

    features, labels, urls = load_data(features_file)
    (train_urls, train_labels, train_features), (test_urls, test_labels, test_features) = partition_data_for_artist(args[1], features, labels, urls)

    # test_results, train_results = logistic(train_features, train_labels, test_features, test_labels)
    test_results, train_results = forest(train_features, train_labels, test_features, test_labels)
    # test_results, train_results = naive_bayes(train_features, train_labels, test_features, test_labels)
    # test_results, train_results = svc(train_features, train_labels, test_features, test_labels)

    tp = []
    fp = []
    fn = []
    tn = []
    for i in range(len(test_results)):
        if test_results[i] == 1:
            if test_labels[i] == 1:
                tp.append(test_urls[i])
            else:
                fp.append(test_urls[i])
        else:
            if test_labels[i] == 0:
                tn.append(test_urls[i])
            else:
                fn.append(test_urls[i])

    print "TP"
    print tp
    print "FP"
    print fp
    print "FN"
    print fn
    # print "TN"
    # print tn


    print "TESTING RESULTS"
    print "Scores", metrics.classification_report(test_labels, test_results)
    print "CONFUSION"
    print metrics.confusion_matrix(test_labels, test_results)

    print "============"
    print "TRAINING RESULTS"
    print "Scores", metrics.classification_report(train_labels, train_results)
    print "CONFUSION"
    print metrics.confusion_matrix(train_labels, train_results)

if __name__ == "__main__":
    main(sys.argv)
