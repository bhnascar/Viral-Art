#!/usr/bin/env python
"""
Reads in features, and prints out the trained weight vector.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn import linear_model
from sklearn.cross_validation import cross_val_predict

DEFAULT_DATA_FILE = "output/features.txt"

def compute_score(labels, results):
    """
    Computes a score for the regression given the actual
    labels and the predicted labels.
    """
    return np.sum(np.abs(np.divide(results - np.array(labels), np.array(labels))))

def svr(train_features, train_labels, test_features, test_labels):
    """
    Trains a support vector regressor on the given training data
    and then runs it against the test data. Returns the result.
    """
    rbf_svr = svm.SVR(kernel = "rbf", C = 80e3, gamma = "auto")
    rbf_svr.fit(train_features, train_labels)
    test_results = rbf_svr.predict(test_features)
    return test_results

def linear_regressor(train_features, train_labels, test_features, test_labels):
    """
    Trains a linear regressor on the given training data and then
    runs it against the test data. Returns the result.
    """
    lr = linear_model.LinearRegression()
    lr.fit(train_features, train_labels)
    test_results = cross_val_predict(lr, test_features, test_labels, cv=5)
    test_results[test_results < 0] = 0
    return test_results

def partition_data(features, labels):
    """
    Partitions the input data into a training and testing dataset.
    Returns a tuple of tuples:
    ((train_labels, train_features), 
     (test_labels, test_features))
    """
    # Partition into training and testing datasets (approx. half and half right now)
    train_labels = labels[:500]
    train_features = features[:500,:]

    test_labels = labels[500:]
    test_features = features[500:,:]

    return (train_features, train_labels), (test_features, test_labels)

def load_data(datafile = DEFAULT_DATA_FILE):
    """
    Loads feature and classification data from the given file.
    Returns a tuple of (features, labels) where both are
    features is an NP array and labels is a list.
    """
    # Read data
    dataframe = pd.read_csv(datafile)
    views = dataframe["views"].tolist()

    # Uncomment this and fix the return line if you want to do 
    # favorites prediction instead
    # favs = dataframe["favorites"].tolist()

    # 5 is the first column after 'favs'.
    # -1 means last column from the end, because I added an extra
    # column for the artist name, which we want to ignore.
    features = dataframe.iloc[:, 5:-1]
    features = np.array(features)
    return features, views

def main(args):
    if len(args) == 2 and args[1] == "help":
        print """
              Usage: ./learner.py [features file]
              """
        return
    elif len(args) < 1:
        print "Insufficient or incorrect arguments. Try 'help' for more information";
        return

    # Load data.
    if len(args) > 1:
        features, labels = load_data(args[1])
    else:
        features, labels = load_data() # using default features file

    # Partition into training and test datasets
    (train_features, train_labels), (test_features, test_labels) = partition_data(features, labels)

    # Linear regression
    test_results = linear_regressor(train_features, train_labels, test_features, test_labels)
    test_score = compute_score(test_labels, test_results)
    print "Test score: {}".format(test_score)

    train_results = linear_regressor(train_features, train_labels, train_features, train_labels)
    train_score = compute_score(train_labels, train_results)
    print "Train score: {}".format(train_score)
    
    # Plot output
    fig, ax = plt.subplots()
    ax.scatter(train_labels, train_results, color="r")
    ax.scatter(test_labels, test_results, color="b")

    # Plot reference line
    ax.plot([min(test_labels), max(test_labels)], [min(test_labels), max(test_labels)], 'k--', lw=4)

    # Label axes
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')

    plt.show()

if __name__ == "__main__":
    main(sys.argv)