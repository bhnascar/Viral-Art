#!/usr/bin/env python
"""
Reads in features, and prints out the trained weight vector.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import pydot

from sklearn.externals.six import StringIO
from sklearn import metrics
from sklearn import tree
from sklearn import svm
from sklearn import ensemble
from sklearn import grid_search
from sklearn import linear_model, cross_validation
from sklearn.preprocessing import PolynomialFeatures

DEFAULT_DATA_FILE = "output/features.txt"


def cap_results(train_results):
    retval = []
    for i in range(len(train_results)):
        temp = min(1.0, train_results[i])
        temp = max(0.0, temp)
        retval.append(temp)
    return retval


def boost(train_features, train_labels, test_features, test_labels):
    regressor = ensemble.GradientBoostingRegressor()
    regressor.fit(train_features, train_labels)

    test_results = regressor.predict(test_features)
    train_results = regressor.predict(train_features)

    print "test result", metrics.mean_squared_error(test_labels, test_results)
    print "test r2", metrics.r2_score(test_labels, test_results)
    print "train result", metrics.mean_squared_error(train_labels, train_results)
    print "train r2", metrics.r2_score(train_labels, train_results)

    return (test_results, train_results)


def decision_tree(train_features, train_labels, test_features, test_labels, feature_names):
    regressor = tree.DecisionTreeRegressor()
    regressor.fit(train_features, train_labels)

    test_results = cap_results(regressor.predict(test_features))
    train_results = cap_results(regressor.predict(train_features))

    print "test result", metrics.mean_squared_error(test_labels, test_results)
    print "test r2", metrics.r2_score(test_labels, test_results)
    print "train result", metrics.mean_squared_error(train_labels, train_results)
    print "train r2", metrics.r2_score(train_labels, train_results)

    # print "importances"
    # temp = []
    # for index, val in enumerate(regressor.feature_importances_):
    #     if val > 0.001:
    #         temp.append((index, val))
    # print sorted(temp, key=lambda x: x[1])

    '''graph stuff'''
    dot_data = StringIO()
    tree.export_graphviz(regressor, out_file=dot_data,
                        special_characters=True,
                        class_names=regressor.classes_,
                        impurity=False,
                        feature_names=feature_names)

    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("tree.pdf") 

    return (test_results, train_results)


def forest(train_features, train_labels, test_features, test_labels):
    regressor = ensemble.RandomForestRegressor(max_depth=5)
    regressor.fit(train_features, train_labels)

    test_results = cap_results(regressor.predict(test_features))
    train_results = cap_results(regressor.predict(train_features))

    print "test result", metrics.mean_squared_error(test_labels, test_results)
    print "test r2", metrics.r2_score(test_labels, test_results)
    print "train result", metrics.mean_squared_error(train_labels, train_results)
    print "train r2", metrics.r2_score(train_labels, train_results)

    print "importnaces"

    temp = []
    for index, val in enumerate(regressor.feature_importances_):
        if val > 0.001:
            temp.append((index, val))
    print sorted(temp, key=lambda x: x[1])

    return (test_results, train_results)


def svr(train_features, train_labels, test_features, test_labels):
    """
    Trains a support vector regressor on the given training data
    and then runs it against the test data. Returns the result.
    """
    rbf_svr = svm.SVR()
    search_params = {'kernel':['poly'], 'C':[1, 10, 100, 1000], 'gamma': [0.000001, 0.0001, 0.01, 1, 10, 100]}
    svm_cv = grid_search.GridSearchCV(rbf_svr, param_grid=search_params, cv=5, n_jobs=1, verbose=5, scoring='mean_absolute_error')
    svm_cv.fit(train_features, train_labels)
    print(svm_cv.best_params_)

    test_results = svm_cv.predict(test_features)
    train_results = svm_cv.predict(train_features)
    return (test_results, train_results)


def linear_regressor(train_features, train_labels, test_features, test_labels):
    """
    Trains a linear regressor on the given training data and then
    runs it against the test data. Returns the result.
    """
    # lr = linear_model.RANSACRegressor(min_samples=2)
    lr = linear_model.RidgeCV()
    # lr = linear_model.LassoCV(verbose=True, n_jobs=-1)
    lr.fit(train_features, train_labels)
    test_results = lr.predict(test_features)
    train_results = lr.predict(train_features)
    return test_results, train_results


def load_data(datafile = DEFAULT_DATA_FILE):
    """
    Loads feature and classification data from the given file.
    Returns a tuple of (features, labels) where both are
    features is an NP array and labels is a list.
    """
    # Read data
    dataframe = pd.read_csv(datafile)
    views = [ float(x) for x in dataframe["views"].tolist()]
    favs = [ float(x) for x in dataframe["favorites"].tolist()]
    ratio = []
    for i in range(len(views)):
        ratio.append(favs[i] / views[i])
    ratio = cap_results(ratio)

    # Uncomment this and fix the return line if you want to do 
    # favorites prediction instead
    # favs = dataframe["favorites"].tolist()

    # 5 is the first column after 'favs'.
    # -1 means last column from the end, because I added an extra
    # column for the artist name, which we want to ignore.
    dataframe = dataframe.drop("artist", axis=1)
    dataframe = dataframe.iloc[:, 5:]
    features = dataframe.values

    # # interaction terms
    poly = PolynomialFeatures(interaction_only=True)
    features = poly.fit_transform(features)
    print "interaction"

    return features, ratio, dataframe.columns.values


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
        features, labels, feature_names = load_data(args[1])
    else:
        features, labels, feature_names = load_data()  # using default features file

    # Partition into training and test datasets
    train_features, test_features, train_labels, test_labels = cross_validation.train_test_split(features, labels, test_size=0.33)

     # Linear regression
    # test_results, train_results = boost(train_features, train_labels, test_features, test_labels)
    # test_results, train_results = decision_tree(train_features, train_labels, test_features, test_labels, feature_names)
    # test_results, train_results = forest(train_features, train_labels, test_features, test_labels)
    # test_results, train_results = svr(train_features, train_labels, test_features, test_labels)
    test_results, train_results = linear_regressor(train_features, train_labels, test_features, test_labels)
    test_results = cap_results(test_results)
    train_results = cap_results(train_results)

    print "test result", metrics.mean_squared_error(test_labels, test_results)
    print "test r2", metrics.r2_score(test_labels, test_results)
    print "train result", metrics.mean_squared_error(train_labels, train_results)
    print "train r2", metrics.r2_score(train_labels, train_results)

    # Plot output
    fig, ax = plt.subplots()
    ax.scatter(train_labels, train_results, color="r")
    ax.scatter(test_labels, test_results, color="b")

    # Plot reference line
    ax.plot([min(test_labels), max(test_labels)], [min(test_labels), max(test_labels)], 'k--', lw=4)

    # Label axes
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.title("RANSAC")

    plt.show()

if __name__ == "__main__":
    main(sys.argv)