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

def main(args):
	# Read data
	dataframe = pd.read_csv('output/features.txt')
	
	views = dataframe["views"].tolist()
	favs = dataframe["favorites"].tolist()
	
	features = dataframe.iloc[:,4:]
	features = np.array(features)

	# Partition into training and testing datasets (approx. half and half right now)
	train_views = views[:500]
	train_favs = favs[:500]
	train_features = features[:500,:]

	test_views = views[500:]
	test_favs = favs[500:]
	test_features = features[500:,:]

	# SVR
	rbf_svr = svm.SVR(kernel = "poly", C = 1e3, gamma = "auto")
	rbf_svr.fit(train_features, train_views)
	train_results = rbf_svr.predict(train_features)
	predicted = rbf_svr.predict(test_features)

	# Linear regression
	# lr = linear_model.LinearRegression()
	# train_results = cross_val_predict(lr, train_features, train_views, cv=5)
	# train_results[train_results < 0] = 0
	# predicted = cross_val_predict(lr, test_features, test_views, cv=5)
	# predicted[predicted < 0] = 0

	train_score = np.sum(np.abs(np.divide(train_results - np.array(train_views),
	                                	  np.array(train_views))))
	test_score = np.sum(np.abs(np.divide(predicted - np.array(test_views),
	                                	 np.array(test_views))))
	print "Train score: {}".format(train_score)
	print "Test score: {}".format(test_score)

	# Plot output
	fig, ax = plt.subplots()
	ax.scatter(train_views, train_results, color="r")
	ax.scatter(test_views, predicted, color="b")
	ax.plot([min(train_views), max(train_views)], [min(train_views), max(train_views)], 'k--', lw=4)
	ax.set_xlabel('Measured')
	ax.set_ylabel('Predicted')
	plt.show()

if __name__ == "__main__":
	main(sys.argv)