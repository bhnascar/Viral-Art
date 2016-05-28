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
	
	selected_features = [7, 8, 9, 10, 11, 12, 34, 35, 38, 49]
	features = dataframe.iloc[:, 5:]
	features = np.array(features)
	print features[0,:]

	# Partition into training and testing datasets (approx. half and half right now)
	train_views = views[:500]
	train_favs = favs[:500]
	train_features = features[:500,:]

	test_views = views[500:]
	test_favs = favs[500:]
	test_features = features[500:,:]

	# SVR
	#rbf_svr = svm.SVR(kernel = "rbf", C = 1e3, gamma = "auto")
	#rbf_svr.fit(train_features, train_views)
	#train_results = rbf_svr.predict(train_features)
	#test_results = rbf_svr.predict(test_features)

	# Linear regression
	lr = linear_model.LinearRegression()
	train_results = cross_val_predict(lr, train_features, train_favs, cv=5)
	train_results[train_results < 0] = 0
	test_results = cross_val_predict(lr, test_features, test_favs, cv=5)
	test_results[test_results < 0] = 0

	train_score = np.sum(np.abs(np.divide(train_results - np.array(train_favs),
	                                	  np.array(train_favs))))
	test_score = np.sum(np.abs(np.divide(test_results - np.array(test_favs),
	                                	 np.array(test_favs))))
	print "Train score: {}".format(train_score)
	print "Test score: {}".format(test_score)

	# Plot output
	fig, ax = plt.subplots()
	ax.scatter(train_views, train_results, color="r")
	ax.scatter(test_views, test_results, color="b")
	# ax.plot([min(train_views), max(train_views)], [min(train_views), max(train_views)], 'k--', lw=4)
	ax.plot([min(train_favs), max(train_favs)], [min(train_favs), max(train_favs)], 'k--', lw=4)
	ax.set_xlabel('Measured')
	ax.set_ylabel('Predicted')
	plt.show()

if __name__ == "__main__":
	main(sys.argv)