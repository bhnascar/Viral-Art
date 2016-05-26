'''
Reads in features, and prints out the trained weight vector.
'''
import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.cross_validation import cross_val_predict
from sklearn import linear_model

train_dataframe = pd.read_csv('output/features.txt')
train_views = train_dataframe["views"].tolist()
train_favs = train_dataframe["favorites"].tolist()
train_features = train_dataframe.iloc[:,4:]
train_features = np.array(train_features)

# clf = svm.SVR()
lr = linear_model.LinearRegression()
# clf.fit(train_features, train_views)

# predicted = clf.predict(train_features)
predicted = cross_val_predict(lr, train_features, train_views, cv=5)
negative_indices = predicted < 0
predicted[negative_indices] = 0

score = np.sum(np.abs(np.divide(predicted - np.array(train_views),
                                np.array(train_views))))
print score

'''plot'''
fig, ax = plt.subplots()
ax.scatter(train_views, predicted)
ax.plot([min(train_views), max(train_views)], [min(train_views), max(train_views)], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()