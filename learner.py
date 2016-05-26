'''
Reads in features, and prints out the trained weight vector.
'''
import numpy as np
import pandas as pd

train_dataframe = pd.read_csv('output/features.txt')
train_views = train_dataframe["views"].tolist()
train_favs = train_dataframe["favorites"].tolist()
train_features = train_dataframe.iloc[:,4:]
train_features = np.array(train_features)

print "train labels: "
print train_views
print 
print "train features:"
print train_features
