# Decision Tree Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train.csv')
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values
dataset1 = pd.read_csv('test.csv')
X_real = dataset1.iloc[:,:].values
# Splitting the dataset into the Training set and Test set
#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeClassifier
regressor = DecisionTreeClassifier()
regressor.fit(X, y)

# Predicting a new result
y_pred = regressor.predict(X_real)

import csv
fp = open('pred.csv','w')
writer = csv.writer(fp)

writer.writerow(["ImageId", "Label"])
e_sum = 0
false_predictions = 0

for i in range(y_pred.size):
	writer.writerow([i+1,y_pred[i]])


