# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 11:02:20 2019

@author: Hemraj
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import cross_validate
dataset = pd.read_csv('winequality.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 11].values

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)"""

# Predicting the Test set results
#y_pred = regressor.predict(X_test)
cv_results=(cross_validate(regressor,X, y, cv=3, scoring='neg_mean_squared_error'))
print(-(cv_results['test_score'].mean()))
