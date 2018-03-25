#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 18:03:53 2018

@author: root
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values #Independent variable - Years of experience
y = dataset.iloc[:, 1].values #Dependent variable - Salary

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

#Fitting simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regresor = LinearRegression()
regresor.fit(X_train, y_train)

#Predicting the test set results
y_pred = regresor.predict(X_test) #Vector of predictions of dependent variable
plt.scatter(X_train,y_train, color='red')
plt.plot(X_train,regresor.predict(X_train), color='blue') #Prediction of salary 
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years')
plt.ylabel('Salary')
plt.show()

#Predicting the training set results
plt.scatter(X_test,y_test, color='red') #New sample points
plt.plot(X_train,regresor.predict(X_train), color='blue') #Prediction of salary 
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years')
plt.ylabel('Salary')
plt.show()