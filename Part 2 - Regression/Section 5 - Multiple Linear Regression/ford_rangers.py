# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('ford_rangers.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 2] = labelencoder_X.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [2])
X = onehotencoder.fit_transform(X).toarray()

# avoid dummy variable trap
X = X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# fitting multiple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# predict the test set
y_pred = regressor.predict(X_test)

singlePred = 
# building the optimal model using backwards elimination
# add a column of ones to the features
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((24,1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0,1,2,3,4,5,6,7,8,9]]
regressor_OLS = sm.OLS(endog=y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0,1,2,3,4,6,7,8,9]]
regressor_OLS = sm.OLS(endog=y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0,1,2,4,6,7,8,9]]
regressor_OLS = sm.OLS(endog=y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0,1,3,4,6,7,8,9]]
regressor_OLS = sm.OLS(endog=y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0,1,4,6,7,8,9]]
regressor_OLS = sm.OLS(endog=y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0,4,6,7,8,9]]
regressor_OLS = sm.OLS(endog=y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0,4,6,7,9]]
regressor_OLS = sm.OLS(endog=y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0,4,6,9]]
regressor_OLS = sm.OLS(endog=y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt_test = X_test[:, [3,5,8]]
y_pred = regressor_OLS.predict(X_opt_test)
