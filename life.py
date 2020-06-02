# Predicting Life expectancy 

# Import Libraries 
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt
import statsmodels.api as sm
from scipy import stats
from sklearn import metrics

# Import the dataset
life = pd.read_csv('Life Expectancy Data.csv')

# Remove the Country column
life = life.drop(['Country'], axis=1)

# Now let us try to do something with the missing values
life.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
life.isna().sum()


# One hot encoding the categorical values 

from sklearn.base import TransformerMixin
class DataFrameEncoder(TransformerMixin):

    def __init__(self):
        """Encode the data.

        Columns of data type object are appended in the list. After 
        appending Each Column of type object are taken dummies and 
        successively removed and two Dataframes are concated again.

        """
    def fit(self, life, y=None):
        self.object_col = []
        for col in life.columns:
            if(life[col].dtype == np.dtype('O')):
                self.object_col.append(col)
        return self

    def transform(self, life, y=None):
        dummy_df = pd.get_dummies(life[self.object_col],drop_first=True)
        life = life.drop(life[self.object_col],axis=1)
        life = pd.concat([dummy_df,life],axis=1)
        return life
life_Enc = DataFrameEncoder().fit_transform(life)
print(life_Enc)


# Divide the dataset into depenedent and independent variables
X = life_Enc.iloc[:, :-1]
y = life_Enc.iloc[:, -1]


#Splitting the dataset into training and testing 
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X, y,test_size = 0.2 , random_state = 0)


# Now let us start Regression 

# Simple Linear Regression
# fitting simple linear regression on the training set
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#prediciting the test set results
y_pred = regressor.predict(X_test)

# Print result of MAE
print(metrics.mean_absolute_error(y_test, y_pred))
# Print result of MSE
print(metrics.mean_squared_error(y_test, y_pred))
# Print result of RMSE
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
r2 = r2_score(y_test,y_pred)
print(r2)


# Decision Tree Regression 
# APPLYING DECISION TREE ON THE TRAINING DATASET

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)

# Predicting a new result
y_pred = pd.DataFrame(regressor.predict(X_test))
# Print result of MAE
print(metrics.mean_absolute_error(y_test, y_pred))
# Print result of MSE
print(metrics.mean_squared_error(y_test, y_pred))
# Print result of RMSE
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# r2 Score
r2 = r2_score(y_test,y_pred)
print(r2)


# FITTING RANDOM FOREST TO THE TRAINING SET

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 1000, random_state = 0)
regressor.fit(X_train, y_train)

# Predicting on test set 
y_pred = pd.DataFrame(regressor.predict(X_test))

# Print result of MAE
print(metrics.mean_absolute_error(y_test, y_pred))
# Print result of MSE
print(metrics.mean_squared_error(y_test, y_pred))
# Print result of RMSE
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# r2 Score
r2 = r2_score(y_test,y_pred)
print(r2)
















