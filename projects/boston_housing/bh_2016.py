import numpy as np
import matplotlib.pyplot as pl
import os
import warnings
import pandas as pd

from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor
import sklearn.cross_validation as cv
from sklearn.cross_validation import ShuffleSplit as ss
from sklearn.cross_validation import train_test_split as tts
from sklearn.metrics import r2_score
#import train_test_split

#import visuals as vs
cwd = os.getcwd()
print "Current Working Directory:\n", str(cwd)
####%matplotlib inline

data = pd.read_csv('housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)

#city_data = datasets.load_boston()print features.head()


print "\nBoston Housing dataset loaded successfully!\n"


CLIENT_FEATURES = [[11.95, 0.00, 18.100, 0, 0.6590, 5.6090, 90.00, 1.385, 24, 680.0, 20.20, 332.09, 12.13]]


# Number of houses in the dataset
total_houses = len(prices)

# Number of features in the dataset
#total_features = len(features[0])

# Minimum housing value in the dataset
minimum_price = min(prices)

# Maximum housing value in the dataset
maximum_price = max(prices)

# Mean house value of the dataset
mean_price = np.mean(prices)

# Median house value of the dataset
median_price = np.median(prices)

# Standard deviation of housing values of the dataset
std_dev = np.std(prices)

# Show the calculated statistics
print "\n ############## EDA ##############\n"

print "Boston housing dataset has {} data points with {} variables each.\n".format(*data.shape)
print "Boston Housing dataset statistics (in $1000's):"
print "Total number of houses:", total_houses
#print "Total number of features:", total_features, "(does not include MEDV)"
print "Minimum house price:", minimum_price
print "Maximum house price:", maximum_price
print "Mean house price: {0:.3f}".format(mean_price)
print "Median house price:", median_price
print "Standard deviation of house price: {0:.3f}\n".format(std_dev)
print "\n ############## EDA ##############\n"

print '\n###########################################################################'
print "Question 1"
print '###########################################################################\n'


def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between 
        true and predicted values based on the metric chosen. """
    
    # TODO: Calculate the performance score between 'y_true' and 'y_predict'
    score = r2_score(y_true, y_predict)
     # Return the score
    return score


#print '\n###########################################################################'

#print '###########################################################################\n'



score = performance_metric([3, -0.5, 2, 7, 4.2], [2.5, 0.0, 2.1, 7.8, 5.3])
print "Question 2: Model has a coefficient of determination, R^2, of {:.3f}.".format(score)

X_train, X_test, y_train, y_test = (None, None, None, None)

print "Question 2: {}".format(score)

def split_shuffle_data(X, y):

	X_train, X_test, y_train, y_test = cv.train_test_split(X,y, test_size = 0.2, train_size = 0.8, random_state=18)
	return X_train,X_test,y_train,y_test

print "Training and testing split was successful"

#def split_shuffle_data2(city_data):
	#, y = city_data.data, city_data.target	
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, train_size=0.8, random_state=18)


# TODO: Import 'make_scorer', 'DecisionTreeRegressor', and 'GridSearchCV'

def fit_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a 
        decision tree regressor trained on the input data [X, y]. """
    
    # Create cross-validation sets from the training data
    cv_sets = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.20, random_state = 0)

    # TODO: Create a decision tree regressor object
    regressor = None

    # TODO: Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
    params = {}

    # TODO: Transform 'performance_metric' into a scoring function using 'make_scorer' 
    scoring_fnc = None

    # TODO: Create the grid search object
    grid = None

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_

#def change_variable(X,y,n):

    #try to build a model that uses a sigmoid function and applies it to the logistic regression




