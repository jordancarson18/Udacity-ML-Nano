###########################################
# Suppress matplotlib user warnings
# Necessary for newer version of matplotlib
import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
#
# Display inline matplotlib plots with IPython
from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')
###########################################

import numpy as np

import pandas as pd

from IPython.display import display


in_file = '/Users/jordancarson/Documents/Programming/ML/projects/titanic_survival_exploration/titanic_data.csv'

full_data = pd.read_csv(in_file)
display(full_data.head())

outcomes = full_data['Survived']
data = full_data.drop('Survived', axis=1)
display(data.head())

outputs = []
source = []

def accuracy_score(truth, pred):
	if len(truth) == len(pred):
		return "Predictions have an accuracy of {:.2f}%".format((truth ==pred).mean()*100)
	else:
		return "Number of predictions does not match number of outcomes!"

predictions = pd.Series(np.ones(5, dtype=int))
print accuracy_score(outcomes[:5], predictions)

def predictions_0(data):
	predictions = []

	for _, passenger in data.iterrows():

		predictions.append(0)

	return pd.Series(predictions)\
	ipop
-predictions = predictions_0(data)

print accuracy_score(outcomes, predictions)

print type(predictions)
print type(outcomes)

def predictions_1(data):

	predictions = []
	for _, passenger in data.iterrows():
		if passenger['Sex'] == 'female': predictions.append(1)
		else: predictions.append(0)

	return pd.Series(predictions)
predictions = predictions_1(data)

print accuracy_score(outcomes, predictions)
