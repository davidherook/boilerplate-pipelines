# Example using scikit's cross_validate function for binary classification
# while tracking several scoring metrics

import pandas as pd 
import numpy as np 
from sklearn.datasets import make_blobs
from sklearn.model_selection import cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier


if __name__ == '__main__':

	print('\n\n')

	# Create dataset 
	X, y = make_blobs(n_samples=100000, 
		centers=2, 
		n_features=2, 
		random_state=0
	)

	# Define model 
	model = MLPClassifier(
		hidden_layer_sizes=(32, 32, 4),
		solver='adam',
		batch_size=1024,
		learning_rate='adaptive',
		validation_fraction=0.1,
		early_stopping=True
	)

	# Train with cross validation over [cv] folds
	cv = 4
	print(f'Running Cross Validation over {cv} folds...\n')
	scoring = ['accuracy', 'precision', 'recall', 'f1']
	scores = cross_validate(model, X, y, scoring=scoring, cv=cv, return_train_score=True)
	for metric, scores_array in scores.items():
		print('{}: mean = {:.2f}, std = {:.2f}'.format(metric, scores_array.mean(), scores_array.std()))

	print('\n\n')