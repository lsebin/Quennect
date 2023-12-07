# Model 0: Testing out non-neural network models that should be less computationally expensive

# imports
import pandas as pd

import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from numpy.linalg import eigh

# loading our dataset
q_cleaned = pd.read_csv('q_data_cleaned.csv')
print(q_cleaned.columns)

features = q_cleaned.drop(['ia_status_Withdrawn'], axis = 1)
target = q_cleaned['ia_status_Withdrawn']

print(target.value_counts())


# Conduct 80/20 train test split with random_state = 42
seed = 42
X_train, X_test, y_train, y_test = train_test_split(features, target,
                                                     test_size = 0.2,
                                                     random_state = seed)

# TODO: Implement baseline model: multiclass logistic regression 
#       - 85% accuracy, but only predicting based on year project was 
#       entered, not a go

# Initialize model with default parameters and fit it on the training set
log_clf = LogisticRegression(max_iter = 1000) # higher max_iter for large data size
log_clf.fit(X_train, y_train)

# Use the model to predict on the test set
y_pred = log_clf.predict(X_test)

# Find the accuracy
log_acc = accuracy_score(y_test, y_pred)
log_f1 = f1_score(y_test, y_pred)
print("Accuracy score: " + str(log_acc))
print("F1 score: " + str(log_f1))

