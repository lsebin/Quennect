import pandas as pd

import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.decomposition import PCA
from numpy.linalg import eigh

q_cleaned = pd.read_csv('/Users/shanewilliams/Quennect/src/q_data_cleaned.csv')
print(q_cleaned.columns)

features = q_cleaned.drop(['ia_status_Withdrawn'], axis = 1)
target = q_cleaned['ia_status_Withdrawn']

print(target.value_counts())


# Conduct 80/20 train test split with random_state = 42
seed = 42
X_train, X_test, y_train, y_test = train_test_split(features, target,
                                                     test_size = 0.2,
                                                     random_state = seed)

rf_clf = RandomForestClassifier(criterion = 'entropy',
                                max_depth = 10,
                                random_state = seed)
# Hyperparameter tuning

# Number of features to consider when looking for the best split
max_features = ['auto', 'sqrt']

# Minimum number of samples required at each leaf node
min_samples_leaf = [2, 4, 6]

# Create the random grid
random_grid = {'max_features': max_features,
               'min_samples_leaf': min_samples_leaf}

# find best parameters from randomized search CV
rf_random = RandomizedSearchCV(estimator = rf_clf,
                               param_distributions = random_grid,
                               n_iter = 10,
                               cv = 3,
                               verbose = 2,
                               random_state= seed,
                               n_jobs = -1,
                               )

rf_random.fit(X_train, y_train)

# Use the model to predict on the test set
y_pred = rf_random.predict(X_test)

# Find the accuracy
rf_acc = accuracy_score(y_test, y_pred)
rf_f1 = f1_score(y_test, y_pred)
print("Accuracy score: " + str(rf_acc))
print("F1 score: " + str(rf_f1))

