import pandas as pd

import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import json
from plotly import graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.decomposition import PCA
from numpy.linalg import eigh

q_cleaned = pd.read_csv('q_data_cleaned.csv')
'''
q_raw = pd.read_csv('q_data_encoded.csv')

# peek at the data
# q_raw.head()
# q_raw.describe()
# q_raw.info()

# keep only relevant content
columns_to_keep = ['q_date', 'q_year', 'prop_year', 'ia_status_raw', 'mw1', 'mw2', 'mw3']
q_revised = q_raw[columns_to_keep].copy()
# print(q_revised)

# drop IA statuses that are not within range of answers
# unique_ia_vals = q_revised['ia_status_raw'].unique()
# print(unique_ia_vals)

value_counts = q_revised['ia_status_raw'].value_counts().head(40)
# print(value_counts)
# exit()


# print(value_counts)

# drop irrelevant columns and one hot encode
q_revised['ia_status_category'] = q_revised['ia_status_raw'].replace({
    'System Impact Study': 'System Impact Study',
    'IA Executed': 'IA Executed',
    'Executed': 'IA Executed',
    'Withdrawn': 'Withdrawn',
    'Feasibility Study': 'Feasibility Study',
    'Feasibility': 'Feasibility Study',
    'In progress': 'In progress',
    'Active': 'Operational',
    'Operational': 'Operational',
    'operational': 'Operational',
    'Facility Study': 'Facility Study'
})

q_revised = q_revised[q_revised['ia_status_category'].isin([
    'System Impact Study',
    'IA Executed',
    'Withdrawn',
    'Feasibility Study',
    'In progress',
    'Operational',
    'Facility Study'
])]


# One-hot encode the new column
encoded_categories = pd.get_dummies(q_revised['ia_status_category'], prefix='ia_status').astype(int)

# Concatenate the one-hot encoded columns with the original DataFrame
q_revised = pd.concat([q_revised, encoded_categories], axis=1)
# print(q_revised)

# Create total MW
columns_to_sum = ['mw1', 'mw2', 'mw3']

# Create a new column 'total_MW' by summing across specified columns
q_revised['total_mw'] = q_revised[columns_to_sum].sum(axis=1, skipna=True)
# print(q_revised)

q_cleaner = q_revised.copy().drop(columns=['mw1', 'mw2','mw3']).dropna()
# print(q_cleaner)

q_cleaner = q_cleaner[q_cleaner['q_date'].str.match(r'\d{1,2}/\d{1,2}/\d{4}$')]
# # Convert the remaining dates to datetime
q_cleaner['q_date'] = pd.to_datetime(q_cleaner['q_date'])

q_cleaner = q_cleaner.drop(columns=['ia_status_raw', 'ia_status_category', 'q_date'])


print(q_cleaner.info())
'''
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