# Model 0: Testing out non-neural network models that should be less computationally expensive

# imports
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
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.decomposition import PCA
from numpy.linalg import eigh

# loading our dataset
q_raw = pd.read_csv('Users\carol\Downloads\q_data_encoded.csv')
# q_raw = pd.read_csv('data/q_data_encoded.csv')

# peek at the data
q_raw.head()
q_raw.describe()
q_raw.info()

# keep only relevant content
columns_to_keep = ['q_date', 'q_year', 'prop_year', 'ia_status_raw', 'mw1', 'mw2', 'mw3']
q_revised = q_raw[columns_to_keep].copy()
# print(q_revised)

# drop IA statuses that are not within range of answers
# unique_ia_vals = q_revised['ia_status_raw'].unique()
# print(unique_ia_vals)

value_counts = q_revised['ia_status_raw'].value_counts().head(20)
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
print(q_revised)

# q_revised['q_date'] = pd.to_datetime(q_revised['q_date'])

# make train/test split
#q_train = 