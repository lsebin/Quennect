# Model 0: Testing out non-neural network models that should be less computationally expensive

# imports
import pandas as pd

import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from numpy.linalg import eigh
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC

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

models = {}

# Ridge Regression(Logistic Regression with L2 regularization)
log_clf = LogisticRegression(max_iter = 1000, random_state=seed) 
log_clf.fit(X_train, y_train)
models["Ridge regression"] = log_clf

# Lasso Regression(Logistic Regression with L1 regularization)
log_clf = LogisticRegression(max_iter = 1000, random_state=seed, penalty='l1', solver='saga') 
log_clf.fit(X_train, y_train)
models["Lasso regression"] = log_clf

# Perceptron
perceptron = make_pipeline(StandardScaler(), SGDClassifier(loss="perceptron", eta0=1, learning_rate="constant", penalty=None, max_iter=1000, random_state=seed))
perceptron.fit(X_train, y_train)
models["perceptron"] = perceptron

# Linear SVM 
svm = make_pipeline(StandardScaler(), SVC(kernel = 'linear', gamma='auto', max_iter=1000, random_state=seed))
svm.fit(X_train, y_train)
models["Linear SVM"] = svm

# RBF SVM
svm = make_pipeline(StandardScaler(), SVC(kernel = 'rbf', gamma='auto', max_iter=1000, random_state=seed))
svm.fit(X_train, y_train)
models["RBF SVM"] = svm

# Random Forest
rf_clf = RandomForestClassifier(criterion = 'entropy',
                                max_depth = 10,
                                random_state = seed)

max_features = ['auto', 'sqrt']

min_samples_leaf = [2, 4, 6]

random_grid = {'max_features': max_features,
               'min_samples_leaf': min_samples_leaf}

rf_random = RandomizedSearchCV(estimator = rf_clf,
                               param_distributions = random_grid,
                               n_iter = 10,
                               cv = 3,
                               verbose = 2,
                               random_state= seed,
                               n_jobs = -1,
                               )

rf_random.fit(X_train, y_train)
models["random forest"] = rf_random

# Gradient Boosted Tree
gb_bt = HistGradientBoostingClassifier(
                                max_depth = 10,
                                random_state = seed)

gb_bt.fit(X_train, y_train)
models["Gradient Boosted Tree"] = gb_bt

# Ada Boost
ad_bt = AdaBoostClassifier(n_estimators=100, random_state=seed)
ad_bt.fit(X_train, y_train)
models["Ada Boost"] = ad_bt



# Performance Comparison
for md in models: 
    model = models[md]
    # Use the model to predict on the test set
    y_pred = model.predict(X_test)

    # Find the accuracy
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"{md}------------------------")
    print("Accuracy score: " + str(acc))
    print("F1 score: " + str(f1))


