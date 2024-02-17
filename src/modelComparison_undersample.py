# imports
import pandas as pd
import argparse

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
from numpy.linalg import eigh
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from imblearn.under_sampling import RandomUnderSampler
from sklearn.neural_network import MLPClassifier

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="All")
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--seed", type=float, default=42)
    
    return parser.parse_args()

# loading our dataset
# add cleaning data
def prepare_data():
    
    q_cleaned_old = pd.read_csv('data/data_vectorized_240228.csv')
    q_cleaned_old.drop(['ia_status_Facility Study', 'ia_status_Feasibility Study',
        'ia_status_IA Executed', 'ia_status_Operational',
        'ia_status_System Impact Study', 'Unnamed: 0'], axis = 1, inplace=True)

    exempt = []
    for col in list(q_cleaned_old.columns):
        if q_cleaned_old[col].max() < 1:
            exempt.append(col)
    q_cleaned_old.drop(columns = exempt, inplace=True)
    
    # Use batch normalization here - subtract by mean of data + divide by variance
    scaler = StandardScaler()
    scaler.fit(q_cleaned_old)
    q_cleaned_array = scaler.transform(q_cleaned_old)
    q_cleaned = pd.DataFrame(q_cleaned_array, columns=q_cleaned_old.columns)
    
    # OLD: min-max scale the vectors
    #q_cleaned.apply(lambda x: (x-x.min())/(x.max()-x.min()) if x.max() > 1 else x, axis=0)
    # print(q_cleaned.max())

    features = q_cleaned.drop(['ia_status_Withdrawn'], axis = 1)
    target = q_cleaned_old['ia_status_Withdrawn']

    seed = 42

    rus = RandomUnderSampler(random_state=seed)
    X_rus, y_rus= rus.fit_resample(features, target)
    X_train, X_test, y_train, y_test = train_test_split(X_rus, y_rus,
                                                            test_size = 0.2,
                                                            random_state = seed)
    return X_train, X_test, y_train, y_test

    # q_cleaned = pd.read_csv('data/data_vectorized_240228.csv')

    # features = q_cleaned.drop(['ia_status_Facility Study', 'ia_status_Feasibility Study',
    #    'ia_status_IA Executed', 'ia_status_Operational',
    #    'ia_status_System Impact Study', 'ia_status_Withdrawn'], axis = 1)
    # target = q_cleaned['ia_status_Withdrawn']


    # # Conduct 80/20 train test split with random_state = 42
    # seed = 42

    # rus = RandomUnderSampler(random_state=seed)
    # X_rus, y_rus= rus.fit_resample(features, target)
    # X_train, X_test, y_train, y_test = train_test_split(X_rus, y_rus,
    #                                                     test_size = 0.2,
    #                                                     random_state = seed)
    
    # return X_train, X_test, y_train, y_test

def train(args=get_args()): 
    X_train, X_test, y_train, y_test = prepare_data()
    seed = 42
    models = {}

    # Ridge Regression(Logistic Regression with L2 regularization)
    if args.model == "RidgeRegression" or "All":
        print("RidgeRegression----------------------")
        log_clf = make_pipeline(StandardScaler(), SGDClassifier(loss="log_loss", eta0=1, learning_rate="constant",random_state=seed, verbose=1))
        log_clf.fit(X_train, y_train)
        models["RidgeRegression"] = log_clf

    # Lasso Regression(Logistic Regression with L1 regularization)
    if args.model == "LassoRegression" or "All":
        print("LassoRegression----------------------")
        log_clf = make_pipeline(StandardScaler(), SGDClassifier(loss="log_loss", eta0=1, learning_rate="constant", penalty='l1', random_state=seed, verbose=1))
        log_clf.fit(X_train, y_train)
        models["LassoRegression"] = log_clf

    # Perceptron
    if args.model == "Perceptron" or "All":
        print("Perceptron----------------------")
        perceptron = make_pipeline(StandardScaler(), SGDClassifier(loss="perceptron", eta0=1, learning_rate="constant", penalty=None, random_state=seed, verbose=1))
        perceptron.fit(X_train, y_train)
        models["Perceptron"] = perceptron

    # Linear SVM 
    if args.model == "LinearSVM" or "All":
        print("LinearSVM----------------------")
        svm = make_pipeline(StandardScaler(), SVC(kernel = 'linear', gamma='auto', verbose=1, random_state=seed))
        svm.fit(X_train, y_train)
        models["LinearSVM"] = svm

    # RBF SVM
    if args.model == "RBFSVM" or "All":
        print("RBFSVM----------------------")
        svm = make_pipeline(StandardScaler(), SVC(kernel = 'rbf', gamma='auto', verbose=1, random_state=seed))
        svm.fit(X_train, y_train)
        models["RBFSVM"] = svm

    # Random Forest
    if args.model == "RandomForest" or "All":
        print("RandomForest----------------------")
        rf_clf = RandomForestClassifier(criterion = 'entropy',
                                        max_depth = 10,
                                        random_state = seed, verbose=1)


        min_samples_leaf = [2, 4, 6]

        random_grid = {
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
        models["RandomForest"] = rf_random

    # Gradient Boosted Tree
    if args.model == "GradientBoostedTree" or "All":
        print("GradientBoostedTree----------------------")
        gb_bt = HistGradientBoostingClassifier(
                                        max_depth = 10,
                                        random_state = seed, verbose=1)

        gb_bt.fit(X_train, y_train)
        models["GradientBoostedTree"] = gb_bt

    # Ada Boost
    if args.model == "AdaBoost" or "All":
        print("AdaBoost----------------------")
        ad_bt = AdaBoostClassifier(n_estimators=100, random_state=seed)
        ad_bt.fit(X_train, y_train)
        models["AdaBoost"] = ad_bt

    # Neural Net
    if args.model == "NeuralNet" or "All":
        print("NeuralNet----------------------")
        nn_clf = MLPClassifier(random_state=42, max_iter=50)
        nn_clf.fit(X_train, y_train)
        models["NeuralNet"] = nn_clf

    # Performance Comparison
    for md in models: 
        model = models[md]
        # Use the model to predict on the test set
        y_pred = model.predict(X_test)
        y_trained = model.predict(X_train)

        # Find the accuracy
        
        acc_t = accuracy_score(y_train, y_trained)
        f1_t = f1_score(y_train, y_trained)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print(f"{md}------------------------")
        print("Accuracy score trained: " + str(acc_t))
        print("F1 score trained : " + str(f1_t))
        print("Accuracy score: " + str(acc))
        print("F1 score: " + str(f1))


if __name__ == "__main__":
    train()