import pandas as pd
import argparse, os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
import shap

BATCH_SIZE = 100
LR = 0.001
HIDDEN1 = 64
HIDDEN2 = 32

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

hidden1 = HIDDEN1
hidden2 = HIDDEN2
input_size = 30

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 2)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return(x)

def explain(point, X_train):
    model = NeuralNetwork()
    model.load_state_dict(torch.load('model/model_full_state_dict_epoch1k.pth'))
    model.eval()

    ex_point = point
    
    ex_numpy = np.expand_dims(ex_point.to_numpy(), axis=0)
    ex_tensor = torch.tensor(ex_numpy, dtype = torch.float)

    train_numpy = X_train.to_numpy()
    train_tensor = torch.tensor(train_numpy, dtype = torch.float)

    explainer = shap.GradientExplainer(model, train_tensor)
    shap_values = explainer.shap_values(ex_tensor) #[:,:,0]
    print(len(shap_values))
    shap.summary_plot(shap_values, ex_tensor, feature_names = X_train.columns)

    shap_pd = pd.DataFrame(shap_values, columns = X_train.columns)
    top_ind = np.argsort(shap_pd.values.flatten())[-5:][::-1]
    top_features = shap_pd.columns[top_ind].to_numpy()
    top_shap = shap_pd.values.flatten()[top_ind]
    return top_features, top_shap

if __name__ == "__main__":
    X_train = pd.read_csv('data/X_train.csv')
    X_train = X_train.drop('Unnamed: 0', axis=1)
    user_input = X_train.iloc[0,:] #pd.Series((1 for i in range(30)))
    top_neg_features, shap_vals = explain(user_input, X_train)
    print(top_neg_features)
    print(shap_vals)