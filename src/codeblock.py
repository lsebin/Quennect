import sklearn
import sklearn.datasets
import sklearn.ensemble
import numpy as np
import lime
import lime.lime_tabular
from __future__ import print_function
import pandas as pd
import argparse, os
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader 

classes = np.array(['Not Withdrawn', 'Withdrawn'])
feature_names = ['year_entering_queue', 'proposed_year', 'region_CAISO', 'region_MISO',
       'region_PJM', 'region_Southeast (non-ISO)', 'region_West (non-ISO)',
       'project_size_mw', 'project_latitude', 'population_density',
       'votes_dem', 'votes_rep', 'votes_total', 'voting_density',
       'pct_dem_lead', 'solar_potential', 'wind_potential', 'is_deregulated',
       'has_100_clean_energy_goal', 'top_ten_renewable_generators', 'is_solar',
       'is_storage', 'is_wind', 'is_bioenergy', 'is_wasteuse',
       'is_cleanenergy', 'is_fossilfuels', 'is_hybrid',
       'high_revenue_utility']

explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=feature_names, class_names=classes, discretize_continuous=True)

class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(29, int(128)), 
                nn.ReLU(), 
                nn.Linear(int(128), int(64)),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, 2),
            )
            
        def forward(self, x): 
            x = self.linear_relu_stack(x)
            return x # changed to squeeze
        
filepath = "" #os.path.join("data", "model", "epoch5_lime_woutil.pt")
the_model = NeuralNetwork()
the_model = torch.load(filepath)

# Define a function to convert input from NumPy arrays to PyTorch tensors
def numpy_to_tensor(x):
    return torch.tensor(x, dtype=torch.float32)

# Define a function to convert output from PyTorch tensors to NumPy arrays
def tensor_to_numpy(x):
    return x.detach().numpy()

# Wrap the forward pass of your model with NumPy-to-tensor and tensor-to-NumPy conversions
def model_forward_numpy(x_numpy):
    x_tensor = numpy_to_tensor(x_numpy)
    output_tensor = the_model(x_tensor)
    output_tensor = torch.nn.functional.softmax(output_tensor, dim=1)
    output_numpy = tensor_to_numpy(output_tensor)
    return output_numpy

X_test = ""

exp = explainer.explain_instance(X_test, model_forward_numpy, num_features=29, top_labels=1, num_samples=1000)

pred = exp.predict_proba
k = 0 if pred[0] > pred[1] else 1
li = exp.as_map()
res = li[k][:10]
dic = {feature_names[x]: y for (x,y) in res}
