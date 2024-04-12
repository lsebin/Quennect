import numpy as np
import lime
import lime.lime_tabular
import pandas as pd
import torch
from torch import nn
import joblib

def run_lime(X_test, X_train):
    feature_names = X_train.columns
    classes = np.array(['Not Withdrawn', 'Withdrawn'])
    
    X_train = np.array(X_train)
    
    explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=feature_names, class_names=classes, discretize_continuous=True)

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
    
    # Load the model       
    filepath = "model/model_full_state_dict_epoch1k.pth"
    the_model = NeuralNetwork()
    the_model.load_state_dict(torch.load(filepath))
    the_model.eval()

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


    # This runs lime
    exp = explainer.explain_instance(X_test, model_forward_numpy, num_features=30, top_labels=1, num_samples=1000)

    pred = exp.predict_proba
    k = 0 if pred[0] > pred[1] else 1
    li = exp.as_map()
    res = li[k][:10]
    dic = {feature_names[x]: y for (x,y) in res}
    
    return pred[0], dic, feature_names

def scale_user_input(user_input):
    scaler_filename = "standard_scaler.joblib"
    scaler = joblib.load(scaler_filename)
    user_input = np.array(user_input).reshape(1, 30)
    scaled_input = scaler.transform(user_input)
    
    return scaled_input
    

if __name__ == "__main__":
    user_input = [1 for i in range(30)]
    user_input = scale_user_input(user_input)
    
    X_train = pd.read_csv('data/X_train.csv')
    X_train = X_train.drop('Unnamed: 0', axis=1)
    X_test = pd.Series((x for x in user_input[0]))
    
    predict, features_weight = run_lime(X_test, X_train)
    
    print(features_weight)
    print(predict)
    print(user_input)