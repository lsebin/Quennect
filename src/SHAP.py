# imports
import pandas as pd
import argparse, os
import numpy as np
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader
import shap 

if __name__ == "__main__":
    
    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--layer", type=float, default=3)
        parser.add_argument("--hidden", type=int, default=128)
        parser.add_argument("--hidden1", type=int, default=64)
        parser.add_argument("--hidden2", type=int, default=32)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--seed", type=float, default=42)
        parser.add_argument("--epoch", type=int, default=50)
        
        return parser.parse_args()
    
    # Make Custom dataset
    class Q_vecDataset(torch.utils.data.Dataset):
        def __init__(self, X_rus, y_rus, train):
            self.feature = X_rus
            self.target = y_rus
            self.train = train

        def __len__(self):
            return self.target.shape[0]

        def __getitem__(self, index):
            X = self.feature.iloc[[index]]
            X = X.to_numpy()
            y = self.target.iloc[[index]]
            y= y.to_numpy()

            return X, y
        
    def prepare_data(args=get_args()):
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
    

        features = q_cleaned.drop(['ia_status_Withdrawn'], axis = 1)
        target = q_cleaned_old['ia_status_Withdrawn']

        seed = 42

        rus = RandomUnderSampler(random_state=seed)
        X_rus, y_rus= rus.fit_resample(features, target)
        X_train, X_test, y_train, y_test = train_test_split(X_rus, y_rus,
                                                            test_size = 0.2,
                                                            random_state = seed)
        return X_train, X_test, y_train, y_test
    
    class NeuralNetwork(nn.Module): # nn.Module = base case for all neural network modules
    # we define model as a subclass of nn.Module -> it creates parameters of the modules with utility methods like eval()
        def __init__(self):
            super().__init__()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(140, int(128)), # apply linear transformation to the incoming data : y = x*W^T+b
                                        # weight here will be size of output * input
                nn.ReLU(),  # rectified linear unit function: 0 for values < 0 and linear function if > 0
                nn.Linear(int(128), int(64)),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, 2),
            )
            #self.sig = nn.Sigmoid() 
            #self.softmax = nn.Softmax(dim=1)
            # TODO: BCELoss does not expect raw logits - every value should be in the range [0,1].
            # TODO: Check what the previous model was doing, if there was regularization, learning rate, etc.
            
        def forward(self, x): 
            x = self.linear_relu_stack(x)
            #x = self.sig(x)
            #x = self.softmax(x)
            return x # changed to squeeze
        
    def train(dataloader, model, loss_fn, optimizer, device):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        correct , train_loss = 0, 0
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device).to(torch.float32), y.to(device)
            y = y.squeeze()
            pred = model(X)
            # print(pred, y)
            # print(pred.dtype, y.dtype)
            loss = loss_fn(pred, y)
            _, lbls = torch.max(pred.data, 1)
            correct += (lbls == y).type(torch.float).sum().item() 
            # print(pred, y)
        
            # Backpropagation
            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()
            train_loss += (loss.item())

        train_loss /= num_batches 
        correct /= size 
        print(f"Train: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {train_loss:>8f} \n")
        return correct, train_loss
    
    def test(dataloader, model, loss_fn, device):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0
        model.eval() # parameters no update
        with torch.no_grad(): # disable gradient calculation
            for X, y in dataloader:
                X, y = X.to(device).to(torch.float32), y.to(device)
                y = y.squeeze()
                pred = model(X)
                # print(outputs,pred, y)

                _, lbls = torch.max(pred.data, 1)
                correct += (lbls == y).type(torch.float).sum().item() 
                loss = loss_fn(pred, y)
                test_loss += loss.item() 
        test_loss /= num_batches      
        correct /= size # the overall accuracy 
        print(f"Test: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        return correct, test_loss
        
    X_train, X_test, y_train, y_test = prepare_data()
    training_data = Q_vecDataset(X_rus = X_train, y_rus = y_train, train=True)
    test_data = Q_vecDataset(X_rus = X_test, y_rus = y_test, train=False)
    args = get_args()
    epochs = 5
    batch_size = 200
    
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    input_dim = 0
    for X, y in test_dataloader:# X = image, y = label
        print(f"Shape of X [N, C, H, W]: {X.shape}") # Batch Dimension, Channel, Feature #
        print(f"Shape of y: {y.shape} {y.dtype}")
        input_dim =list(X.size())[2]
        break
    print("input_dim", input_dim)
    
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"  
    
    model = NeuralNetwork().to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) # lr = learning rate
    loss_fn = nn.CrossEntropyLoss()

    # one training epoch -> algo made one pass through the training dataset
    test_acc_list = []
    test_loss_list = []
    train_acc_list = []
    train_loss_list = []
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_acc, train_loss = train(train_dataloader, model, loss_fn, optimizer, device)
        test_acc, test_loss= test(test_dataloader, model, loss_fn, device)
        
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)
        test_acc_list.append(test_acc)
        test_loss_list.append(test_loss) 
    print("Done!")
    
    def model_wrapper(x):
        model.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x).to(device)
            predictions = model(x_tensor)
            return(predictions.cpu().numpy())
            
    background_data =  X_train.numpy()
    explainer = shap.DeepExplainer(model_wrapper, background_data)
    test_data_numpy = X_test.numpy()
    shap_values = explainer.shap_values(test_data_numpy)
    shap.summary_plot(shap_values, test_data_numpy, feature_names=X_test.columns)
    shap.force_plot(explainer.expected_value[0], shap_values[0][0], test_data_numpy[0], feature_names = X_test.columns)