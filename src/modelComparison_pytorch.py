# imports
import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader 

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
    
    # OLD: min-max scale the vectors
    #q_cleaned.apply(lambda x: (x-x.min())/(x.max()-x.min()) if x.max() > 1 else x, axis=0)
    # print(q_cleaned.max())

    features = q_cleaned.drop(['ia_status_Withdrawn'], axis = 1)
    target = q_cleaned_old['ia_status_Withdrawn']

    seed = args.seed

    rus = RandomUnderSampler(random_state=seed)
    X_rus, y_rus= rus.fit_resample(features, target)
    X_train, X_test, y_train, y_test = train_test_split(X_rus, y_rus,
                                                            test_size = 0.2,
                                                            random_state = seed)
    return X_train, X_test, y_train, y_test

def run(args=get_args()):
    # print(args.hidden, args.hidden1, args.hidden2)
    # exit()
    X_train, X_test, y_train, y_test = prepare_data()
    training_data = Q_vecDataset(X_rus = X_train, y_rus = y_train, train=True)
    test_data = Q_vecDataset(X_rus = X_test, y_rus = y_test, train=False)

    batch_size = 200

    # Create data loaders.
    # TODO: Double check that batches are as expected
    # params: shuffle, num_workers, drop_last, etc...
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    input_dim = 0
    for X, y in test_dataloader:# X = image, y = label
        print(f"Shape of X [N, C, H, W]: {X.shape}") # Batch Dimension, Channel, Feature #
        print(f"Shape of y: {y.shape} {y.dtype}")
        input_dim =list(X.size())[2]
        break

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
        
    print(f"Using {device} device")


    # Define model
    # In fully connected nn, number of units should always decrease
    # Neural network -> input-hidden-output layers
    class NeuralNetwork(nn.Module): # nn.Module = base case for all neural network modules
    # we define model as a subclass of nn.Module -> it creates parameters of the modules with utility methods like eval()
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(input_dim, int(args.hidden)), # apply linear transformation to the incoming data : y = x*W^T+b
                                        # weight here will be size of output * input
                nn.ReLU(),  # rectified linear unit function: 0 for values < 0 and linear function if > 0
                nn.Linear(int(args.hidden), int(args.hidden1)),
                nn.ReLU(),
                nn.Linear(args.hidden1, args.hidden2),
                nn.ReLU(),
                nn.Linear(args.hidden2, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )
            self.sig = nn.Sigmoid() 
            # TODO: BCELoss does not expect raw logits - every value should be in the range [0,1].
            # TODO: Check what the previous model was doing, if there was regularization, learning rate, etc.
            
        def forward(self, x): 
            x = self.flatten(x) # collapse into one dimensions
            x = self.linear_relu_stack(x)
            x = self.sig(x)
            return x.squeeze() # changed to squeeze
        

    model = NeuralNetwork().to(device)
    # print(model)

    loss_fn = nn.BCELoss() # log loss [0, 1]
    # TODO: Check if BCELoss takes 1 value or 2 - what inputs exactly it needs
    print(model.parameters())

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) # lr = learning rate

    def train(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        correct , train_loss = 0, 0
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device).to(torch.float32), y.to(device).to(torch.float32)
            pred = model(X)
            # added squeezing y
            y = y.squeeze()
            # print(pred, y)
            correct += (torch.round(pred) == y).type(torch.float).sum().item() 
            # print(pred, y)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()
            train_loss += (loss.item())

        train_loss /= num_batches 
        correct /= size 
        print(f"Train: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {train_loss:>8f} \n")
        return correct, train_loss

        
                
    def test(dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0
        model.eval() # parameters no update
        with torch.no_grad(): # disable gradient calculation
            for X, y in dataloader:
                X, y = X.to(device).to(torch.float32), y.to(device).to(torch.float32)
                pred = model(X)
                y = y.squeeze()
                # print(pred, y)
                loss = loss_fn(pred,y)
                test_loss += loss.item() 
                correct += (torch.round(pred) == y).type(torch.float).sum().item() 
        test_loss /= num_batches      
        correct /= size # the overall accuracy 
        print(f"Test: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        return correct, test_loss
        

    # one training epoch -> algo made one pass through the training dataset
    epochs = args.epoch
    tl = []
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_acc, train_loss = train(train_dataloader, model, loss_fn, optimizer)
        test_acc, test_loss= test(test_dataloader, model, loss_fn)
    print("Done!")

if __name__ == "__main__":
    run()