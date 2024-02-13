# imports
import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader 

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=float, default=3)
    parser.add_argument("--params", type=float, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=float, default=42)
    
    return parser.parse_args()

q_cleaned = pd.read_csv('data/data_vectorized_240228.csv')
q_cleaned.drop(['ia_status_Facility Study', 'ia_status_Feasibility Study',
    'ia_status_IA Executed', 'ia_status_Operational',
    'ia_status_System Impact Study', 'Unnamed: 0'], axis = 1, inplace=True)

# min-max scale the vectors
#q_cleaned.apply(lambda x: (x-x.min())/(x.max()-x.min()) if x.max() > 1 else x, axis=0)
# print(q_cleaned.max())
exempt = []
for col in list(q_cleaned.columns):
    if q_cleaned[col].max() < 1:
        exempt.append(col)
q_cleaned.drop(columns = exempt, inplace=True)
q_cleaned=(q_cleaned-q_cleaned.min())/((q_cleaned.max()-q_cleaned.min()))
# print(q_cleaned.max())

features = q_cleaned.drop(['ia_status_Withdrawn'], axis = 1)
target = q_cleaned['ia_status_Withdrawn']

seed = 42

rus = RandomUnderSampler(random_state=seed)
X_rus, y_rus= rus.fit_resample(features, target)
X_train, X_test, y_train, y_test = train_test_split(X_rus, y_rus,
                                                        test_size = 0.2,
                                                        random_state = seed)

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


training_data = Q_vecDataset(X_rus = X_train, y_rus = y_train, train=True)
test_data = Q_vecDataset(X_rus = X_test, y_rus = y_test, train=False)

batch_size = 64

# Create data loaders.
# params: shuffle, num_workers, drop_last, etc...
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)
iparam = 0
for X, y in test_dataloader:# X = image, y = label
    print(f"Shape of X [N, C, H, W]: {X.shape}") # Batch Dimension, Channel, Feature #
    print(f"Shape of y: {y.shape} {y.dtype}")
    iparam =list(X.size())[2]
    break

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
    
print(f"Using {device} device")

# Define model
# TODO: what exactly is nn.Module? differece between nn.Module.Functional? 
# Functional is not a full layer -> just arithmetic operation(does not have trainable parameters like weight, bias, etc) therefore use for usu simple operation
# Architecture of the model
# In fully connected nn, number of units should always decrease
# Neural network -> input-hidden-output layers
class NeuralNetwork(nn.Module): # nn.Module = base case for all neural network modules
# we define model as a subclass of nn.Module -> it creates parameters of the modules with utility methods like eval()
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential( # TODO: what is diff btw module and sequential?
                                               # nn.Sequential = sequential container where it accepts any input and forwards it to the first module and chains the output
                                               # allows to treat multiple layers as one container -> quickly implements sequential modules but module has more flexibility 
            nn.Linear(iparam, 128), # apply linear transformation to the incoming data : y = x*W^T+b
                                    # weight here will be size of output * input
            nn.ReLU(),  # rectified linear unit function: 0 for values < 0 and linear function if > 0
            nn.Linear(128, 64),
            nn.ReLU(),
            # nn.Linear(64, 32),
            # nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.sig = nn.Sigmoid()
        
    def forward(self, x): 
        x = self.flatten(x) # collapse into one dimensions
        x = self.linear_relu_stack(x)
        x = self.sig(x)
        # print(x)
        # print(torch.round(x))
        # exit()
        # return label? 
        return torch.round(x)
    

model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.BCELoss() # log loss [0, 1]
print(model.parameters())

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) # lr = learning rate

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device).to(torch.float32), y.to(device).to(torch.float32)
        pred = model(X)
        # print(pred, y)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()
        
        if batch % 100 == 0: 
            loss, current = abs(loss.item()), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
               
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval() # parameters no update
    test_loss, correct = 0, 0
    with torch.no_grad(): # disable gradient calculation
        for X, y in dataloader:
            X, y = X.to(device).to(torch.float32), y.to(device).to(torch.float32)
            pred = model(X)
            # print(pred)
            test_loss += loss_fn(pred, y).item() 
            correct += (pred == y).type(torch.float).sum().item() 
    test_loss /= num_batches 
                            
    correct /= size # the overall accuracy 
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
# one training epoch -> algo made one pass through the training dataset
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")