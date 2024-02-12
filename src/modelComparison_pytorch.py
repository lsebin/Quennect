# imports
import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from sklearn.neural_network import MLPClassifier
import torch
from torch import nn
from torch.utils.data import DataLoader 

q_cleaned = pd.read_csv('data/data_vectorized_240228.csv')

features = q_cleaned.drop(['ia_status_Facility Study', 'ia_status_Feasibility Study',
    'ia_status_IA Executed', 'ia_status_Operational',
    'ia_status_System Impact Study', 'ia_status_Withdrawn'], axis = 1)
target = q_cleaned['ia_status_Withdrawn']

seed = 42

rus = RandomUnderSampler(random_state=seed)
X_rus, y_rus= rus.fit_resample(features, target)

# Make Custom dataset
class Q_vecDataset(torch.utils.data.Dataset):
  def __init__(self, filenames, fine, train, transforms):
        self.labels = fine
        self.list_IDs = filenames
        self.train = train
        self.transforms = transforms

  def __len__(self):
        return len(self.list_IDs)

  def __getitem__(self, index):
        fn = self.list_IDs[index].decode("utf-8")
        partition = 'train' if self.train else 'test'
        label_name = fine_label[self.labels[index]].decode("utf-8")
        X = Image.open(f"./datacifar100/{partition}/{label_name}/{fn}")
        trans = self.transforms
        X = trans(X)
        y = self.labels[index]

        return X, y

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)
# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(), # Why Tensor? also keeps weight and bias; faster math computation + automatically takes care of backpropagation;
                          # take advantage of hardware acceleration(allows implementation in GPU)
)

# batch_size n : n samples from training dataset will be used to estimate the error gradient before the model weights are updated
#              : number of samples in each batch -> smaller, less accurate in estimating gradient
# smaller batch size: more noise, lower generalization error, regularizing effect, faster learning speed
batch_size = 64

# Create data loaders.
# params: shuffle, num_workers, drop_last, etc...
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)
for X, y in test_dataloader:# X = image, y = label
    print(f"Shape of X [N, C, H, W]: {X.shape}") # Batch Dimension, Channel, Height, Width
    print(f"Shape of y: {y.shape} {y.dtype}")
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
            nn.Linear(28*28, 512), # apply linear transformation to the incoming data : y = x*W^T+b
                                    # weight here will be size of output * input
            nn.ReLU(),  # rectified linear unit function: 0 for values < 0 and linear function if > 0
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10) # output equals the number of labels-> contains prob for each label ig? 
        )
        
    def forward(self, x): 
        x = self.flatten(x) # collapse into one dimensions
        logits = self.linear_relu_stack(x)
        return logits 
    

model = NeuralNetwork().to(device)
print(model)


loss_fn = nn.CrossEntropyLoss() # log loss [0, 1]

print(model.parameters())

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3) # lr = learning rate

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        
        loss = loss_fn(pred, y)
        # Backpropagation
        loss.backward() 
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0: 
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
               
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval() # parameters no update
    test_loss, correct = 0, 0
    with torch.no_grad(): # disable gradient calculation
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            print(pred)
            test_loss += loss_fn(pred, y).item() 
            correct += (pred.argmax(1) == y).type(torch.float).sum().item() 
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