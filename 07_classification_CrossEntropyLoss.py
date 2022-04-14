import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from numpy import random
from torch.nn.functional import normalize
'''
This code uses 'CrossEntropyLoss' from Pytorch, while using this loss keep in mind
-> nn.CrossEntropyLoss takes care of softmax, so its not needed in actual network
-> CrossEntropyLoss does not expect a one-hot encoded vector as the target, but class indices:
   input has to be a 2D Tensor of size (minibatch, C).
   This loss/criterion expects a class index (0 to C-1) as the target for each value of a 1D tensor of size minibatch.
'''
data = pd.read_csv('wine_red.csv',delimiter = ';')
print(data.head())

torch.manual_seed(23)

# Device configuration
device = torch.device('cpu')

dat = data.to_numpy(dtype= np.float32) # dataframe  to numpy
# Split data into input and labels
x = torch.from_numpy(dat[:,0:11])
x = torch.nn.functional.normalize(x, p=2.0, dim = 0)
y = torch.from_numpy(dat[:,[11]])

classes = torch.unique(y)

# Split data into test and train
train_size = int(0.8*len(dat))
test_size = len(dat)- train_size
x_train, x_test = torch.utils.data.random_split(x, [train_size, test_size])
y_train, y_test = torch.utils.data.random_split(y, [train_size, test_size])


# define train and test Datasets
class CustomDataset_train(Dataset):
    def __init__(self):
        self.x_train = x_train
        self.y_train = y_train
        self.num_samples = len(x_train)

    def __getitem__(self, index):
        return self.x_train[index], self.y_train[index]

    def __len__(self):
        return self.num_samples

class CustomDataset_test(Dataset):

    def __init__(self):
        self.x_test = x_test
        self.y_test = y_test
        self.num_samples = len(x_test)

    def __getitem__(self, index):
        return self.x_test[index], self.y_test[index]

    def __len__(self):
        return self.num_samples

data_train = CustomDataset_train()
data_test = CustomDataset_test()

batch_size= 10 #set batch size

train_dataloader = DataLoader(dataset = data_train, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(dataset = data_test, batch_size=batch_size, shuffle=True)

train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

#---------------------------------------------------------------------------------
input_size = x.shape[1]
output_size = len(classes)
print("No. of classes:"+str(output_size))
# define network architecture
class myModel(nn.Module):
    def __init__(self,input_dim, output_dim, hidden_size1 =128, hidden_size2 = 128):
        super(myModel, self).__init__()
        #define network layers
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, output_size)

    def forward(self,x):

        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        # no separate softmax, its implemented by CrossEntropyLoss method
        return out

model = myModel(input_size, output_size)
print(model)

#define loss function, optimizer and hyperparameters
loss_fn = nn.CrossEntropyLoss()
learning_rate = 0.0001
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0)
total_step = len(train_dataloader)

def train(dataloader, model, loss_fn, optimizer, batch_size, overall_train_loss):
    model.train()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, torch.max(y, 1)[1])

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        overall_train_loss += loss.item()

        #if batch % 2 == 0:
        #   loss, current = loss.item(), (batch+1) * batch_size
        #    print(f"Train_loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    print('overall_train_loss: '+str(overall_train_loss))
    return overall_train_loss

def test(dataloader, model, loss_fn, overall_test_loss):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, torch.max(y, 1)[1]).item()
    #test_loss /= num_batches
    print(f"overall_Test_loss: {test_loss:>8f} \n")
    return test_loss

epochs = 200
otrl = []
otel = []
for t in range(epochs):
    overall_train_loss = 0
    overall_test_loss = 0
    print(f"Epoch {t+1}\n-------------------------------")
    otrl.append(train(train_dataloader, model, loss_fn, optimizer, batch_size, overall_train_loss))
    otel.append(test(test_dataloader, model, loss_fn, overall_test_loss))

xc = np.arange(len(otrl))
plt.plot(xc, otrl, '-b', label = 'Overall train loss')
plt.plot(xc, otel, '-g', label = 'overall test loss')
plt.legend()
plt.show()
