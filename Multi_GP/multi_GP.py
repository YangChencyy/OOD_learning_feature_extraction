import numpy as np
import torch
# import torchvision
# from torchvision import datasets, transforms
from torch import nn, optim
from torch.utils.data import Dataset
import torch.nn.functional as F
from scipy import stats
import math
from sklearn.metrics import accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# from dataset import *

# Model structure for MNIST dataset
class MNIST_Net(nn.Module):
    def __init__(self, out_size = 18):
        super(MNIST_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 160)
        self.fc2 = nn.Linear(160, 10)
        self.fc3 = nn.Linear(160, out_size)
        # self.fc2 = nn.Linear(320, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        out = self.fc3(x)
        x = self.fc2(x)
        return out, F.log_softmax(x, dim = 1)
    
    def feature_list(self, x):
        out_list = []
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        out_list.append(x)
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        out_list.append(x)
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        out_list.append(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x) 
        out_list.append(x)   
        return F.log_softmax(x, dim = 1), out_list
    
     
    def intermediate_forward(self, x, layer_index):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        if layer_index == 1:
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        elif layer_index == 2:
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            x = x.view(-1, 320)
            x = F.relu(self.fc1(x))
        elif layer_index == 3:
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            x = x.view(-1, 320)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)    
        return x

class Fashion_MNIST_Net(nn.Module):
    
    def __init__(self, out_size = 18):
        super(Fashion_MNIST_Net, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc1 = nn.Linear(in_features=64*6*6, out_features=600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)
        self.fc4 = nn.Linear(in_features=120, out_features=out_size)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        f = self.fc4(out)
        out = self.fc3(out)

        return f, out # F.log_softmax(out, dim = 1)
        
    def feature_list(self, x):
        out_list = []
        out = self.layer1(x)
        out_list.append(out)
        out = self.layer2(out)
        out_list.append(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out_list.append(out)
        out = self.fc2(out)
        out_list.append(out)
        # f = self.fc3(out)
        out = self.fc4(out)

        return F.log_softmax(out, dim = 1), out_list
    
    def intermediate_forward(self, x, layer_index):
        out = self.layer1(x)
        if layer_index == 1:
            out = self.layer2(out)
        elif layer_index == 2:
            out = self.layer2(out)
            out = out.view(out.size(0), -1)
            out = self.fc1(out)
            out = self.drop(out)
        elif layer_index == 3:
            out = self.layer2(out)
            out = out.view(out.size(0), -1)
            out = self.fc1(out)
            out = self.drop(out)
            out = self.fc2(out)

        return out

# parameter refers to k 
def train(network, trloader, epochs, learning_rate = 0.01, momentum = 0.5, verbal = False):
    # optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
    optimizer = optim.Adam(network.parameters(), lr=0.001)  
    error = nn.CrossEntropyLoss()


    network.to(device)
    network.train()

    for epoch in range(1, epochs + 1):
         
        for batch_idx, (data, target) in enumerate(trloader):
            data = data.to(device)
            target = target.to(device)
            data.requires_grad_(True)
        
            optimizer.zero_grad()
            _, output = network(data)

            # loss = F.nll_loss(output, target)
            loss = error(output, target)
            loss.backward()

            optimizer.step()

            if verbal and batch_idx % 400 == 0:
                    print("epoch: ", epoch, ", batch: ", batch_idx, ", loss:", loss.item())

def cifar10_train(network, trloader, epochs, optim=None, learning_rate = 0.01, momentum = 0.5, verbal = False):
    if optim is None:
        optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    elif optim == 'SGD':
        optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)



    network.to(device)
    network.train()

    for epoch in range(1, epochs + 1):
         
        for batch_idx, (data, target) in enumerate(trloader):
            data = data.to(device)
            target = target.to(device)
            data.requires_grad_(True)
        
            optimizer.zero_grad()
            _, output = network(data)

            # # for mnist/fmnist
            # loss = F.nll_loss(output, target)
            # cifar
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, target)
            loss.backward()

            optimizer.step()

            if verbal and batch_idx % 400 == 0:
                    print("epoch: ", epoch, ", batch: ", batch_idx, ", loss:", loss.item())

        scheduler.step()
    

def scores(network, tsloader):
    network.eval()
    outputs, outputs_16, test_losses = [], [], []
    test_loss, correct = 0, 0

    with torch.no_grad():
        for data, target in tsloader:
            data, target = data.to(device), target.to(device)
            output16, output = network(data)
            outputs.append(output)
            outputs_16.append(output16)  # [B, 16, 1, 1]
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(tsloader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(tsloader.dataset),
        100. * correct / len(tsloader.dataset)))
    acc = correct / len(tsloader.dataset)

    outputs_16 = torch.cat(outputs_16, 0)
    outputs = torch.cat(outputs, 0)
    return outputs_16, outputs, acc.item()


def scoresOOD(network, oodloader):
    network.eval()
    outputs_16 = []
    outputs = []
    with torch.no_grad():
        for data, _ in oodloader:
            data = data.to(device)
            output16, output = network(data)
            outputs_16.append(output16)  # [50, 128, 1, 1]
            outputs.append(output)
    outputs_16 = torch.cat(outputs_16, 0)
    outputs = torch.cat(outputs, 0)
    return outputs_16, outputs



def main():
    print("Hello World")

if __name__=="__main__":
    main()