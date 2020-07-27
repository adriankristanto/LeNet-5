# reference: https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # CONV1
        # input size: 32x32x1
        # output size: 28x28x6
        # filter size: 5x5
        self.conv1 = nn.Conv2d(in_channels=1, out_channel=6, kernel_size=(5,5))
        # POOL1
        # input size: 28x28x6
        # output size: 14x14x6
        # filter size: 2x2
        # stride: 2 horizontal steps, 2 vertical steps
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        # CONV2
        # input size: 14x14x6
        # output size: 10x10x16
        # filter size: 5x5
        self.conv2 = nn.Conv2d(in_channels=6, out_channel=16, kernel_size=(5,5))
        # POOL2
        # input size: 10x10x16
        # output size: 5x5x16
        # filter size: 2x2
        # stride: 2 horizontal steps, 2 vertical steps
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        # FC3
        # input size: 5x5x16
        # output size: 120
        self.fc3 = nn.Linear(in_features=5*5*16, out_features=120)
        # FC4
        # input size: 120
        # output size: 84
        self.fc4 = nn.Linear(in_features=120, out_features=84)
        # FC5
        # input size: 84
        # output size: 10 (digit 0-9)
        self.fc5 = nn.Linear(in_features=84, out_features=10)
        
    
    def forward(self, x):
        # pass through the first layer
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        # pass through the second layer
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        # flatten the output of the second layer
        # where x.size(0) is the batch size
        # and -1 will be replaced with 5x5x16
        x = x.view(x.size(0), -1)
        # pass through the third layer
        x = self.fc3(x)
        x = F.relu(x)
        # pass through the fourth layer
        x = self.fc4(x)
        x = F.relu(x)
        # pass through the fifth layer
        x = self.fc5(x)
        # reference: https://towardsdatascience.com/understanding-dimensions-in-pytorch-6edf9972d3be
        x = F.softmax(x, dim=1)
        return x