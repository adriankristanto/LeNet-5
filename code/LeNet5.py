# reference: https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.Sequential(
        # CONV1
        # input size: 32x32x1
        # output size: 28x28x6
        # filter size: 5x5
        nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5,5)),
        # POOL1
        # input size: 28x28x6
        # output size: 14x14x6
        # filter size: 2x2
        # stride: 2 horizontal steps, 2 vertical steps
        nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
        # CONV2
        # input size: 14x14x6
        # output size: 10x10x16
        # filter size: 5x5
        nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5,5)),
        # POOL2
        # input size: 10x10x16
        # output size: 5x5x16
        # filter size: 2x2
        # stride: 2 horizontal steps, 2 vertical steps
        nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
        # FC3
        # input size: 5x5x16
        # output size: 120
        nn.Linear(in_features=5*5*16, out_features=120),
        # FC4
        # input size: 120
        # output size: 84
        nn.Linear(in_features=120, out_features=84),
        # FC5
        # input size: 84
        # output size: 10 (digit 0-9)
        nn.Linear(in_features=84, out_features=10)
        )
        
    
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
        # x = x.view(x.size(0), -1)
        # another way of doing this is to use flatten
        # note that we use start_dim=1 because dim 0 is the batch dimension
        # if we don't use start_dim=1, we will get torch.Size([400]), which doesn't have the batch dim
        # this will cause errors for the subsequent layers
        x = torch.flatten(x, start_dim=1)
        # pass through the third layer
        x = self.fc3(x)
        x = F.relu(x)
        # pass through the fourth layer
        x = self.fc4(x)
        x = F.relu(x)
        # pass through the fifth layer
        x = self.fc5(x)
        # reference: https://towardsdatascience.com/understanding-dimensions-in-pytorch-6edf9972d3be
        # since the input to softmax is of shape (batch_size,84)
        # we only want to perform softmax on 84, thus, we ignore the batch_size, i.e. dim=1
        x = F.softmax(x, dim=1)
        return x


if __name__ == "__main__":
    net = Net()
    print(net)

    # 1 batch, 1 channel, 32x32 input size
    # x = torch.randn([1,1,32,32])
    # print(net(x))