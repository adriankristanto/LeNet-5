import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt 
import numpy as np
import os
import tqdm
import LeNet5

# to allow for training in GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 1. load the MNIST handwriting data
# store the data in the data directory located at the parent directory of the code directory
DATA_PATH = os.path.dirname(os.path.realpath(__file__)) + '/../data'
# reference: https://discuss.pytorch.org/t/generic-question-about-batch-sizes/1321/3
BATCH_SIZE = 256
NUM_WORKERS = 8

# reference: https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457
# the original image size of MNIST handwriting data is (28, 28)
# however, the model expects (32,32)
transform = transforms.Compose(
    [transforms.Resize((32,32)),
    transforms.ToTensor()]
)

trainset = torchvision.datasets.MNIST(root=DATA_PATH, train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

testset = torchvision.datasets.MNIST(root=DATA_PATH, train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# code to show the image:
# dataiter = iter(trainloader)
# images, labels = dataiter.next()
# npimg = images[0].numpy().reshape((28,28))
# print(npimg.shape)
# plt.imshow(npimg)
# plt.show()

# 2. instantiate the network model
net = LeNet5.Net()
net.to(device)

# 3. define the loss function
criterion = nn.CrossEntropyLoss()

# 4. define the optimizer
LEARNING_RATE=0.001
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

# 5. train the model
EPOCH = 10
for epoch in range(EPOCH):
    epoch_loss = 0.0
    for data in tqdm(trainloader):
        inputs, labels = data[0].to(device), data[1].to(device)
        # 5a. zero the gradients
        optimizer.zero_grad()
        # 5b. forward propagation
        outputs = net(inputs)
        # 5c. compute loss
        loss = criterion(outputs, labels)
        # 5d. backward propagation
        loss.backward()
        # 5e. update parameters
        optimizer.step()

        epoch_loss += loss.item()
    print(f'Loss: {epoch_loss / len(trainloader)}')

# 6. save the trained model
MODEL_PATH = os.path.dirname(os.path.realpath(__file__)) + '/../model/lenet5.pth'
torch.save(net.state_dict(), MODEL_PATH)

# 7. test the network on the test data
correct = 0
total = 0

with torch.no_grad():
    for data in tqdm(testloader):
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        correct += (torch.argmax(outputs, 1) == labels).sum().item()
        total += len(labels)

print(f'Accuracy: {correct / total * 100}')