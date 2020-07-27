import torch 
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt 
import numpy as np
import os
import LeNet5

# to allow for training in GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# load the MNIST handwriting data
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