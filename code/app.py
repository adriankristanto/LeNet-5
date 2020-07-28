import torch
import torchvision
import torchvision.transforms as transforms
import LeNet5 
import tqdm
import os
import numpy as np
import PIL

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the test data
transform = transforms.Compose(
    [transforms.Grayscale(),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))]
)

def predict(image_path):
    # convert the image to a greyscale image
    image = PIL.Image.open(image_path)
    # resize the image, transform the image to tensor, and normalise it
    img_tensor = transform(image).unsqueeze_(0)
    # print(img_tensor.shape)
    # forward propagation
    output = net(img_tensor)
    # return prediction
    # print(output)
    return torch.argmax(output, dim=1).item()


if __name__ == "__main__":
    # load the model
    FILEPATH = os.path.dirname(os.path.realpath(__file__)) + '/../model/'
    FILENAME = 'lenet5-0.pth'
    net = LeNet5.Net()
    # reference: https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-across-devices
    net.load_state_dict(torch.load(FILEPATH + FILENAME,  map_location=device))

    # predict image
    IMAGE_PATH = os.path.dirname(os.path.realpath(__file__)) + '/../sample/'
    IMAGE_NAME = 'sample0.png'
    print(f'prediction: {predict(IMAGE_PATH + IMAGE_NAME)}')