import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from model import AutoEncoder
from load_data import ImbalancedCIFAR10

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

train_imbalance_class_ratio = np.array([1., 1., .5, 1., .5, 1., 1., 1., 1., .5])
train_imbalanced_dataset = ImbalancedCIFAR10(train_imbalance_class_ratio)
train_imbalanced_loader = DataLoader(train_imbalanced_dataset, batch_size=4, shuffle=True, num_workers=4)

# Load Model
net = AutoEncoder()
net.load_state_dict(torch.load('model_weights/auto_encoder'))
net = net.to(device)

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Test Model
classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
dataiter = iter(train_imbalanced_loader)
images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images))
outputs = net(images.to(device))
imshow(torchvision.utils.make_grid(outputs.cpu().data))

