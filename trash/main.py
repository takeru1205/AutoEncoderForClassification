"""
Train Script of Combined Model AE and Classifier
These are trained at the same time
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import torch.optim as optim
import matplotlib.pyplot as plt

from load_data import ImbalancedCIFAR10
from model import Combine
from utils import GridMask, AddNoise, imshow, imsave, ImbalancedDatasetSampler, calc_img_stats

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform_mean = [0.4920, 0.4825, 0.4500]
transform_std = [0.2039, 0.2009, 0.2026]

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", default=10, type=int, help='Epoch of train Classification')
parser.add_argument("--seed", default=42, type=int)
args = parser.parse_args()

# Set Seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=20),
            transforms.RandomResizedCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(transform_mean, transform_std)
        ])


evaluate_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(transform_mean, transform_std)
        ])



# Load Train Data
train_imbalance_class_ratio = np.array([1., 1., .5, 1., .5, 1., 1., 1., 1., .5])
# train_imbalance_class_ratio = np.array([1.] * 10)
train_set = ImbalancedCIFAR10(train_imbalance_class_ratio, transform=train_transform)
train_loader = DataLoader(train_set, sampler=ImbalancedDatasetSampler(train_set),
                            batch_size=64, shuffle=False, num_workers=4)
# train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, num_workers=4)

# Load Test Data
test_set = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=evaluate_transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=False, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = Combine()
net = net.to(device)

criterion = nn.CrossEntropyLoss()
ae_criterion = nn.MSELoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=0.01)

# Train Loop
for epoch in range(args.epoch):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        optimizer.zero_grad()
        decoded, predicted = net(inputs)

        # Cals Loss for Predicted
        classifier_loss = criterion(predicted, labels.to(device))

        # Calc Loss for AutoEncoder
        ae_loss = ae_criterion(decoded, inputs)
        # loss = criterion(outputs, labels.to(device))

        loss = classifier_loss + ae_loss

        loss.backward()
        optimizer.step()
        train_loss = loss.item()
        running_loss += loss.item()

        if i % 100 == 99:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0


def imshow(img):
    # ??????????????????
    img = img / 2 + 0.5
    # torch.Tensor?????????numpy.ndarray??????????????????
    print(type(img)) # <class 'torch.Tensor'>
    npimg = img.numpy()
    print(type(npimg))
    # ????????????RGB????????????????????????????????????RGB??????????????????
    print(npimg.shape)
    npimg = np.transpose(npimg, (1, 2, 0))
    print(npimg.shape)
    # ?????????????????????
    plt.imshow(npimg)
    plt.show()


# Evaluate Loop
dataiter = iter(test_loader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images))

print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

_, outputs = net(images.to(device))
_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

correct = 0
total = 0

target_list, predicted_list = [], []

# ????????????????????????????????????????????????????????????
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        target_list += labels.tolist()
        _, outputs = net(images.to(device))
        _, predicted = torch.max(outputs.cpu().detach(), 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        predicted_list += predicted.to('cpu').detach().numpy().copy().tolist()
print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

print(classification_report(target_list, predicted_list))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        _, outputs = net(images.to(device))
        _, predicted = torch.max(outputs, 1)
        c = (predicted.to(device) == labels.to(device)).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))


