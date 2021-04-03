"""
Train and Evaluation Script of Combined Model AutoEncoder and Classifier.
These are trained separatly.
"""
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from load_data import ImbalancedCIFAR10
from model import CAE


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose(
        [transforms.ToTensor(),]
        )

# Load Train Data
train_imbalance_class_ratio = np.array([1., 1., .5, 1., .5, 1., 1., 1., 1., .5])
# train_imbalance_class_ratio = np.array([1.] * 10)
train_set = ImbalancedCIFAR10(train_imbalance_class_ratio, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=8, shuffle=True, num_workers=4)

# Load Test Data
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=False, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = CAE()
net = net.to(device)

criterion = nn.CrossEntropyLoss()
# ae_criterion = nn.MSELoss()
ae_criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters())

# Train Auto Encoder Part
print('Start Auto Encoder Training')
net.classifier.requires_grad = False
for epoch in range(3):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, _ = data
        inputs = inputs.to(device)
        decoded = net(inputs)
        ae_loss = ae_criterion(decoded, inputs)
        optimizer.zero_grad()
        ae_loss.backward()
        optimizer.step()
        running_loss += ae_loss.item()

        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0


# Train Classification Part
print('Start Classification Training')
net.classifier.requires_grad = True
net.encoder.requires_grad = False
net.decoder.requires_grad = False
for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        predicted = net.classify(inputs)
        loss = criterion(predicted, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0


def imshow(img):
    # 非正規化する
    img = img / 2 + 0.5
    # torch.Tensor型からnumpy.ndarray型に変換する
    print(type(img)) # <class 'torch.Tensor'>
    npimg = img.numpy()
    print(type(npimg))
    # 形状を（RGB、縦、横）から（縦、横、RGB）に変換する
    print(npimg.shape)
    npimg = np.transpose(npimg, (1, 2, 0))
    print(npimg.shape)
    # 画像を表示する
    plt.imshow(npimg)
    plt.show()


# Evaluate Loop
dataiter = iter(test_loader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images))

print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

outputs = net.classify(images.to(device))
_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

correct = 0
total = 0
# 勾配を記憶せず（学習せずに）に計算を行う
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net.classify(images.to(device))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.to(device) == labels.to(device)).sum().item()
print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net.classify(images.to(device))
        _, predicted = torch.max(outputs, 1)
        c = (predicted.to(device) == labels.to(device)).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))



