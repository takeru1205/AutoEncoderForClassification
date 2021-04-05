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
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.model_selection import train_test_split

from load_data import ImbalancedCIFAR10
from model import CAE, CAE2
from utils import GridMask, AddNoise, imshow, imsave, ImbalancedDatasetSampler


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.6),
            transforms.RandomVerticalFlip(p=0.6),
            # transforms.RandomAffine(degrees=[-10, 10], translate=(0.1, 0.1), scale=(0.5, 1.5)),
            # transforms.RandomRotation(degrees=10),
            GridMask(p=0.6),
            AddNoise(p=0.6),
            transforms.ToTensor(),
            # transforms.Normalize((0.4915, 0.4823, 0.4468),
            #                      (0.2470, 0.2435, 0.2616))
        ])

evaluate_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize((0.4915, 0.4823, 0.4468),
            #              (0.2470, 0.2435, 0.2616))
        ])


# Set Seed
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

ae_epoch = 100
train_epoch = 30

# Log
writer = SummaryWriter(log_dir=f'cae/cae2-{ae_epoch}-{train_epoch}-noise-TrainOptim-0001-update-fcdp')

# Load Train Data
train_imbalance_class_ratio = np.array([1., 1., .5, 1., .5, 1., 1., 1., 1., .5])
# train_imbalance_class_ratio = np.array([.5] * 10)
train_set = ImbalancedCIFAR10(train_imbalance_class_ratio, transform=evaluate_transform)
#train_loader = torch.utils.data.DataLoader(train_set, batch_size=8, shuffle=True, num_workers=4)

# Cross Validation Dataset(Stratified K Fold)
train_indices, val_indices = train_test_split(list(range(len(train_set.labels))), test_size=0.2, stratify=train_set.labels)
train_dataset = Subset(train_set, train_indices)
train_dataset.dataset.transform = train_transform
val_dataset = Subset(train_set, val_indices)
val_dataset.dataset.transform = evaluate_transform

train_loader = DataLoader(train_dataset, sampler=ImbalancedDatasetSampler(train_dataset),
                          batch_size=16, shuffle=False, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True, num_workers=4)

print(f'Train Size: {len(train_loader.dataset)}')
print(f'Validation Size: {len(val_loader.dataset)}')

print(torch.tensor(train_dataset.dataset.labels[train_indices]).unique(return_counts=True))
print(torch.tensor(val_dataset.dataset.labels[val_indices]).unique(return_counts=True))

# Load Test Data
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=evaluate_transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=False, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Data Visualize
# dataiter = iter(train_loader)
# images, labels = dataiter.next()
# imshow(torchvision.utils.make_grid(images))


net = CAE2()
net = net.to(device)
net.train()

criterion = nn.CrossEntropyLoss()
# ae_criterion = nn.MSELoss()
ae_criterion = nn.BCELoss()
ae_optimizer = optim.Adam(net.parameters())
optimizer = optim.Adam(net.parameters(), lr=0.0001)

# Train Auto Encoder Part
print('Start Auto Encoder Training')
# net.classifier.requires_grad = False
ae_iter = 0
for epoch in range(ae_epoch):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, _ = data
        inputs = inputs.to(device)
        decoded = net(inputs)
        ae_loss = ae_criterion(decoded, inputs)
        ae_optimizer.zero_grad()
        ae_loss.backward()
        ae_optimizer.step()
        running_loss += ae_loss.item()

        ae_iter += 1

        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            writer.add_scalar('Loss/AutoEncoder', running_loss, ae_iter)
            running_loss = 0.0

# Test Auto Encoder Output
dataiter = iter(val_loader)
images, labels = dataiter.next()
imsave(torchvision.utils.make_grid(images), 'target.png')
output = net(images.to(device))
imsave(torchvision.utils.make_grid(output.cpu().data), 'predict.png')

# Train Classification Part
print('Start Classification Training')
net.classifier.requires_grad = True
# net.encoder.requires_grad = False
# net.decoder.requires_grad = False
train_iter, val_iter = 0, 0
for epoch in range(train_epoch):
    running_loss = 0.0
    # Train
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

        train_iter += 1

        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            writer.add_scalar('Loss/Train', running_loss, train_iter)
            running_loss = 0.0
    # Validation
    total, correct = 0, 0
    for i, data in enumerate(val_loader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        predicted = net.classify(inputs)
        loss = criterion(predicted, labels)

        _, predicted = torch.max(predicted.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        val_iter += 1
        if val_iter % 10 == 0:
            writer.add_scalar('Validation/Loss', loss.item(), val_iter)
        
    writer.add_scalar('Validation/Score', (100 * correct / total), epoch)

    print(f'Epoch: {epoch}, Validation Score: %d %%' % (100 * correct / total))

    # Save Model
    if (epoch+1) % 10 == 0: 
        torch.save(net.state_dict(), f'model_weights/CAE-{epoch}')

# Evaluate Loop
'''
dataiter = iter(test_loader)
images, labels = dataiter.next()
# imshow(torchvision.utils.make_grid(images))

# print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

outputs = net.classify(images.to(device))
_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))
'''
net.eval()
correct = 0
total = 0
predicted_list = []
target_list = []
# 勾配を記憶せず（学習せずに）に計算を行う
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        target_list += labels.tolist()
        outputs = net.classify(images.to(device))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.to(device) == labels.to(device)).sum().item()
        predicted_list += predicted.to('cpu').detach().numpy().copy().tolist()
print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

# print(f'F1-Score: {f1_score(target_list, predicted_list)}')
# print(confusion_matrix(target_list, predicted_list))
print(classification_report(target_list, predicted_list))

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



