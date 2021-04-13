"""
Train and Evaluation Script of Combined Model AutoEncoder and Classifier.
These are trained separatly.
"""
import argparse
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
from model import CAE3, CAE4
from utils import GridMask, AddNoise, imshow, imsave, ImbalancedDatasetSampler

parser = argparse.ArgumentParser()
parser.add_argument("--train-epoch", default=50, type=int, help='Epoch of train Classification')
parser.add_argument("--ae-epoch", default=100, type=int, help='Epoch of train Auto Encoder')
parser.add_argument("--load", action='store_true', help='To load model weight')
args = parser.parse_args()

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

transform_mean = (0.4735, 0.4640, 0.4295)
transform_std = (0.2200, 0.2165, 0.2151)

ae_transform = transforms.Compose(
        [
            # transforms.Grayscale(),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            # transforms.RandomAffine(degrees=[-10, 10], translate=(0.1, 0.1), scale=(0.5, 1.5)),
            transforms.RandomRotation(20),
            transforms.RandomResizedCrop(32),
            # GridMask(p=0.3),
            # AddNoise(),
            transforms.ToTensor(),
            transforms.Normalize(transform_mean, transform_std)
        ])


train_transform = transforms.Compose(
        [
            # transforms.Grayscale(),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            # transforms.RandomAffine(degrees=[-10, 10], translate=(0.1, 0.1), scale=(0.5, 1.5)),
            transforms.RandomRotation(degrees=20),
            transforms.RandomResizedCrop(32),
            # GridMask(p=0.3),
            # AddNoise(p=0.6),
            transforms.ToTensor(),
            transforms.Normalize(transform_mean, transform_std)
        ])


ae_evaluate_transform = transforms.Compose(
        [
            # transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(transform_mean, transform_std)
        ])


evaluate_transform = transforms.Compose(
        [
            # transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(transform_mean, transform_std)
        ])


# Set Seed
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


# Log
writer = SummaryWriter(log_dir=f'sample/cae3-{args.ae_epoch}-{args.train_epoch}-convclassify-decdp-rndcrop-bn-avgpool-xnorm-bzero')

# Load Train Data
train_imbalance_class_ratio = np.array([1., 1., .5, 1., .5, 1., 1., 1., 1., .5])
# train_imbalance_class_ratio = np.array([1.] * 10)
# train_imbalance_class_ratio = np.array([.5] * 10)
train_set = ImbalancedCIFAR10(train_imbalance_class_ratio, transform=evaluate_transform)
#train_loader = torch.utils.data.DataLoader(train_set, batch_size=8, shuffle=True, num_workers=4)

# Cross Validation Dataset(Stratified K Fold)
train_indices, val_indices = train_test_split(list(range(len(train_set.labels))), test_size=0.2, stratify=train_set.labels)
train_dataset = Subset(train_set, train_indices)
train_dataset.dataset.transform = ae_transform
val_dataset = Subset(train_set, val_indices)
val_dataset.dataset.transform = ae_evaluate_transform

train_loader = DataLoader(train_dataset, sampler=ImbalancedDatasetSampler(train_dataset),
                          batch_size=64, shuffle=False, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=4)

print(f'Train Size: {len(train_loader.dataset)}')
print(f'Validation Size: {len(val_loader.dataset)}')

print(torch.tensor(train_dataset.dataset.labels[train_indices]).unique(return_counts=True))
print(torch.tensor(val_dataset.dataset.labels[val_indices]).unique(return_counts=True))

# Load Test Data
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=evaluate_transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=False, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


net = CAE3()
net = net.to(device)

if args.load:
    net.encoder.load_state_dict(torch.load('model_weights/Encoder3'))
    net.decoder.load_state_dict(torch.load('model_weights/Decoder3'))
net.train()

_, class_size = torch.tensor(train_dataset.dataset.labels[train_indices]).unique(return_counts=True)
weight = class_size.to(torch.float) / torch.max(class_size).to(torch.float)
criterion = nn.CrossEntropyLoss(weight=weight.to(device))
# criterion = nn.CrossEntropyLoss()
ae_criterion = nn.MSELoss()
# ae_criterion = nn.BCELoss()
# ae_optimizer = optim.Adam(net.parameters(), lr=0.0005, weight_decay=1e-5)
ae_optimizer = optim.AdamW(net.parameters(), lr=0.0005)
# ae_optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)
optimizer = optim.AdamW(net.parameters(), lr=0.0005, betas=(0.9, 0.999))
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Train Auto Encoder Part
print('Start Auto Encoder Training')
ae_iter = 0
for epoch in range(args.ae_epoch):
    if args.load:
        break
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

        if i % 500 == 499:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 500))
            writer.add_scalar('Loss/AutoEncoder', running_loss, ae_iter)
            running_loss = 0.0

# Test Auto Encoder Output
dataiter = iter(val_loader)
images, labels = dataiter.next()
# imsave(torchvision.utils.make_grid(images), 'target.png')
output = net(images.to(device))
# imsave(torchvision.utils.make_grid(output.cpu().data), 'predict.png')

# Train Classification Part
train_loader.dataset.transform = train_transform
val_loader.dataset.transform = evaluate_transform

if args.load:
    for param in net.encoder.parameters():
        param.requires_grad = False

train_iter, val_iter = 0, 0
for epoch in range(args.train_epoch):
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

        if i % 500 == 499:
            print('[%d, %5d] Train Loss: %.3f' % (epoch + 1, i + 1, running_loss / 500))
            writer.add_scalar('Loss/Train', running_loss, train_iter)
            running_loss = 0.0

    # Validation
    total, correct = 0, 0
    val_loss = 0.0
    for i, data in enumerate(val_loader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        predicted = net.classify(inputs)
        loss = criterion(predicted, labels)

        val_loss += loss.item()

        _, predicted = torch.max(predicted.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        val_iter += 1
        if val_iter % 500 == 499:
            writer.add_scalar('Validation/Loss', loss.item(), val_iter)
            print('[%d, %5d] Valid Loss: %.3f' % (epoch + 1, i + 1, running_loss / 500))
            val_loss = 0.0
        
    writer.add_scalar('Validation/Score', (100 * correct / total), epoch)

    print(f'Epoch: {epoch}, Validation Score: %d %%' % (100 * correct / total))

    # Save Model
    if (epoch+1) % 10 == 0: 
        torch.save(net.state_dict(), f'model_weights/CAE-{epoch}')

# Save Mdels
net.save_model()

# Evaluate Loop
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



