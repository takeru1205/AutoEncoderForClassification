"""
Train and Evaluation Script of Combined Model AutoEncoder and Classifier.
These are trained separatly.
"""
import argparse, itertools
from copy import deepcopy
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

from load_data import ImbalancedCIFAR10, separate_data
from model import CAE3
from utils import GridMask, AddNoise, imshow, imsave, ImbalancedDatasetSampler, calc_img_stats

parser = argparse.ArgumentParser()
parser.add_argument("--train-epoch", default=50, type=int, help='Epoch of train Classification')
parser.add_argument("--ae-epoch", default=50, type=int, help='Epoch of train Auto Encoder')
parser.add_argument("--load", action='store_true', help='To load model weight')
parser.add_argument("--ae-lr", default=0.0005, type=float, help='Learing Rate of AutoEncoder')
#parser.add_argument("--ae1-lr", default=0.0005, type=float, help='Learing Rate of AutoEncoder1')
#parser.add_argument("--ae2-lr", default=0.0005, type=float, help='Learing Rate of AutoEncoder2')
parser.add_argument("--classify-lr", default=0.0005, type=float, help='Learing Rate of Classifier')
#parser.add_argument("--classify1-lr", default=0.0005, type=float, help='Learing Rate of Classifier')
#parser.add_argument("--classify1-lr", default=0.0005, type=float, help='Learing Rate of Classifier')
args = parser.parse_args()

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# get from calc_img_stats
transform_mean = [0.4920, 0.4825, 0.4500]
transform_std = [0.2039, 0.2009, 0.2026]

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


# Set Seed
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


# Log
writer = SummaryWriter(log_dir=f'logs/ensemble-{args.ae_epoch}-{args.train_epoch}-{args.ae_lr}-{args.classify_lr}')

# Load Train Data
train_imbalance_class_ratio = np.array([1., 1., .5, 1., .5, 1., 1., 1., 1., .5])
# train_imbalance_class_ratio = np.array([1.] * 10)
# train_imbalance_class_ratio = np.array([.5] * 10)
train_set = ImbalancedCIFAR10(train_imbalance_class_ratio, transform=evaluate_transform)

# Cross Validation Dataset(Stratified K Fold)
train_indices, val_indices = train_test_split(list(range(len(train_set.labels))), test_size=0.2, stratify=train_set.labels)
train_dataset = Subset(train_set, train_indices)
train_dataset.dataset.transform = train_transform
val_dataset = Subset(train_set, val_indices)
val_dataset.dataset.transform = evaluate_transform

train1_indices, train2_indices = separate_data(train_dataset, train_indices)

train1_dataset = deepcopy(train_dataset)
train2_dataset = deepcopy(train_dataset)

train1_dataset.indices = train1_indices
train2_dataset.indices = train2_indices

train_loader = DataLoader(train_dataset, sampler=ImbalancedDatasetSampler(train_dataset),
                            batch_size=128, shuffle=False, num_workers=4)
train1_loader = DataLoader(train1_dataset, batch_size=64, shuffle=True, num_workers=4)
train2_loader = DataLoader(train2_dataset, batch_size=64, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=4)

#print(f'Train Size: {len(train_loader.dataset)}')
#print(f'Train1 Size: {len(train1_loader.dataset)}')
#print(f'Train2 Size: {len(train2_loader.dataset)}')
#print(f'Validation Size: {len(val_loader.dataset)}')

#print(torch.tensor(train_dataset.dataset.labels[train_indices]).unique(return_counts=True))
#print(torch.tensor(train1_dataset.dataset.labels[train1_indices]).unique(return_counts=True))
#print(torch.tensor(train2_dataset.dataset.labels[train2_indices]).unique(return_counts=True))
#print(torch.tensor(val_dataset.dataset.labels[val_indices]).unique(return_counts=True))

# Load Test Data
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=evaluate_transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=False, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


net = CAE3()
net1 = CAE3()
net2 = CAE3()
net = net.to(device)
net1 = net1.to(device)
net2 = net2.to(device)

if args.load:
    net1.encoder.load_state_dict(torch.load('model_weights/Encoder3'))
    net2.decoder.load_state_dict(torch.load('model_weights/Decoder3'))
net.train()
net1.train()
net2.train()

criterion = nn.CrossEntropyLoss()
ae_criterion = nn.MSELoss()
ae_optimizer = optim.AdamW(itertools.chain(net.encoder.parameters(), net.decoder.parameters()), lr=args.ae_lr, betas=(0.9, 0.999))
ae1_optimizer = optim.AdamW(itertools.chain(net1.encoder.parameters(), net1.decoder.parameters()), lr=args.ae_lr, betas=(0.9, 0.999))
ae2_optimizer = optim.AdamW(itertools.chain(net2.encoder.parameters(), net2.decoder.parameters()), lr=args.ae_lr, betas=(0.9, 0.999))
optimizer = optim.AdamW(itertools.chain(net.encoder.parameters(), net.classifier.parameters()), lr=args.classify_lr, betas=(0.9, 0.999))
optimizer1 = optim.AdamW(itertools.chain(net1.encoder.parameters(), net1.classifier.parameters()), lr=args.classify_lr, betas=(0.9, 0.999))
optimizer2 = optim.AdamW(itertools.chain(net2.encoder.parameters(), net2.classifier.parameters()), lr=args.classify_lr, betas=(0.9, 0.999))

# Train Auto Encoder Part

#print('Start Net Auto Encoder Training')
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

        if i % 100 == 99:
            #print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            writer.add_scalar('Loss/AutoEncoder', running_loss, ae_iter)
            running_loss = 0.0


#print('Start Net1 Auto Encoder Training')
ae_iter = 0
for epoch in range(args.ae_epoch):
    if args.load:
        break
    running_loss = 0.0
    for i, data in enumerate(train1_loader, 0):
        inputs, _ = data
        inputs = inputs.to(device)
        decoded = net1(inputs)
        ae_loss = ae_criterion(decoded, inputs)
        ae1_optimizer.zero_grad()
        ae_loss.backward()
        ae1_optimizer.step()
        running_loss += ae_loss.item()

        ae_iter += 1

        if i % 100 == 99:
            #print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            writer.add_scalar('Loss/AutoEncoder1', running_loss, ae_iter)
            running_loss = 0.0

#print('Start Net2 Auto Encoder Training')
ae_iter = 0
for epoch in range(args.ae_epoch):
    if args.load:
        break
    running_loss = 0.0
    for i, data in enumerate(train2_loader, 0):
        inputs, _ = data
        inputs = inputs.to(device)
        decoded = net2(inputs)
        ae_loss = ae_criterion(decoded, inputs)
        ae2_optimizer.zero_grad()
        ae_loss.backward()
        ae2_optimizer.step()
        running_loss += ae_loss.item()

        ae_iter += 1

        if i % 100 == 99:
            #print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            writer.add_scalar('Loss/AutoEncoder2', running_loss, ae_iter)
            running_loss = 0.0


#print('Start Net Classification Training')
if args.load:
    for param in net1.encoder.parameters():
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

        if i % 100 == 99:
            #print('[%d, %5d] Train Loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
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
        if val_iter % 100 == 99:
            writer.add_scalar('Validation/Loss', loss.item(), val_iter)
            #print('[%d, %5d] Valid Loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            val_loss = 0.0
        
    writer.add_scalar('Validation/Score', (100 * correct / total), epoch)

    #print(f'Epoch: {epoch}, Validation Score: %d %%' % (100 * correct / total))


#print('Start Net1 Classification Training')
if args.load:
    for param in net1.encoder.parameters():
        param.requires_grad = False

train_iter, val_iter = 0, 0
for epoch in range(args.train_epoch):
    running_loss = 0.0
    # Train
    for i, data in enumerate(train1_loader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        predicted = net1.classify(inputs)
        loss = criterion(predicted, labels)

        optimizer1.zero_grad()
        loss.backward()
        optimizer1.step()

        running_loss += loss.item()

        train_iter += 1

        if i % 100 == 99:
            #print('[%d, %5d] Train Loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            writer.add_scalar('Loss/Train1', running_loss, train_iter)
            running_loss = 0.0

    # Validation
    total, correct = 0, 0
    val_loss = 0.0
    for i, data in enumerate(val_loader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        predicted = net1.classify(inputs)
        loss = criterion(predicted, labels)

        val_loss += loss.item()

        _, predicted = torch.max(predicted.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        val_iter += 1
        if val_iter % 100 == 99:
            writer.add_scalar('Validation/Loss1', loss.item(), val_iter)
            #print('[%d, %5d] Valid Loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            val_loss = 0.0
        
    writer.add_scalar('Validation/Score1', (100 * correct / total), epoch)

    #print(f'Epoch: {epoch}, Validation Score: %d %%' % (100 * correct / total))


#print('Start Net2 Classification Training')
if args.load:
    for param in net1.encoder.parameters():
        param.requires_grad = False

train_iter, val_iter = 0, 0
for epoch in range(args.train_epoch):
    running_loss = 0.0
    # Train
    for i, data in enumerate(train2_loader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        predicted = net2.classify(inputs)
        loss = criterion(predicted, labels)

        optimizer2.zero_grad()
        loss.backward()
        optimizer2.step()

        running_loss += loss.item()

        train_iter += 1

        if i % 100 == 99:
            #print('[%d, %5d] Train Loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            writer.add_scalar('Loss/Train2', running_loss, train_iter)
            running_loss = 0.0

    # Validation
    total, correct = 0, 0
    val_loss = 0.0
    for i, data in enumerate(val_loader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        predicted = net2.classify(inputs)
        loss = criterion(predicted, labels)

        val_loss += loss.item()

        _, predicted = torch.max(predicted.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        val_iter += 1
        if val_iter % 100 == 99:
            writer.add_scalar('Validation/Loss2', loss.item(), val_iter)
            #print('[%d, %5d] Valid Loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            val_loss = 0.0
        
    writer.add_scalar('Validation/Score2', (100 * correct / total), epoch)

    #print(f'Epoch: {epoch}, Validation Score: %d %%' % (100 * correct / total))

 
# Save Mdels
net.save_model(name='0')
net1.save_model(name='1')
net2.save_model(name='2')

# Evaluate Loop
net.eval()
net1.eval()
net2.eval()
correct = 0
total = 0
predicted_list = []
target_list = []

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        target_list += labels.tolist()
        all_outputs = net.classify(images.to(device))
        outputs1 = net1.classify(images.to(device))
        outputs2 = net2.classify(images.to(device))
        outputs = (all_outputs + outputs1 + outputs2) / 3
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
        all_outputs = net.classify(images.to(device))
        outputs1 = net1.classify(images.to(device))
        outputs2 = net2.classify(images.to(device))
        outputs = (all_outputs + outputs1 + outputs2) / 2
        _, predicted = torch.max(outputs, 1)
        c = (predicted.to(device) == labels.to(device)).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))



