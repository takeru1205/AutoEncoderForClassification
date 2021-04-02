"""
Train Script of Auto Encoder Model
"""
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import AutoEncoder, CAE
from load_data import ImbalancedCIFAR10

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

#train_imbalance_class_ratio = np.array([1., 1., .5, 1., .5, 1., 1., 1., 1., .5])
train_imbalance_class_ratio = np.array([1.] * 10)
train_imbalanced_dataset = ImbalancedCIFAR10(train_imbalance_class_ratio)
train_imbalanced_loader = DataLoader(train_imbalanced_dataset, batch_size=4, shuffle=True, num_workers=4)

# net = AutoEncoder()
net = CAE()
net = net.to(device)

criterion = nn.BCELoss()
# criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters())

# Train Model
for epoch in range(3):
    running_loss = 0.0
    
    for i, (inputs, _) in enumerate(train_imbalanced_loader, 0):
        inputs = inputs.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, inputs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 200 == 199:
            print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0

# Save Model
torch.save(net.state_dict(), 'model_weights/auto_encoder')

