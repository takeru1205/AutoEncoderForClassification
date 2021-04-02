import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
                nn.Conv2d(3, 8, 5, stride=2, padding=2),
                nn.PReLU(),
                nn.Conv2d(8, 8, 5, stride=2, padding=2),
                nn.PReLU(),
                )
        self.decoder = nn.Sequential(
                nn.ConvTranspose2d(8, 8, 6, stride=2, padding=2),
                nn.PReLU(),
                nn.ConvTranspose2d(8, 3, 6, stride=2, padding=2),
                nn.Sigmoid(),
                )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Combine(nn.Module):
    def __init__(self):
        super(Combine, self).__init__()
        self.encoder = nn.Sequential(
                nn.Conv2d(3, 8, 5, stride=2, padding=2),
                nn.PReLU(),
                nn.Conv2d(8, 8, 5, stride=2, padding=2),
                nn.PReLU(),
                )

        self.decoder = nn.Sequential(
                nn.ConvTranspose2d(8, 8, 6, stride=2, padding=2),
                nn.PReLU(),
                nn.ConvTranspose2d(8, 3, 6, stride=2, padding=2),
                nn.Sigmoid(),
                )

        self.classifier = nn.Sequential(
                nn.Linear(8 * 8 * 8, 126),
                nn.ReLU(),
                nn.Linear(126, 64),
                nn.ReLU(),
                nn.Linear(64, 10),
                )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        predicted = self.classifier(encoded.view(-1, 8 * 8 * 8))
        return decoded, predicted
                


"""
ae = AutoEncoder()
ae.encoder.load_state_dict(torch.load('model_weights/auto_encoder'))
"""
if __name__ == '__main__':
    ae = AutoEncoder()
    ae.encoder.load_state_dict(torch.load('model_weights/auto_encoder'))
    print(ae)

