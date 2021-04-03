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


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 12, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(12, 24, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(24, 48, 4, stride=2, padding=1)

        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1)

        
    def forward(self, x):
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = self.deconv3(x)
        x = torch.sigmoid(x)
        return x

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(48 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 48 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        

class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.classifier = Classifier()
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def classify(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
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

    def autoencode(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)

    def predict(self, x):
        encoded = self.encoder(x)
        return self.classifier(encoded.view(-1, 8 * 8 * 8))
                


"""
ae = AutoEncoder()
ae.encoder.load_state_dict(torch.load('model_weights/auto_encoder'))
"""
if __name__ == '__main__':
    ae = AutoEncoder()
    ae.encoder.load_state_dict(torch.load('model_weights/auto_encoder'))
    print(ae)

