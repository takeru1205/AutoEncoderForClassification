import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder2(nn.Module):
    def __init__(self):
        super(Encoder2, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 8, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(8, 8, 3, stride=1, padding=1)

        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.pool2 = nn.MaxPool2d(2, stride=2)

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(8)
        self.bn3 = nn.BatchNorm2d(8)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        # x = F.elu(x)
        x = F.leaky_relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        # x = F.elu(x)
        x = F.leaky_relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        # x = F.elu(x)
        x = F.leaky_relu(x)

        return x

class Decoder2(nn.Module):
    def __init__(self):
        super(Decoder2, self).__init__()
        self.deconv3 = nn.ConvTranspose2d(8, 8, 3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(8, 16, 3, stride=1, padding=1)
        self.deconv1 = nn.ConvTranspose2d(16, 3, 3, stride=1, padding=1)
        
        # self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample2 = nn.Upsample(scale_factor=2)
        # self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample1 = nn.Upsample(scale_factor=2)
        
        self.bn3 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        
    def forward(self, x):
        x = self.deconv3(x)
        x = self.bn3(x)
        # x = F.elu(x)
        x = F.leaky_relu(x)
        
        x = self.upsample2(x)
        
        x = self.deconv2(x)
        x = self.bn2(x)
        # x = F.elu(x)
        x = F.leaky_relu(x)
        
        x = self.upsample1(x)
        
        x = self.deconv1(x)
        x = torch.sigmoid(x)
        
        return x

class Classifier2(nn.Module):
    def __init__(self):
        super(Classifier2, self).__init__()
        # self.fc1 = nn.Linear(512, 256)
        # self.fc2 = nn.Linear(256, 128)
        # self.fc3 = nn.Linear(128, 64)
        # self.fc4 = nn.Linear(64, 10)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

        self.dp1 = nn.Dropout(p=0.2)
        self.dp2 = nn.Dropout(p=0.2)

        self.weights_init()

    def forward(self, x):
        x = x.view(-1, 512)
        x = self.fc1(x)
        x = F.elu(x)
        x = self.dp1(x)

        x = self.fc2(x)
        x = F.elu(x)
        x = self.dp2(x)

        x = self.fc3(x)
        # x = F.elu(x)

        # x = self.fc4(x)
        return x

    def weights_init(self):
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)
        # nn.init.xavier_normal_(self.fc4.weight)

class CAE2(nn.Module):
    def __init__(self):
        super(CAE2, self).__init__()
        self.encoder = Encoder2()
        self.decoder = Decoder2()
        self.classifier = Classifier2()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def classify(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x

    def save_model(self, path='model_weights'):
        torch.save(self.encoder.state_dict(), f'{path}/Encoder')
        torch.save(self.decoder.state_dict(), f'{path}/Decoder')
        torch.save(self.classifier.state_dict(), f'{path}/Classifier')

class Encoder3(nn.Module):
    def __init__(self):
        super(Encoder3, self).__init__()

        self.conv1 = nn.Conv2d(3, 12, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(12, 24, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(24, 48, 3, stride=2, padding=1)

        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.conv2.weight)
        nn.init.xavier_normal_(self.conv3.weight)
        
        # nn.init.zeros(self.conv1.bias)
        # nn.init.zeros(self.conv2.bias)
        # nn.init.zeros(self.conv3.bias)

        # self.pool1 = nn.MaxPool2d(2, stride=2)
        # self.pool2 = nn.MaxPool2d(2, stride=2)

        self.dp1 = nn.Dropout2d(p=0.1)
        self.dp2 = nn.Dropout2d(p=0.1)

        self.bn1 = nn.BatchNorm2d(12)
        self.bn2 = nn.BatchNorm2d(24)
        self.bn3 = nn.BatchNorm2d(48)

    def forward(self, img):
        x = self.conv1(img)
        x = self.bn1(x)
        x = self.dp1(x)
        x = F.relu(x)
        # x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.dp2(x)
        x = F.relu(x)
        # x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.elu(x)

        return x

class Decoder3(nn.Module):
    def __init__(self):
        super(Decoder3, self).__init__()

        self.deconv3 = nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(24, 12, 3, stride=1, padding=1)
        self.deconv1 = nn.ConvTranspose2d(12, 3, 3, stride=1, padding=1)

        nn.init.xavier_normal_(self.deconv1.weight)
        nn.init.xavier_normal_(self.deconv2.weight)
        nn.init.xavier_normal_(self.deconv3.weight)
        
        # nn.init.zeros(self.deconv1.bias)
        # nn.init.zeros(self.deconv2.bias)
        # nn.init.zeros(self.deconv3.bias)

        # self.upsample2 = nn.Upsample(scale_factor=2)
        # self.upsample1 = nn.Upsample(scale_factor=2)
        
        self.dp1 = nn.Dropout2d(p=0.2)
        self.dp2 = nn.Dropout2d(p=0.2)

        self.bn3 = nn.BatchNorm2d(24)
        self.bn2 = nn.BatchNorm2d(12)

    def forward(self, x):
        x = self.deconv3(x)
        x = self.bn3(x)
        x = self.dp1(x)
        x = F.relu(x)

        # x = self.upsample2(x)

        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.dp2(x)
        x = F.relu(x)

        # x = self.upsample1(x)

        x = self.deconv1(x)
        x = torch.sigmoid(x)

        return x


class Classifier3(nn.Module):
    def __init__(self):
        super(Classifier3, self).__init__()
        self.conv1 = nn.Conv2d(48, 48, 3, padding=1)
        self.conv2 = nn.Conv2d(48, 48, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1)

        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.conv2.weight)
        nn.init.xavier_normal_(self.conv3.weight)
        
        # nn.init.zeros(self.conv1.bias)
        # nn.init.zeros(self.conv2.bias)
        # nn.init.zeros(self.conv3.bias)

        self.bn1 = nn.BatchNorm2d(48)
        self.bn2 = nn.BatchNorm2d(48)
        
        # self.pool1 = nn.MaxPool2d(2, stride=2)
        # self.pool2 = nn.MaxPool2d(2, stride=2)
        
        self.pool1 = nn.AvgPool2d(2, stride=2)
        self.pool2 = nn.AvgPool2d(2, stride=2)

        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 64)
        # self.fc1 = nn.Linear(192, 64)
        # self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(64, 10)

        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)
        
        # nn.init.zeros(self.fc1.bias)
        # nn.init.zeros(self.fc2.bias)
        # nn.init.zeros(self.fc3.bias)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(self.pool1(x))

        x = F.relu(self.conv2(x))
        x = self.bn2(self.pool2(x))

        x = F.dropout(F.relu(self.conv3(x)), p=0.2)
        x = x.view(-1, 48 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class CAE3(nn.Module):
    def __init__(self):
        super(CAE3, self).__init__()
        self.encoder = Encoder3()
        self.decoder = Decoder3()
        self.classifier = Classifier3()
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def classify(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x

    def save_model(self, path='model_weights', name=None):
        if name is not None:
            torch.save(self.encoder.state_dict(), f'{path}/Encoder3-{name}')
            torch.save(self.decoder.state_dict(), f'{path}/Decoder3-{name}')
            torch.save(self.classifier.state_dict(), f'{path}/Classifier3-{name}')
            return
        torch.save(self.encoder.state_dict(), f'{path}/Encoder3')
        torch.save(self.decoder.state_dict(), f'{path}/Decoder3')
        torch.save(self.classifier.state_dict(), f'{path}/Classifier3')



"""
ae = AutoEncoder()
ae.encoder.load_state_dict(torch.load('model_weights/auto_encoder'))
"""
if __name__ == '__main__':
    ae = AutoEncoder()
    ae.encoder.load_state_dict(torch.load('model_weights/auto_encoder'))
    print(ae)

