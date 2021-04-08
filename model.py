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

        # self.pool1 = nn.MaxPool2d(2, stride=2)
        # self.pool2 = nn.MaxPool2d(2, stride=2)

        self.bn1 = nn.BatchNorm2d(12)
        self.bn2 = nn.BatchNorm2d(24)
        self.bn3 = nn.BatchNorm2d(48)

    def forward(self, img):
        x = self.conv1(img)
        # x = self.bn1(x)
        x = F.relu(x)
        # x = self.pool1(x)

        x = self.conv2(x)
        # x = self.bn2(x)
        x = F.relu(x)
        # x = self.pool2(x)

        x = self.conv3(x)
        # x = self.bn3(x)
        x = F.elu(x)

        return x

class Decoder3(nn.Module):
    def __init__(self):
        super(Decoder3, self).__init__()

        self.deconv3 = nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(24, 12, 3, stride=1, padding=1)
        self.deconv1 = nn.ConvTranspose2d(12, 3, 3, stride=1, padding=1)

        # self.upsample2 = nn.Upsample(scale_factor=2)
        # self.upsample1 = nn.Upsample(scale_factor=2)

        self.bn3 = nn.BatchNorm2d(24)
        self.bn2 = nn.BatchNorm2d(12)

    def forward(self, x):
        x = self.deconv3(x)
        # x = self.bn3(x)
        x = F.relu(x)

        # x = self.upsample2(x)

        x = self.deconv2(x)
        # x = self.bn2(x)
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

        self.bn1 = nn.BatchNorm2d(48)
        self.bn2 = nn.BatchNorm2d(48)
        
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.pool2 = nn.MaxPool2d(2, stride=2)

        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(self.pool1(x))

        x = F.relu(self.conv2(x))
        x = self.bn2(self.pool2(x))

        x = F.dropout(F.relu(self.conv3(x)), p=0.1)
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

    def save_model(self, path='model_weights'):
        torch.save(self.encoder.state_dict(), f'{path}/Encoder3')
        torch.save(self.decoder.state_dict(), f'{path}/Decoder3')
        torch.save(self.classifier.state_dict(), f'{path}/Classifier3')


class Encoder4(nn.Module):
    def __init__(self):
        super(Encoder4, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 8, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(8, 8, 3, stride=1, padding=1)

        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.pool2 = nn.MaxPool2d(2, stride=2)

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(8)
        self.bn3 = nn.BatchNorm2d(8)

    def forward(self, img):
        x = self.conv1(img)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.elu(x)

        return x

class Decoder4(nn.Module):
    def __init__(self):
        super(Decoder4, self).__init__()

        self.deconv3 = nn.ConvTranspose2d(8, 8, 3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(8, 16, 3, stride=1, padding=1)
        self.deconv1 = nn.ConvTranspose2d(16, 1, 3, stride=1, padding=1)

        self.upsample2 = nn.Upsample(scale_factor=2)
        self.upsample1 = nn.Upsample(scale_factor=2)

        self.bn3 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)

    def forward(self, x):
        x = self.deconv3(x)
        x = self.bn3(x)
        x = F.elu(x)

        x = self.upsample2(x)

        x = self.deconv2(x)
        x = self.bn2(x)
        x = F.elu(x)

        x = self.upsample1(x)

        x = self.deconv1(x)
        x = torch.sigmoid(x)

        return x


class Classifier4(nn.Module):
    def __init__(self):
        super(Classifier4, self).__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 512)
        x = self.fc1(x)
        x = F.elu(x)

        x = self.fc2(x)
        x = F.elu(x)

        x = self.fc3(x)
        return x


class CAE4(nn.Module):
    def __init__(self):
        super(CAE4, self).__init__()
        self.encoder = Encoder4()
        self.decoder = Decoder4()
        self.classifier = Classifier4()
    
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

