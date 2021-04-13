import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    Encoder part of auto encoder. Encoder is used to feature extraction.
    """
    def __init__(self):
        """

        """
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(3, 12, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(12, 24, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(24, 48, 3, stride=2, padding=1)

        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.conv2.weight)
        nn.init.xavier_normal_(self.conv3.weight)
        
        self.dp1 = nn.Dropout2d(p=0.1)
        self.dp2 = nn.Dropout2d(p=0.1)

        self.bn1 = nn.BatchNorm2d(12)
        self.bn2 = nn.BatchNorm2d(24)
        self.bn3 = nn.BatchNorm2d(48)

    def forward(self, img):
        """
        To decrease dimention and feature extract

        Args:
            img(torch.Tensor): batch_size x channel x  height x width

        Returns(torch.Tensor): batch_size x 48 x 16 x 16

        """
        x = self.conv1(img)
        x = self.bn1(x)
        x = self.dp1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.dp2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.elu(x)

        return x


class Decoder(nn.Module):
    """
    To reconstruct image from output which is predicted from encoder
    """
    def __init__(self):
        """

        """
        super(Decoder, self).__init__()

        self.deconv3 = nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(24, 12, 3, stride=1, padding=1)
        self.deconv1 = nn.ConvTranspose2d(12, 3, 3, stride=1, padding=1)

        nn.init.xavier_normal_(self.deconv1.weight)
        nn.init.xavier_normal_(self.deconv2.weight)
        nn.init.xavier_normal_(self.deconv3.weight)
        
        self.dp1 = nn.Dropout2d(p=0.2)
        self.dp2 = nn.Dropout2d(p=0.2)

        self.bn3 = nn.BatchNorm2d(24)
        self.bn2 = nn.BatchNorm2d(12)

    def forward(self, x):
        """
        To reconstruct from encoder output

        Args:
            x(torch.Tensor): batch_size x 48 x 16 x 16

        Returns(torch.Tensor): batch_size x channel x height x width

        """
        x = self.deconv3(x)
        x = self.bn3(x)
        x = self.dp1(x)
        x = F.relu(x)

        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.dp2(x)
        x = F.relu(x)

        x = self.deconv1(x)
        x = torch.sigmoid(x)

        return x


class Classifier(nn.Module):
    """
    Classifier image which is feature extracted from encoder
    """
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(48, 48, 3, padding=1)
        self.conv2 = nn.Conv2d(48, 48, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1)

        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.conv2.weight)
        nn.init.xavier_normal_(self.conv3.weight)

        self.bn1 = nn.BatchNorm2d(48)
        self.bn2 = nn.BatchNorm2d(48)
        
        self.pool1 = nn.AvgPool2d(2, stride=2)
        self.pool2 = nn.AvgPool2d(2, stride=2)

        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)
        
    def forward(self, x):
        """
        To predict from input

        Args:
            x(torch.Tensor): batch_size x 48 x 16 x 16

        Returns(torch.Tensor): batch_size x 10

        """
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


class CAE(nn.Module):
    """
    Convolutional Auto Encoder
    """
    def __init__(self):
        super(CAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.classifier = Classifier()
    
    def forward(self, x):
        """
        To reconstruct input image

        Args:
            x(torch.Tensor): batch_size x channel x  height x width

        Returns(torch.Tensor): batch_size x channel x height x width

        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def classify(self, x):
        """
        To classify image.

        Args:
            x(torch.Tensor):  batch_size x channel x  height x width

        Returns(torch.Tensor): batch_size x 10

        """
        x = self.encoder(x)
        x = self.classifier(x)
        return x

    def save_model(self, path='model_weights', name=None):
        """
        Save model weights

        Args:
            path(str): directory to save weights
            name(str): to indentify the model weights

        Returns: None

        """
        if name is not None:
            torch.save(self.encoder.state_dict(), os.path.join(path, f'Encoder-{name}'))
            torch.save(self.decoder.state_dict(), os.path.join(path, f'Decoder-{name}'))
            torch.save(self.classifier.state_dict(), os.path.join(path, f'Classifier-{name}'))
            return
        torch.save(self.encoder.state_dict(), os.path.join(path, 'Encoder'))
        torch.save(self.decoder.state_dict(), os.path.join(path, 'Decoder'))
        torch.save(self.classifier.state_dict(), os.path.join(path, 'Classifier'))

    def load_model(self, path='model_weights', name=None):
        """
        Load model weights

        Args:
            path(str): directory to save weights
            name(str): to indentify the model weights

        Returns: None

        """
        if name is not None:
            self.encoder.load_state_dict(torch.load(os.path.join(path, f'Encoder-{name}')))
            self.decoder.load_state_dict(torch.load(os.path.join(path, f'Decoder-{name}')))
            self.classifier.load_state_dict(torch.load(os.path.join(path, f'Classifier-{name}')))
            return
        self.encoder.load_state_dict(torch.load(os.path.join(path, 'Encoder')))
        self.decoder.load_state_dict(torch.load(os.path.join(path, 'Decoder')))
        self.classifier.load_state_dict(torch.load(os.path.join(path, 'Classifier')))
