import itertools
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from sklearn.metrics import confusion_matrix, f1_score, classification_report

from model import CAE3
from utils import image_result

class Trainer:
    def __init__(self, train0_loader, train1_loader, train2_loader, val_loader, ae_epoch=40, train_epoch=40, ae_lr=0.0005, classify_lr=0.0005, writer=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.net0 = CAE3()
        self.net1 = CAE3()
        self.net2 = CAE3()

        self.net0 = self.net0.to(self.device)
        self.net1 = self.net1.to(self.device)
        self.net2 = self.net2.to(self.device)

        
        self.ae_epoch = 40
        self.train_epoch = 40

        # Optimizer
        self.ae0_optimizer = optim.AdamW(itertools.chain(self.net0.encoder.parameters(), self.net0.decoder.parameters()), lr=ae_lr, betas=(0.9, 0.999))
        self.ae1_optimizer = optim.AdamW(itertools.chain(self.net1.encoder.parameters(), self.net1.decoder.parameters()), lr=ae_lr, betas=(0.9, 0.999))
        self.ae2_optimizer = optim.AdamW(itertools.chain(self.net2.encoder.parameters(), self.net2.decoder.parameters()), lr=ae_lr, betas=(0.9, 0.999))
        self.optimizer0 = optim.AdamW(itertools.chain(self.net0.encoder.parameters(), self.net0.classifier.parameters()), lr=classify_lr, betas=(0.9, 0.999))
        self.optimizer1 = optim.AdamW(itertools.chain(self.net1.encoder.parameters(), self.net1.classifier.parameters()), lr=classify_lr, betas=(0.9, 0.999))
        self.optimizer2 = optim.AdamW(itertools.chain(self.net2.encoder.parameters(), self.net2.classifier.parameters()), lr=classify_lr, betas=(0.9, 0.999))

        # Loss Function
        self.criterion = nn.CrossEntropyLoss()
        self.ae_criterion = nn.MSELoss()

        # Data
        self.train0_loader = train0_loader
        self.train1_loader = train1_loader
        self.train2_loader = train2_loader
        self.val_loader = val_loader

        self.writer = writer

    def ae_train(self, net, ae_optimizer, train_loader, val_loader, name='Net'):
        print(f'Start {name} Auto Encoder Training')
        ae_iter = 0
        for epoch in range(self.ae_epoch):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, _ = data
                inputs = inputs.to(self.device)
                decoded = net(inputs)
                ae_loss = self.ae_criterion(decoded, inputs)
                ae_optimizer.zero_grad()
                ae_loss.backward()
                ae_optimizer.step()
                running_loss += ae_loss.item()

                ae_iter += 1

                if i % 100 == 99:
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                    if self.writer is not None:
                        self.writer.add_scalar(f'Loss/AutoEncoder-{name}', running_loss, ae_iter)
                    running_loss = 0.0
        # Reconstruct Image
        dataiter = iter(val_loader)
        images, _ = dataiter.next()
        self.reconstruct_image(net, images, name)

    def reconstruct_image(self, net, images, name):
        target_img_grid = torchvision.utils.make_grid(images)
        images = images.to(self.device)
        output = net(images)
        img_grid = torchvision.utils.make_grid(output.cpu().data)
        if self.writer is not None:
            self.writer.add_image(f'{name} Reconstruct Image', target_img_grid, self.ae_epoch)
            self.writer.add_image('Target Image', img_grid, self.ae_epoch)

    def classifier_train(self, net, optimizer, train_loader, val_loader, name='Net'):
        print(f'Start Classifier Train {name}')
        train_iter, val_iter = 0, 0
        for epoch in range(self.train_epoch):
            running_loss = 0.0
            # Train
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                predicted = net.classify(inputs)
                loss = self.criterion(predicted, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                train_iter += 1

                if i % 100 == 99:
                    print('[%d, %5d] Train Loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                    if self.writer is not None:
                        self.writer.add_scalar(f'Loss/Train-{name}', running_loss, train_iter)
                    running_loss = 0.0

            # Validation
            total, correct = 0, 0
            val_loss = 0.0
            for i, data in enumerate(val_loader, 0):
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                predicted = net.classify(inputs)
                loss = self.criterion(predicted, labels)

                val_loss += loss.item()

                _, predicted = torch.max(predicted.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                val_iter += 1
                if val_iter % 100 == 99:
                    if self.writer is not None:
                        self.writer.add_scalar(f'Validation/Loss-{name}', loss.item(), val_iter)
                    print('[%d, %5d] Valid Loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                    val_loss = 0.0

            if self.writer is not None:
                self.writer.add_scalar('Validation/Score-{name}', (100 * correct / total), epoch)

            print(f'Epoch: {epoch}, Validation Score: %d %%' % (100 * correct / total))

    def train(self):
        self.ae_train(self.net0, self.ae0_optimizer, self.train0_loader, self.val_loader, name='Net0')
        self.ae_train(self.net1, self.ae1_optimizer, self.train1_loader, self.val_loader, name='Net1')
        self.ae_train(self.net2, self.ae2_optimizer, self.train2_loader, self.val_loader, name='Net2')

        self.classifier_train(self.net0, self.optimizer0, self.train0_loader, self.val_loader, name='Net0')
        self.classifier_train(self.net1, self.optimizer1, self.train1_loader, self.val_loader, name='Net1')
        self.classifier_train(self.net2, self.optimizer2, self.train2_loader, self.val_loader, name='Net2')

    def test(self, test_loader):
        # Evaluate Loop
        self.net0.eval()
        self.net1.eval()
        self.net2.eval()
        correct = 0
        total = 0
        predicted_list = []
        target_list = []

        class_correct = [0.] * 10
        class_total = [0.] * 10

        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                target_list += labels.tolist()
                all_outputs = self.net0.classify(images.to(self.device))
                outputs1 = self.net1.classify(images.to(self.device))
                outputs2 = self.net2.classify(images.to(self.device))
                outputs = (all_outputs + outputs1 + outputs2) / 3
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted.to(self.device) == labels.to(self.device)).sum().item()
                predicted_list += predicted.to('cpu').detach().numpy().copy().tolist()

        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

        classes = ['plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        print(classification_report(target_list, predicted_list, target_names=classes))

    def save_model(self, path='model_weights'):
        self.net0.save_model(path=path, name='0')
        self.net1.save_model(path=path, name='1')
        self.net2.save_model(path=path, name='2')

    def load_model(self, path='model_weights'):
        self.net0.load_model(path=path, name='0')
        self.net1.load_model(path=path, name='1')
        self.net2.load_model(path=path, name='2')




