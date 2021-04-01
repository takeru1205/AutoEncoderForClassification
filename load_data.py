import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


class ImbalancedCIFAR10(Dataset):
    def __init__(self, imbal_class_prop, root='./data', train=True, download=True, transform=transform):
        self.dataset = torchvision.datasets.CIFAR10(
            root=root, train=train, download=download, transform=transform)
        self.train = train
        self.imbal_class_prop = imbal_class_prop
        self.idxs = self.resample()

    def resample(self):
        '''
        Resample the indices to create an artificially imbalanced dataset.
        '''
        # Get class indices for resampling
        targets, class_counts = np.array(self.dataset.targets), 10
        classes, class_datasize = torch.tensor(self.dataset.targets).unique(return_counts=True)
        class_indices = [np.where(targets == i)[0] for i in range(class_counts)]
        # Reduce class count by proportion
        self.imbal_class_counts = [
            int(count * prop)
            for count, prop in zip(class_datasize, self.imbal_class_prop)
        ]
        # Get class indices for reduced class count
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                'dog', 'frog', 'horse', 'ship', 'truck']
        idxs = []
        for c in range(class_counts):
            imbal_class_count = self.imbal_class_counts[c]
            idxs.append(class_indices[c][:imbal_class_count])
            print(f'Label {c}, {classes[c]} Data Size: {imbal_class_count}')
        idxs = np.hstack(idxs)
        self.labels = targets[idxs]
        return idxs

    def __getitem__(self, index):
        img, target = self.dataset[self.idxs[index]]
        return img, target

    def __len__(self):
        return len(self.idxs)


if __name__ == '__main__':
    # train_imbalance_class_ratio = np.hstack(([0.1] * 5, [1.0] * 5))
    train_imbalance_class_ratio = np.array([1., 1., .5, 1., .5, 1., 1., 1., 1., .5])
    train_imbalanced_dataset = ImbalancedCIFAR10(train_imbalance_class_ratio)
    train_imbalanced_loader = torch.utils.data.DataLoader(
            train_imbalanced_dataset, batch_size=4, shuffle=True, num_workers=4)
    import matplotlib.pyplot as plt
    
    def imshow(img):
        img = img / 2 + 0.5
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    dataiter = iter(train_imbalanced_loader)
    images, labels = dataiter.next()

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    imshow(torchvision.utils.make_grid(images))
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

