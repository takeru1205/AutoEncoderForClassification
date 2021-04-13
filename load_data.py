"""
Refer from https://github.com/statsu1990/yoto_class_balanced_loss
"""
from copy import deepcopy
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from utils import ImbalancedDatasetSampler

transform = transforms.Compose(
        [transforms.ToTensor(),])


class ImbalancedCIFAR10(Dataset):
    def __init__(self, imbal_class_prop, root='./data', train=True, download=True, transform=transform):
        """

        Args:
            imbal_class_prop(np.ndarray): ratio of each class to be used
            root(string): data directory
            train(bool): If True download train data, else download test data
            download(bool): whether data download or not
            transform(transforms): preprocessing data
        """
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
        idxs = []
        for c in range(class_counts):
            imbal_class_count = self.imbal_class_counts[c]
            idxs.append(class_indices[c][:imbal_class_count])
        idxs = np.hstack(idxs)
        self.labels = targets[idxs]
        return idxs

    def __getitem__(self, index):
        img, target = self.dataset[self.idxs[index]]
        return img, target

    def __len__(self):
        return len(self.idxs)


def separate_data(dataset, indices, minority=(2, 4, 9)):
    """
    To separate one dataset to two dataset of same minority size

    Args:
        dataset(Dataset): dataset includes majority data and minority data
        indices(list): indices of used to be from dataset
        minority(sequence): indexes of minority class

    Returns(list): two set of indices of used to be from dataset

    """
    class_dict = {i:[] for i in range(10)}
    data_len = len(indices)
    for idx in range(data_len):
        _, label = dataset[idx]
        class_dict[label].append(indices[idx])
    d1_indices = [class_dict[k] if k in minority else class_dict[k][:len(class_dict[k])//2] for k in class_dict.keys() ]
    d2_indices = [class_dict[k] if k in minority else class_dict[k][len(class_dict[k])//2:] for k in class_dict.keys() ]
    return sum(d1_indices, []), sum(d2_indices, [])


def generate_data(train_imbalance_class_ratio, train_transform, evaluate_transform, over_sample_batch_size=128, under_sample_batch_size=64, val_batch_size=64, test_batch_size=4):
    """
    To generate dataset from imbalanced dataset
    Args:
        train_imbalance_class_ratio(np.ndarray):  ratio of each class to be used
        train_transform(torchvision.transform): preprocessing data for training
        evaluate_transform(torchvision.transform): preprocessing data for evaluate
        over_sample_batch_size(int): batch size when over sample data
        under_sample_batch_size(int): batch size when under sample data
        val_batch_size(int): batch size for validation data
        test_batch_size(int): batch size for test data

    Returns(DataLoader): over sampled dataloader, under sampled dataloader1, under sampled dataloader2, validation dataloader, test dataloader

    """
    # Load Train Data
    train_set = ImbalancedCIFAR10(train_imbalance_class_ratio, transform=evaluate_transform)

    # Cross Validation Dataset(Stratified K Fold)
    train_indices, val_indices = train_test_split(list(range(len(train_set.labels))), test_size=0.2, stratify=train_set.labels)
    train_dataset = Subset(train_set, train_indices)
    train_dataset.dataset.transform = train_transform
    val_dataset = Subset(train_set, val_indices)
    val_dataset.dataset.transform = evaluate_transform

    # Tom choose indexes for under sampling
    train1_indices, train2_indices = separate_data(train_dataset, train_indices)

    train1_dataset = deepcopy(train_dataset)
    train2_dataset = deepcopy(train_dataset)
    train1_dataset.indices = train1_indices
    train2_dataset.indices = train2_indices

    # Data Loader
    train_loader = DataLoader(train_dataset, sampler=ImbalancedDatasetSampler(train_dataset),
                                             batch_size=over_sample_batch_size, shuffle=False, num_workers=4)  # Over Sampling Data
    train1_loader = DataLoader(train1_dataset, batch_size=under_sample_batch_size, shuffle=True, num_workers=4)  # Under Sampling Data 1
    train2_loader = DataLoader(train2_dataset, batch_size=under_sample_batch_size, shuffle=True, num_workers=4)  # Under Sampling Data 2
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True, num_workers=4)  # Validation Data

    print(f'Train Size: {len(train_loader.dataset)}')
    print(f'Train1 Size: {len(train1_loader.dataset)}')
    print(f'Train2 Size: {len(train2_loader.dataset)}')
    print(f'Validation Size: {len(val_loader.dataset)}')

    print(torch.tensor(train_dataset.dataset.labels[train_indices]).unique(return_counts=True)[1])
    print(torch.tensor(train1_dataset.dataset.labels[train1_indices]).unique(return_counts=True)[1])
    print(torch.tensor(train2_dataset.dataset.labels[train2_indices]).unique(return_counts=True)[1])
    print(torch.tensor(val_dataset.dataset.labels[val_indices]).unique(return_counts=True)[1])

    # Load Test Data
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=evaluate_transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, shuffle=False, num_workers=4)

    return train_loader, train1_loader, train2_loader, val_loader, test_loader



if __name__ == '__main__':
    train_imbalance_class_ratio = np.array([1., 1., .5, 1., .5, 1., 1., 1., 1., .5])
    # Generate Datasets
    train_loader, train1_loader, train2_loader, val_loader, test_loader = generate_data(train_imbalance_class_ratio, transform, transform)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    import matplotlib.pyplot as plt
    
    def imshow(img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
    dataiter = iter(test_loader)
    images, labels = dataiter.next()

    imshow(torchvision.utils.make_grid(images))
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

