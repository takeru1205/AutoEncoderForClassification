import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class GridMask():
    """
    Refer from https://github.com/ufoym/imbalanced-dataset-sampler
    """
    def __init__(self, p=0.6, d_range=(96, 224), r=0.6):
        self.p = p
        self.d_range = d_range
        self.r = r

    def __call__(self, sample):
        """
        sample: torch.Tensor(3, height, width)
        """
        if np.random.uniform() > self.p:
            return sample
        sample = sample.numpy()
        side = sample.shape[1]
        d = np.random.randint(*self.d_range, dtype=np.uint8)
        r = int(self.r * d)

        mask = np.ones((side+d, side+d), dtype=np.uint8)
        for i in range(0, side+d, d):
            for j in range(0, side+d, d):
                mask[i: i+(d-r), j: j+(d-r)] = 0
        delta_x, delta_y = np.random.randint(0, d, size=2)
        mask = mask[delta_x: delta_x+side, delta_y: delta_y+side]
        sample *= np.expand_dims(mask, 0)
        return sample

class AddNoise():
    """
    Refer from https://qiita.com/MuAuan/items/539e8075a7225494775e
    """
    def __init__(self, p=0.6, mean=0, std=0.1):
        self.p = p
        self.mean = mean
        self.std = std
    
    def __call__(self, sample):
        if np.random.uniform() > self.p:
            return sample
        sample = sample.numpy()
        return sample + np.random.randn(sample.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class ImbalancedDatasetSampler(data.sampler.Sampler):
    """
    Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
        callback_get_label func: a callback-like function which takes two arguments - dataset and index

    https://github.com/ufoym/imbalanced-dataset-sampler
    """

    def __init__(self, dataset, indices=None, num_samples=None, callback_get_label=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = {}
        for idx in self.indices:
            # label = self._get_label(dataset, idx)
            _, label = dataset[idx]
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
                
        # weight for each sample
        # weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
        weights = [1.0 / label_to_count[dataset[idx][1]] for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        _, label = dataset[idx]
        return label
                
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


def calc_img_stats(loader):
    """
    infer from https://github.com/abhinav3/CIFAR-Autoencoder-Classification/blob/master/Notebook2-AutoEncoder-MSELoss-Compressionfactor2.ipynb
    """
    mean = 0.
    std = 0.
    nb_samples = 0.
    for imgs,_ in loader:
        batch_samples = imgs.size(0)
        imgs = imgs.view(batch_samples, imgs.size(1), -1)
        mean += imgs.mean(2).sum(0)
        std += imgs.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    return mean,std


def imshow(img):
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    plt.imshow(npimg)
    plt.show()


def imsave(img, fname='sample.png'):
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    plt.imsave(fname, npimg)


tmp_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=20),
            transforms.RandomResizedCrop(32),
            transforms.ToTensor(),
        ])


tmp_eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ])


