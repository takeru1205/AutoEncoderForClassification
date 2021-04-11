import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


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
            _, label = dataset[idx]
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
                
        # weight for each sample
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


def unnormalize(img, std, mean):
    for i, s, m in zip(img, std, mean):
        i.mul_(s).add_(s)
    return img


def imsave(img, mean, std, fname='sample.png'):
    for i, s, m in zip(img, std, mean):
        i.mul_(s)
        i.add_(m)
    img = torch.clamp(img, min=0., max=1.)
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    plt.imsave(fname, npimg)


def image_result(images, net, net1, net2, transform_mean, transform_std, writer, epoch=0):
    # net
    output = net(images)
    img_grid = torchvision.utils.make_grid(output.cpu().data)
    imsave(img_grid, transform_mean, transform_std, 'predict.png')
    writer.add_image('Net Reconstruct Image', img_grid, epoch)
    # net1
    output = net1(images)
    img_grid = torchvision.utils.make_grid(output.cpu().data)
    imsave(img_grid, transform_mean, transform_std, 'predict1.png')
    writer.add_image('Net1 Reconstruct Image', img_grid, epoch)
    # net2
    output = net2(images)
    img_grid = torchvision.utils.make_grid(output.cpu().data)
    imsave(img_grid, transform_mean, transform_std, 'predict2.png')
    writer.add_image('Net2 Reconstruct Image', img_grid, epoch)


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


