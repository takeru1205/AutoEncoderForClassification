"""
Train and Evaluation Script of Combined Model AutoEncoder and Classifier.
These are trained separatly.
"""
import numpy as np
import torch
import torchvision.transforms as transforms

from utils import calc_img_stats
from load_data import generate_data


train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=20),
            transforms.RandomResizedCrop(32),
            transforms.ToTensor(),
        ])


evaluate_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ])


# Set Seed
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# Load Train Data
train_imbalance_class_ratio = np.array([1., 1., .5, 1., .5, 1., 1., 1., 1., .5])

train_loader, _, _, _, _ = generate_data(train_imbalance_class_ratio, train_transform, evaluate_transform, over_sample_batch_size=128, under_sample_batch_size=64, val_batch_size=64, test_batch_size=4)

mean, std = calc_img_stats(train_loader)
print(f'Mean: {mean.tolist()}')
print(f'Std: {std.tolist()}')

