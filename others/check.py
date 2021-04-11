from load_data import ImbalancedCIFAR10, separate_data
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

train_imbalance_class_ratio = np.array([1., 1., .5, 1., .5, 1., 1., 1., 1., .5])
train_set = ImbalancedCIFAR10(train_imbalance_class_ratio)
train_indices, val_indices = train_test_split(list(range(len(train_set.labels))), test_size=0.2, stratify=train_set.labels)
train_dataset = Subset(train_set, train_indices)
val_dataset = Subset(train_set, val_indices)
train1_indices, train2_indices = separate_data(train_dataset, train_indices)
train1_dataset = Subset(train_dataset, train1_indices)
train2_dataset = Subset(train_dataset, train2_indices)
train1_loader = DataLoader(train1_dataset, batch_size=64, shuffle=True, num_workers=4)
train2_loader = DataLoader(train2_dataset, batch_size=64, shuffle=True, num_workers=4)
print(torch.tensor(train1_dataset.dataset.dataset.labels[train1_indices]).unique(return_counts=True))
print(torch.tensor(train2_dataset.dataset.dataset.labels[train2_indices]).unique(return_counts=True))

