import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from trainer import Trainer


# Get from calc_img_stats() with Imbalanced Data
transform_mean = [0.4920, 0.4825, 0.4500]
transform_std = [0.2039, 0.2009, 0.2026]

# Preprocessing for Evaluate
evaluate_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(transform_mean, transform_std)
        ])

# Test data
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=evaluate_transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=False, num_workers=4)

trainer = Trainer()

trainer.load_model()

# Test
trainer.test(test_loader=test_loader)


