import argparse
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from load_data import generate_data
from trainer import Trainer


# Command Line Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--train-epoch", default=50, type=int, help='Epoch of train Classification')
parser.add_argument("--ae-epoch", default=50, type=int, help='Epoch of train Auto Encoder')
parser.add_argument("--load", action='store_true', help='To load model weight')
parser.add_argument("--save", action='store_true', help='To save model weight')
parser.add_argument("--ae-lr", default=0.0005, type=float, help='Learing Rate of AutoEncoder')
parser.add_argument("--classify-lr", default=0.0005, type=float, help='Learing Rate of Classifier')
parser.add_argument("--seed", default=42, type=int, help='Seed Value')
args = parser.parse_args()


# Get from calc_img_stats() with Imbalanced Data
transform_mean = [0.4920, 0.4825, 0.4500]
transform_std = [0.2039, 0.2009, 0.2026]

# Preprocessing for Training
train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=20),
            transforms.RandomResizedCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(transform_mean, transform_std)
        ])

# Preprocessing for Evaluate
evaluate_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(transform_mean, transform_std)
        ])


# Set Seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Log
writer = SummaryWriter(log_dir=f'logs/ensemble-{args.ae_epoch}-{args.train_epoch}-{args.ae_lr}-{args.classify_lr}')

# Define Imbalance Ratio
train_imbalance_class_ratio = np.array([1., 1., .5, 1., .5, 1., 1., 1., 1., .5])

# Generate Datasets
train_loader, train1_loader, train2_loader, val_loader, test_loader = generate_data(train_imbalance_class_ratio, train_transform, evaluate_transform, over_sample_batch_size=128, under_sample_batch_size=64, val_batch_size=64, test_batch_size=4)

trainer = Trainer(train0_loader=train_loader, train1_loader=train1_loader, train2_loader=train2_loader, val_loader=val_loader, ae_epoch=args.ae_epoch, train_epoch=args.train_epoch, writer=writer)

if args.load:
    trainer.load_model()

# Train
trainer.train()

# Model Save
if args.save:
    trainer.save_model()

# Test
trainer.test(test_loader=test_loader)



