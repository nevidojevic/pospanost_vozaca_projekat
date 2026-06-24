

import torch
import random
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


def get_dataloaders(batch_size=16, seed=42):

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    generator = torch.Generator().manual_seed(seed)


    # IMAGE TRANSFORMS


    train_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


    # DATASET
    full_dataset = datasets.ImageFolder(
        "../Drowsy_datset/train",
        transform=train_transform
    )

    test_dataset = datasets.ImageFolder(
        "../Drowsy_datset/test",
        transform=test_transform
    )

    # TRAIN / VAL SPLIT

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=generator
    )


    # DATALOADERS

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        full_dataset.classes,
        test_dataset
    )