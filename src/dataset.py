import torch
import random
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset


def get_dataloaders(batch_size=16, seed=42):

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    generator = torch.Generator().manual_seed(seed)


    train_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
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

    train_dataset_full = datasets.ImageFolder(
        "../Drowsy_datset/train",
        transform=train_transform
    )

    val_dataset_full = datasets.ImageFolder(
        "../Drowsy_datset/train",
        transform=test_transform
    )

    test_dataset = datasets.ImageFolder(
        "../Drowsy_datset/test",
        transform=test_transform
    )

    train_size = int(0.8 * len(train_dataset_full))
    val_size = len(train_dataset_full) - train_size

    train_subset, val_subset = random_split(
        train_dataset_full,
        [train_size, val_size],
        generator=generator
    )

    train_indices = train_subset.indices
    val_indices = val_subset.indices

    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(val_dataset_full, val_indices)



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
        train_dataset_full.classes,
        test_dataset
    )