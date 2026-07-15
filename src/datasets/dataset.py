import os
import collections
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from PIL import Image

class TransformSubset(torch.utils.data.Dataset):
    """Wraps a Subset and applies an independent transform — avoids data leakage
    that would occur if both splits shared the same augmented ImageFolder."""
    
    def __init__(self, subset: torch.utils.data.Subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        # Retrieve the underlying image path to apply independent PIL transform
        path, _ = self.subset.dataset.samples[self.subset.indices[idx]]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label


def stratified_split(
    dataset: datasets.ImageFolder,
    val_split: float,
    seed: int
) -> tuple[list[int], list[int]]:
    """Per-class stratified split — preserves class-frequency ratios in both sets.
    
    Ensures that each class contributes exactly val_split% to the validation set.
    """
    rng = torch.Generator().manual_seed(seed)
    
    # Group indices by class label
    class_indices = collections.defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        class_indices[label].append(idx)
        
    train_indices, val_indices = [], []
    for label in sorted(class_indices):
        idxs = class_indices[label]
        # Shuffle within each class
        perm = torch.randperm(len(idxs), generator=rng).tolist()
        idxs = [idxs[i] for i in perm]
        n_val_c = max(1, int(len(idxs) * val_split))  # At least 1 validation image/class
        val_indices.extend(idxs[:n_val_c])
        train_indices.extend(idxs[n_val_c:])
        
    return train_indices, val_indices


def build_loaders(
    data_dir: str,
    image_size: int,
    batch_size: int,
    val_split: float,
    num_workers: int,
    seed: int,
    pin_memory: bool = True
) -> tuple[DataLoader, DataLoader, int, list[str]]:
    """Creates stratified train and validation dataloaders with distinct transforms.
    
    Returns:
        (train_loader, val_loader, num_classes, class_names)
    """
    # Define baseline transforms
    # Note: No RandomHorizontalFlip because it invalidates signs like "Keep Right/Left"
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.10, 0.10),
            scale=(0.90, 1.10),
            shear=5,
        ),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.05,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.3337, 0.3064, 0.3171],  # Official GTSRB statistics
            std=[0.2672, 0.2564, 0.2629],
        ),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.3337, 0.3064, 0.3171],
            std=[0.2672, 0.2564, 0.2629],
        ),
    ])
    
    # Load once without transform
    base_dataset = datasets.ImageFolder(root=data_dir)
    class_names = base_dataset.classes
    num_classes = len(class_names)
    
    train_indices, val_indices = stratified_split(base_dataset, val_split, seed)
    
    train_set = TransformSubset(Subset(base_dataset, train_indices), train_transform)
    val_set = TransformSubset(Subset(base_dataset, val_indices), val_transform)
    
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    
    return train_loader, val_loader, num_classes, class_names
