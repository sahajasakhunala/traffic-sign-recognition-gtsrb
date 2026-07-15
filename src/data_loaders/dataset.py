import os
import collections
from typing import Dict, Any, List, Tuple
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets
from PIL import Image

from data_loaders.transforms import get_train_transforms, get_val_transforms
from utils.paths import get_absolute_path

class TransformSubset(Dataset):
    """Wraps a Subset and applies an independent transform — avoids data leakage
    that would occur if both splits shared the same augmented ImageFolder."""
    
    def __init__(self, subset: Subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx) -> Tuple[Any, int]:
        image, label = self.subset[idx]
        path, _ = self.subset.dataset.samples[self.subset.indices[idx]]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label


def stratified_split(
    dataset: datasets.ImageFolder,
    val_split: float,
    seed: int
) -> Tuple[List[int], List[int]]:
    """Per-class stratified split — preserves class-frequency ratios in both sets.
    
    Ensures that each class contributes exactly val_split% to the validation set.
    """
    import torch
    rng = torch.Generator().manual_seed(seed)
    
    class_indices = collections.defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        class_indices[label].append(idx)
        
    train_indices, val_indices = [], []
    for label in sorted(class_indices):
        idxs = class_indices[label]
        perm = torch.randperm(len(idxs), generator=rng).tolist()
        idxs = [idxs[i] for i in perm]
        n_val_c = max(1, int(len(idxs) * val_split))
        val_indices.extend(idxs[:n_val_c])
        train_indices.extend(idxs[n_val_c:])
        
    return train_indices, val_indices


def build_loaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, int, List[str]]:
    """Creates stratified train and validation dataloaders from configuration.
    
    Args:
        config: Combined configuration dictionary.
        
    Returns:
        (train_loader, val_loader, num_classes, class_names)
    """
    data_cfg = config["data"]
    train_cfg = config["training"]
    
    # Define transforms based on image size in config
    image_size = data_cfg["image_size"]
    train_transform = get_train_transforms(image_size)
    val_transform = get_val_transforms(image_size)
    
    # Load raw dataset without transform
    data_dir = get_absolute_path(data_cfg["raw_dir"])
    base_dataset = datasets.ImageFolder(root=data_dir)
    class_names = base_dataset.classes
    num_classes = len(class_names)
    
    train_indices, val_indices = stratified_split(
        dataset=base_dataset,
        val_split=data_cfg["val_split"],
        seed=train_cfg["seed"]
    )
    
    train_set = TransformSubset(Subset(base_dataset, train_indices), train_transform)
    val_set = TransformSubset(Subset(base_dataset, val_indices), val_transform)
    
    loader_kwargs = dict(
        batch_size=train_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
        persistent_workers=(data_cfg["num_workers"] > 0),
    )
    
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    
    return train_loader, val_loader, num_classes, class_names


class GTSRBTestDataset(Dataset):
    """Dataset for the official GTSRB test set.
    
    Reads filenames and class labels from the official GT-final_test.csv.
    """
    
    def __init__(self, csv_file: str, img_dir: str, transform=None) -> None:
        import pandas as pd
        self.df = pd.read_csv(csv_file, sep=";")
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        row = self.df.iloc[idx]
        img_name = row["Filename"]
        img_path = os.path.join(self.img_dir, img_name)
        
        image = Image.open(img_path).convert("RGB")
        label = int(row["ClassId"])
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


def build_test_loader(config: Dict[str, Any]) -> DataLoader:
    """Creates a dataloader for the official GTSRB test dataset based on config.
    
    Args:
        config: Combined configuration dictionary.
        
    Returns:
        DataLoader: Test dataloader.
    """
    data_cfg = config["data"]
    train_cfg = config["training"]
    
    image_size = data_cfg["image_size"]
    test_transform = get_val_transforms(image_size)
    
    # Resolve paths relative to project root
    csv_path = get_absolute_path(os.path.join(data_cfg["raw_dir"], "GT-final_test.csv"))
    img_dir = get_absolute_path(os.path.join(os.path.dirname(data_cfg["raw_dir"]), "test"))
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Official test labels CSV not found at: {csv_path}")
        
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Official test images directory not found at: {img_dir}")
        
    test_set = GTSRBTestDataset(csv_file=csv_path, img_dir=img_dir, transform=test_transform)
    
    test_loader = DataLoader(
        test_set,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
        persistent_workers=(data_cfg["num_workers"] > 0)
    )
    
    return test_loader

