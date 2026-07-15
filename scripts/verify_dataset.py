import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# Resolve project root (one level up from scripts/)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Add src to python path
sys.path.append(os.path.join(project_root, 'src'))

from utils.config import load_config
from datasets.integrity import verify_dataset
from datasets.dataset import build_loaders

def main():
    config_path = os.path.join(project_root, "configs", "cnn_v2.yaml")
    if not os.path.exists(config_path):
        print(f"[ERROR] Configuration file not found at: {config_path}")
        return
        
    config = load_config(config_path)
    
    # Check if data directory is configured
    data_raw_dir = config['data']['raw_dir']
    
    # Raw path relative to project root
    raw_path = os.path.abspath(os.path.join(project_root, data_raw_dir))
    
    print(f"[INFO] Configured raw data path: {raw_path}")
    
    # Run dataset integrity verification
    dataset_ok = verify_dataset(raw_path)
    
    if not dataset_ok:
        print("[ERROR] Dataset verification failed. Please download the dataset and place it correctly.")
        return
        
    # Build loaders
    print("[INFO] Building PyTorch dataloaders...")
    try:
        train_loader, val_loader, num_classes, class_names = build_loaders(
            data_dir=raw_path,
            image_size=config['data']['image_size'],
            batch_size=config['training']['batch_size'],
            val_split=config['data']['val_split'],
            num_workers=config['data']['num_workers'],
            seed=config['training']['seed']
        )
        
        print("[SUCCESS] Dataloaders built successfully!")
        print(f"  Total train batches : {len(train_loader)}")
        print(f"  Total val batches   : {len(val_loader)}")
        print(f"  Number of classes   : {num_classes}")
        
        # Pull one batch
        images, labels = next(iter(train_loader))
        print(f"  Batch images shape  : {images.shape}")
        print(f"  Batch labels shape  : {labels.shape}")
        
        # Save sample images
        print("[INFO] Saving sample images for confirmation...")
        os.makedirs(os.path.join(project_root, "results"), exist_ok=True)
        
        mean = np.array([0.3337, 0.3064, 0.3171])
        std = np.array([0.2672, 0.2564, 0.2629])
        
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        for i, ax in enumerate(axes.flat):
            if i >= len(images):
                break
            img = images[i].permute(1, 2, 0).numpy()
            img = img * std + mean
            img = np.clip(img, 0, 1)
            
            ax.imshow(img)
            ax.set_title(f"Class: {labels[i].item()}")
            ax.axis('off')
            
        sample_path = os.path.join(project_root, "results", "sample_batch.png")
        plt.tight_layout()
        plt.savefig(sample_path)
        plt.close()
        print(f"[SUCCESS] Sample images saved to: {os.path.abspath(sample_path)}")
        
    except Exception as e:
        print(f"[ERROR] Failed to build or test dataloaders: {e}")

if __name__ == "__main__":
    main()
