import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.config import load_config
from datasets.integrity import verify_dataset
from datasets.dataset import build_loaders

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "configs", "cnn_v2.yaml")
    if not os.path.exists(config_path):
        print(f"[ERROR] Configuration file not found at: {config_path}")
        return
        
    config = load_config(config_path)
    
    # Check if data directory is configured
    data_raw_dir = config['data']['raw_dir']
    
    # Sibling folder raw directory path:
    raw_path = os.path.abspath(os.path.join(script_dir, data_raw_dir))
    
    print(f"[INFO] Configured raw data path: {raw_path}")
    
    # Run dataset integrity verification
    # Note: verify_dataset will tell them where to download if it doesn't exist
    dataset_ok = verify_dataset(raw_path)
    
    if not dataset_ok:
        print("[ERROR] Dataset verification failed. Please follow the instructions above to download and place the dataset.")
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
        print(f"  Batch images shape  : {images.shape}")  # [B, 3, H, W]
        print(f"  Batch labels shape  : {labels.shape}")  # [B]
        
        # Save sample images for visual confirmation
        print("[INFO] Saving sample images for confirmation...")
        os.makedirs("results", exist_ok=True)
        
        # Un-normalize images for display
        # Mean & Std of GTSRB used in transformations:
        mean = np.array([0.3337, 0.3064, 0.3171])
        std = np.array([0.2672, 0.2564, 0.2629])
        
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        for i, ax in enumerate(axes.flat):
            if i >= len(images):
                break
            img = images[i].permute(1, 2, 0).numpy()
            img = img * std + mean  # Denormalize
            img = np.clip(img, 0, 1)
            
            ax.imshow(img)
            ax.set_title(f"Class: {labels[i].item()}")
            ax.axis('off')
            
        sample_path = "results/sample_batch.png"
        plt.tight_layout()
        plt.savefig(sample_path)
        plt.close()
        print(f"[SUCCESS] Sample images saved to: {os.path.abspath(sample_path)}")
        
    except Exception as e:
        print(f"[ERROR] Failed to build or test dataloaders: {e}")

if __name__ == "__main__":
    main()
