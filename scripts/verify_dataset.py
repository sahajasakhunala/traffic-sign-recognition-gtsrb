import os
import sys

# Resolve project root (one level up from scripts/)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Add src to python path
sys.path.append(os.path.join(project_root, 'src'))

from utils.config import load_config
from datasets.verify_dataset import verify_dataset
from datasets.visualize_dataset import generate_dataset_visualizations
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
        
    # Generate visualization grid
    visual_output = os.path.join(project_root, "results", "predictions", "dataset_samples.png")
    print(f"[INFO] Generating sample grid...")
    generate_dataset_visualizations(
        data_dir=raw_path,
        output_path=visual_output,
        num_samples=8
    )
        
    # Build loaders
    print("[INFO] Building PyTorch dataloaders...")
    try:
        train_loader, val_loader, num_classes, class_names = build_loaders(config)
        
        print("[SUCCESS] Dataloaders built successfully!")
        print(f"  Total train batches : {len(train_loader)}")
        print(f"  Total val batches   : {len(val_loader)}")
        print(f"  Number of classes   : {num_classes}")
        
        # Pull one batch
        images, labels = next(iter(train_loader))
        print(f"  Batch images shape  : {images.shape}")
        print(f"  Batch labels shape  : {labels.shape}")
        
    except Exception as e:
        print(f"[ERROR] Failed to build or test dataloaders: {e}")

if __name__ == "__main__":
    main()
