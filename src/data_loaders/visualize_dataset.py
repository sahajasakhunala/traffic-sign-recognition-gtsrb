import os
import random
import matplotlib.pyplot as plt
from PIL import Image
from typing import Optional

def generate_dataset_visualizations(
    data_dir: str,
    output_path: str,
    num_samples: int = 8
) -> None:
    """Selects random images from the dataset and saves them as a combined grid plot.
    
    Args:
        data_dir: Path to the raw data directory containing class subfolders.
        output_path: Path where the output figure should be saved.
        num_samples: Number of random images to include in the grid.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
        
    subdirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    if not subdirs:
        raise ValueError(f"No class folders found in directory: {data_dir}")
        
    # Collect all image paths and labels
    all_samples = []
    for folder in subdirs:
        try:
            class_id = int(folder)
        except ValueError:
            continue
            
        folder_path = os.path.join(data_dir, folder)
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.ppm', '.png', '.jpg', '.jpeg'))]
        for f in files:
            all_samples.append((os.path.join(folder_path, f), class_id))
            
    if not all_samples:
        raise ValueError(f"No valid image files found in class folders.")
        
    random.seed(42)  # For deterministic visualizations
    samples = random.sample(all_samples, min(num_samples, len(all_samples)))
    
    # Create grid (2 rows, 4 cols)
    cols = 4
    rows = (len(samples) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 7))
    
    for idx, (path, class_id) in enumerate(samples):
        ax = axes.flat[idx] if len(samples) > 1 else axes
        with Image.open(path) as img:
            w, h = img.size
            ax.imshow(img)
            ax.set_title(f"Class: {class_id:02d}\nRes: {w}x{h}")
            
        ax.axis('off')
        
    # Turn off unused axes
    for idx in range(len(samples), rows * cols):
        if len(samples) > 1:
            axes.flat[idx].axis('off')
            
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"[INFO] Saved dataset visualization grid to: {output_path}")
