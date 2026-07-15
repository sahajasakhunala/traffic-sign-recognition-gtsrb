import os
import matplotlib.pyplot as plt
from typing import Dict, List, Any

def plot_training_curves(history: Dict[str, List[float]], output_path: str) -> None:
    """Plots training and validation loss/accuracy curves.
    
    Args:
        history: Dictionary containing 'train_loss', 'val_loss', and 'val_acc' keys.
        output_path: Path where the curve plot will be saved.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Loss curves
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="blue", marker="o")
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color="red", marker="x")
    ax1.set_title("Training & Validation Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy curves
    val_acc_pct = [acc * 100 for acc in history["val_acc"]]
    ax2.plot(epochs, val_acc_pct, label="Val Accuracy", color="green", marker="s")
    ax2.set_title("Validation Accuracy")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[INFO] Training curves saved to: {output_path}")
