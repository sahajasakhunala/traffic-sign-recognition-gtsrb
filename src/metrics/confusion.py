import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List

def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    output_path: str
) -> None:
    """Generates and saves a confusion matrix plot using matplotlib.
    
    Args:
        y_true: List of ground-truth class indices.
        y_pred: List of predicted class indices.
        class_names: Names/IDs of the classes.
        output_path: Path where the plot will be saved.
    """
    num_classes = len(class_names)
    
    # Calculate confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
        
    fig, ax = plt.subplots(figsize=(16, 16))
    
    # Normalize rows
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
    
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Show all ticks and label them with the class names
    ax.set(
        xticks=np.arange(num_classes),
        yticks=np.arange(num_classes),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Normalized Confusion Matrix",
        ylabel="True Label",
        xlabel="Predicted Label"
    )
    
    # Rotate tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations for large values
    thresh = cm_norm.max() / 2.
    for i in range(num_classes):
        for j in range(num_classes):
            if cm[i, j] > 0:
                ax.text(
                    j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > thresh else "black",
                    fontsize=8
                )
                
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[INFO] Confusion matrix saved to: {output_path}")
