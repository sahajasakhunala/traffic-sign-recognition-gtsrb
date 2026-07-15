import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple, List, Any

@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, List[int], List[int]]:
    """Evaluates the model on the provided dataloader.
    
    Args:
        model: PyTorch model.
        dataloader: PyTorch DataLoader containing evaluation samples.
        criterion: Loss function.
        device: Device to run evaluation on (CPU or CUDA).
        
    Returns:
        (avg_loss, accuracy, all_predictions, all_targets)
    """
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_targets = []
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Stats
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        
        all_preds.extend(predicted.cpu().numpy().tolist())
        all_targets.extend(labels.cpu().numpy().tolist())
        
    avg_loss = total_loss / total
    accuracy = correct / total
    
    return avg_loss, accuracy, all_preds, all_targets
