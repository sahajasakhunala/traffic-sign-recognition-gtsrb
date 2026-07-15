import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

class FocalLoss(nn.Module):
    """Focal Loss implementation for handling class imbalance.
    
    Focal Loss downweights easy examples and focuses the model on hard ones.
    Formula: FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


def get_loss_fn(config: Dict[str, Any]) -> nn.Module:
    """Returns the loss function instance configured in the training dictionary."""
    train_cfg = config["training"]
    loss_type = train_cfg.get("loss_type", "ce").lower()
    
    if loss_type == "focal":
        alpha = train_cfg.get("focal_alpha", 1.0)
        gamma = train_cfg.get("focal_gamma", 2.0)
        return FocalLoss(alpha=alpha, gamma=gamma)
    else:
        label_smoothing = train_cfg.get("label_smoothing", 0.0)
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
