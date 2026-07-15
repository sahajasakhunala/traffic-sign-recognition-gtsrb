import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingWithWarmup(_LRScheduler):
    """Cosine Annealing learning rate scheduler with linear warmup.
    
    Linearly increases learning rate during warmup epochs, then decays
    following a cosine curve.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        base_lr: float,
        eta_min: float = 1e-6,
        last_epoch: int = -1
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        # During warmup epochs
        if self.last_epoch < self.warmup_epochs:
            alpha = (self.last_epoch + 1) / self.warmup_epochs
            return [self.base_lr * alpha for _ in self.base_lrs]
            
        # Cosine decay phase
        progress = (self.last_epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
        decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = self.eta_min + (self.base_lr - self.eta_min) * decay
        return [lr for _ in self.base_lrs]
