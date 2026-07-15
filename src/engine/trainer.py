import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, Any, Optional

from utils.logger import setup_logger
from engine.evaluator import evaluate
from engine.scheduler import CosineAnnealingWithWarmup

class EMA:
    """Exponential Moving Average (EMA) of model parameters.
    
    Smooths model weights over training steps to improve generalization
    and robustness.
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Register shadow weights
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self) -> None:
        """Updates the shadow weights using the current model weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self) -> None:
        """Copies the shadow weights into the active model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self) -> None:
        """Restores the original model weights from backup."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup.clear()


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    config: Dict[str, Any],
    device: torch.device,
    exp_dir: str
) -> Dict[str, Any]:
    """Runs the training loop for the given model and configuration.
    
    Features:
    - Cosine LR Annealing with linear warmup.
    - Automatic Mixed Precision (AMP) training.
    - Gradient accumulation support.
    - Exponential Moving Average (EMA) model tracking.
    - Config-driven Early Stopping and automatic checkpoint naming.
    """
    train_cfg = config["training"]
    logger = setup_logger("gtsrb", log_file=os.path.join(exp_dir, "train.log"))
    
    epochs = train_cfg["epochs"]
    base_lr = train_cfg["lr"]
    weight_decay = train_cfg.get("weight_decay", 1e-4)
    use_amp = train_cfg.get("use_amp", False)
    grad_accum_steps = train_cfg.get("gradient_accumulation_steps", 1)
    early_stopping_patience = train_cfg.get("early_stopping_patience", 10)
    
    # Initialize Optimizer
    optimizer_type = train_cfg.get("optimizer", "adam").lower()
    if optimizer_type == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)
        
    # Initialize Scheduler
    warmup_epochs = train_cfg.get("warmup_epochs", 3)
    scheduler = CosineAnnealingWithWarmup(
        optimizer=optimizer,
        warmup_epochs=warmup_epochs,
        total_epochs=epochs,
        base_lr=base_lr,
        eta_min=1e-6
    )
    
    # Optional EMA setup
    use_ema = train_cfg.get("use_ema", False)
    ema_decay = train_cfg.get("ema_decay", 0.999)
    ema = EMA(model, decay=ema_decay) if use_ema else None
    
    # AMP setup
    scaler = GradScaler(enabled=use_amp)
    
    # TensorBoard setup (optional)
    tb_writer = None
    if train_cfg.get("use_tensorboard", False):
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_writer = SummaryWriter(log_dir=os.path.join(exp_dir, "runs"))
            logger.info("TensorBoard logging enabled.")
        except ImportError:
            logger.warning("TensorBoard is not installed. Skipping SummaryWriter initialization.")
            
    best_val_acc = 0.0
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "lr": []}
    
    logger.info(f"Starting training on device: {device}")
    logger.info(f"Optimizing model '{config['model']['name']}' with {optimizer_type} (lr={base_lr})")
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass under autocast (AMP)
            with autocast(enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)
                # Scale loss for gradient accumulation
                loss = loss / grad_accum_steps
                
            # Backward pass
            scaler.scale(loss).backward()
            
            # Step optimizer after accumulating gradients
            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                if train_cfg.get("grad_clip", 0.0) > 0.0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
                    
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # Update EMA weights
                if ema:
                    ema.update()
                    
            running_loss += loss.item() * grad_accum_steps * images.size(0)
            
        epoch_train_loss = running_loss / len(train_loader.dataset)
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        
        # Validation Evaluation
        # If EMA is active, validate using the smoothed weights
        if ema:
            ema.apply_shadow()
            
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        
        if ema:
            ema.restore()
            
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)
        
        # Tensorboard log
        if tb_writer:
            tb_writer.add_scalar("Loss/Train", epoch_train_loss, epoch)
            tb_writer.add_scalar("Loss/Val", val_loss, epoch)
            tb_writer.add_scalar("Accuracy/Val", val_acc, epoch)
            tb_writer.add_scalar("LR", current_lr, epoch)
            
        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc * 100:.2f}% | "
            f"LR: {current_lr:.6f}"
        )
        
        # Checkpoint Saving
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            
            # Save checkpoint
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema.shadow if ema else None,
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "config": config
            }
            
            # Save best checkpoint
            best_path = os.path.join(exp_dir, "best_model.pth")
            torch.save(checkpoint, best_path)
            
            # Automatic descriptive checkpoint naming
            desc_name = f"{config['model']['name']}_epoch{epoch:02d}_acc{val_acc*100:.2f}.pth"
            desc_path = os.path.join(exp_dir, desc_name)
            torch.save(checkpoint, desc_path)
            
            # Keep copy under models/checkpoints
            global_chk_dir = os.path.abspath(os.path.join(exp_dir, "..", "..", "models", "checkpoints"))
            os.makedirs(global_chk_dir, exist_ok=True)
            torch.save(checkpoint, os.path.join(global_chk_dir, f"{config['model']['name']}_best.pth"))
            
            logger.info(f"[SAVED] New best model saved to: {desc_name}")
        else:
            epochs_no_improve += 1
            
        # Early Stopping check
        if epochs_no_improve >= early_stopping_patience:
            logger.info(f"Early stopping triggered after {epoch} epochs (no validation improvement for {early_stopping_patience} epochs).")
            break
            
    if tb_writer:
        tb_writer.close()
        
    # Save training metrics to JSON
    metrics_path = os.path.join(exp_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)
        
    logger.info(f"Training completed. Best Validation Accuracy: {best_val_acc * 100:.2f}%")
    return history
