import os
import sys
import argparse
import torch
import yaml

# Resolve project root dynamically and add src/ to python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, 'src'))

from utils.config import load_yaml
from utils.seed import set_seed
from utils.logger import setup_logger
from data_loaders.dataset import build_loaders
from models.factory import create_model
from engine.losses import get_loss_fn
from engine.trainer import train_model
from engine.evaluator import evaluate
from metrics.confusion import plot_confusion_matrix
from metrics.classification import generate_classification_report
from metrics.plots import plot_training_curves

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GTSRB PyTorch Training Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/models/cnn_v2.yaml",
        help="Path to the experiment configuration YAML file."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load configuration
    config = load_yaml(args.config)
    exp_name = config["experiment_name"]
    
    # Setup experiment directory
    exp_dir = os.path.join(project_root, "experiments", exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Initialize logger
    logger = setup_logger("gtsrb", log_file=os.path.join(exp_dir, "train.log"))
    logger.info(f"Loaded config: {args.config}")
    logger.info(f"Experiment directory: {exp_dir}")
    
    # Save the configuration used to the experiment folder for reproducibility
    with open(os.path.join(exp_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)
        
    # Set seed for reproducibility
    seed = config["training"].get("seed", 42)
    set_seed(seed)
    logger.info(f"Deterministic seed set to: {seed}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Build loaders
    logger.info("Initializing datasets and dataloaders...")
    train_loader, val_loader, num_classes, class_names = build_loaders(config)
    logger.info(f"Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}")
    
    # Create model
    logger.info("Instantiating model architecture...")
    model = create_model(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total model parameters: {total_params:,}")
    
    # Get loss function
    criterion = get_loss_fn(config)
    
    # Run training loop
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        config=config,
        device=device,
        exp_dir=exp_dir
    )
    
    # --- Post-Training Evaluation on Best Model Checkpoint ---
    logger.info("Loading best model checkpoint for full evaluation...")
    best_path = os.path.join(exp_dir, "best_model.pth")
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        
    # Evaluate best model on validation loader
    logger.info("Evaluating best model performance on validation data...")
    val_loss, val_acc, y_pred, y_true = evaluate(model, val_loader, criterion, device)
    logger.info(f"Best Model Validation Loss: {val_loss:.4f} | Accuracy: {val_acc * 100:.2f}%")
    
    # Plot curves
    logger.info("Generating training and loss curve plots...")
    curves_path = os.path.join(project_root, "results", "curves", f"{exp_name}_curves.png")
    # Also save in experiment directory
    plot_training_curves(history, curves_path)
    plot_training_curves(history, os.path.join(exp_dir, "curves.png"))
    
    # Generate confusion matrix
    logger.info("Generating confusion matrix...")
    cm_path = os.path.join(project_root, "results", "confusion_matrix", f"{exp_name}_cm.png")
    plot_confusion_matrix(y_true, y_pred, class_names, cm_path)
    plot_confusion_matrix(y_true, y_pred, class_names, os.path.join(exp_dir, "confusion_matrix.png"))
    
    # Generate per-class classification report
    logger.info("Generating per-class classification metrics...")
    report_path = os.path.join(project_root, "results", "metrics", f"{exp_name}_report.json")
    generate_classification_report(y_true, y_pred, class_names, report_path)
    generate_classification_report(y_true, y_pred, class_names, os.path.join(exp_dir, "classification_report.json"))
    
    logger.info("Training pipeline execution completed successfully!")

if __name__ == "__main__":
    main()
