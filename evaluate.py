import os
import sys
import json
import argparse
import torch
import torch.nn as nn
from PIL import Image

# Add src to python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, 'src'))

from utils.config import load_yaml
from utils.logger import setup_logger
from data_loaders.dataset import build_test_loader
from models.factory import create_model
from metrics.confusion import plot_confusion_matrix
from metrics.classification import generate_classification_report

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GTSRB Trained Model on Official Test Set")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/models/cnn_v2.yaml",
        help="Path to the config YAML file."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the trained model checkpoint (best_model.pth)."
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Use EMA weights from the checkpoint if available."
    )
    return parser.parse_args()

@torch.no_grad()
def evaluate_test_set(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device
) -> tuple[float, list[int], list[int], list[dict]]:
    """Evaluates the model on the test loader and tracks misclassified samples."""
    model.eval()
    correct = 0
    total = 0
    
    all_preds = []
    all_targets = []
    misclassified = []
    
    dataset = dataloader.dataset
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        
        preds_cpu = predicted.cpu().tolist()
        targets_cpu = labels.cpu().tolist()
        
        all_preds.extend(preds_cpu)
        all_targets.extend(targets_cpu)
        
        # Track misclassified files in this batch
        batch_size = images.size(0)
        for i in range(batch_size):
            pred_idx = preds_cpu[i]
            target_idx = targets_cpu[i]
            
            if pred_idx != target_idx:
                # Find original index in dataset
                global_idx = batch_idx * dataloader.batch_size + i
                if global_idx < len(dataset):
                    filename = dataset.df.iloc[global_idx]["Filename"]
                    misclassified.append({
                        "filename": filename,
                        "true_class": target_idx,
                        "predicted_class": pred_idx,
                        "original_index": global_idx
                    })
                    
    accuracy = correct / total
    return accuracy, all_preds, all_targets, misclassified

def main():
    args = parse_args()
    
    # Load configuration
    config = load_yaml(args.config)
    exp_name = config["experiment_name"]
    
    # Setup logger
    logger = setup_logger("gtsrb_eval")
    logger.info(f"Loaded config: {args.config}")
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create model
    model = create_model(config).to(device)
    
    # Load state dict
    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint not found at: {args.checkpoint}")
        return
        
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    if args.use_ema and checkpoint.get("ema_state_dict") is not None:
        logger.info("Loading EMA weights for evaluation...")
        model.load_state_dict(checkpoint["ema_state_dict"])
    else:
        logger.info("Loading normal model weights for evaluation...")
        model.load_state_dict(checkpoint["model_state_dict"])
        
    # Build test loader
    logger.info("Loading official test dataset...")
    test_loader = build_test_loader(config)
    logger.info(f"Test samples found: {len(test_loader.dataset)}")
    
    # Evaluate
    logger.info("Running evaluation on test set...")
    accuracy, y_pred, y_true, misclassified = evaluate_test_set(model, test_loader, device)
    
    logger.info(f"==================================================")
    logger.info(f"  Official GTSRB Test Accuracy: {accuracy * 100:.2f}%")
    logger.info(f"  Total Misclassified Images : {len(misclassified)}")
    logger.info(f"==================================================")
    
    # Get class names
    class_names = [f"{i:02d}" for i in range(43)]
    
    # Save outputs to results/ directory
    results_dir = os.path.join(project_root, "results")
    
    # Confusion matrix
    cm_path = os.path.join(results_dir, "confusion_matrix", f"{exp_name}_test_cm.png")
    logger.info(f"Saving test confusion matrix to: {cm_path}")
    plot_confusion_matrix(y_true, y_pred, class_names, cm_path)
    
    # Classification report
    report_path = os.path.join(results_dir, "metrics", f"{exp_name}_test_report.json")
    logger.info(f"Saving test classification report to: {report_path}")
    generate_classification_report(y_true, y_pred, class_names, report_path)
    
    # Misclassification JSON
    mis_json_path = os.path.join(results_dir, "misclassified", f"{exp_name}_misclassified.json")
    logger.info(f"Saving misclassified samples list to: {mis_json_path}")
    os.makedirs(os.path.dirname(mis_json_path), exist_ok=True)
    with open(mis_json_path, "w", encoding="utf-8") as f:
        json.dump(misclassified, f, indent=4)
        
    # Save a copy of the results inside the experiment directory for reference
    exp_dir = os.path.join(project_root, "experiments", exp_name)
    if os.path.exists(exp_dir):
        logger.info(f"Mirroring evaluation results to: {exp_dir}")
        plot_confusion_matrix(y_true, y_pred, class_names, os.path.join(exp_dir, "test_confusion_matrix.png"))
        with open(os.path.join(exp_dir, "test_classification_report.json"), "w", encoding="utf-8") as f:
            json.dump(generate_classification_report(y_true, y_pred, class_names, os.path.join(exp_dir, "test_dummy_report.json")), f, indent=4)
        with open(os.path.join(exp_dir, "test_misclassified.json"), "w", encoding="utf-8") as f:
            json.dump(misclassified, f, indent=4)
            
    logger.info("Evaluation pipeline completed successfully!")

if __name__ == "__main__":
    main()
