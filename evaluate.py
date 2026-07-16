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
    parser.add_argument(
        "--use_tta",
        action="store_true",
        help="Use Test-Time Augmentation (TTA) during evaluation."
    )
    return parser.parse_args()

@torch.no_grad()
def evaluate_test_set(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    use_tta: bool = False
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
        
        if use_tta:
            import torchvision.transforms.functional as F_t
            # 5-augmentation TTA: original, rotation +/-3 deg, scale +/-5%
            outputs1 = model(images)
            outputs2 = model(F_t.rotate(images, angle=3.0))
            outputs3 = model(F_t.rotate(images, angle=-3.0))
            outputs4 = model(F_t.affine(images, angle=0.0, translate=[0, 0], scale=1.05, shear=0.0))
            outputs5 = model(F_t.affine(images, angle=0.0, translate=[0, 0], scale=0.95, shear=0.0))
            
            probs = (
                torch.softmax(outputs1, dim=1) +
                torch.softmax(outputs2, dim=1) +
                torch.softmax(outputs3, dim=1) +
                torch.softmax(outputs4, dim=1) +
                torch.softmax(outputs5, dim=1)
            ) / 5.0
            _, predicted = probs.max(1)
        else:
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
    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        backup_path = os.path.join("/content/drive/MyDrive/gtsrb_backups/experiments", exp_name, os.path.basename(checkpoint_path))
        if os.path.exists(backup_path):
            logger.info(f"Checkpoint not found at {checkpoint_path}. Found backup in Google Drive! Loading: {backup_path}")
            checkpoint_path = backup_path
        else:
            logger.error(f"Checkpoint not found at: {checkpoint_path}")
            return
            
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Always load normal weights first to initialize all buffers (including BatchNorm running statistics)
    logger.info("Loading normal model weights to initialize parameters and buffers...")
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if args.use_ema and checkpoint.get("ema_state_dict") is not None:
        logger.info("Overwriting model parameters with EMA weights for evaluation (strict=False)...")
        model.load_state_dict(checkpoint["ema_state_dict"], strict=False)
        
    # Build test loader
    logger.info("Loading official test dataset...")
    test_loader = build_test_loader(config)
    logger.info(f"Test samples found: {len(test_loader.dataset)}")
    
    # Evaluate
    logger.info("Running evaluation on test set...")
    accuracy, y_pred, y_true, misclassified = evaluate_test_set(model, test_loader, device, use_tta=args.use_tta)
    
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
            
    # --- AUTOMATIC LEADERBOARD CSV & MD GENERATION ---
    try:
        import datetime
        import subprocess
        import pandas as pd
        
        # Calculate parameter count and theoretical deployment size (float32 = 4 bytes)
        total_params = sum(p.numel() for p in model.parameters())
        model_size_mb = (total_params * 4) / (1024 * 1024)
        
        # Read saved config
        cfg_used = checkpoint.get("config", {})
        train_cfg = cfg_used.get("training", {})
        
        # Collect runtime duration details
        t_seconds = checkpoint.get("training_time_seconds", "N/A")
        t_minutes = checkpoint.get("training_time_minutes", "N/A")
        
        # Check training-time regularizations active
        mixup_active = train_cfg.get("mixup_alpha", 0.0) > 0.0 and train_cfg.get("mixup_prob", 0.0) > 0.0
        cutmix_active = train_cfg.get("cutmix_alpha", 0.0) > 0.0 and train_cfg.get("cutmix_prob", 0.0) > 0.0
        
        # Retrieve Git Commit hash
        try:
            git_commit = subprocess.check_output(["git", "-C", project_root, "rev-parse", "--short", "HEAD"]).decode("utf-8").strip()
        except Exception:
            git_commit = "N/A"
            
        # Retrieve Hardware specifications from checkpoint
        hw = checkpoint.get("hardware", {})
        gpu_name = hw.get("gpu", "N/A")
        cpu_name = hw.get("cpu", "N/A")
        torch_version = hw.get("torch_version", "N/A")
        cuda_version = hw.get("cuda_version", "N/A")
        python_version = hw.get("python_version", "N/A")
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Build CSV leaderboard row representation
        result_row = {
            "experiment": exp_name,
            "model": cfg_used.get("model", {}).get("name", "N/A"),
            "parameters": f"{total_params / 1e6:.2f}M",
            "size_mb": f"{model_size_mb:.1f} MB",
            "image_size": f"{cfg_used.get('data', {}).get('image_size', 'N/A')}x{cfg_used.get('data', {}).get('image_size', 'N/A')}",
            "epochs": train_cfg.get("epochs", "N/A"),
            "best_epoch": checkpoint.get("best_epoch", "N/A"),
            "ema": "✅" if args.use_ema else "❌",
            "mixup": "✅" if mixup_active else "❌",
            "cutmix": "✅" if cutmix_active else "❌",
            "tta": "✅" if args.use_tta else "❌",
            "optimizer": train_cfg.get("optimizer", "N/A"),
            "learning_rate": train_cfg.get("lr", "N/A"),
            "weight_decay": train_cfg.get("weight_decay", "N/A"),
            "official_accuracy": f"{accuracy * 100:.2f}%",
            "errors": len(misclassified),
            "training_time_seconds": f"{t_seconds:.1f}" if isinstance(t_seconds, float) else "N/A",
            "training_time_minutes": f"{t_minutes:.1f}" if isinstance(t_minutes, float) else "N/A",
            "checkpoint": os.path.basename(checkpoint_path),
            "timestamp": timestamp,
            "git_commit": git_commit,
            "gpu": gpu_name,
            "cpu": cpu_name,
            "torch_version": torch_version,
            "cuda_version": cuda_version,
            "python_version": python_version
        }
        
        # Write/Update CSV file
        csv_path = os.path.join(results_dir, "leaderboard.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
        else:
            df = pd.DataFrame(columns=result_row.keys())
            
        # Uniquely identify a run based on experiment, ema activation, and tta active
        match_mask = (df["experiment"] == exp_name) & (df["ema"] == result_row["ema"]) & (df["tta"] == result_row["tta"])
        if match_mask.any():
            # Update row values
            for col, val in result_row.items():
                df.loc[match_mask, col] = val
        else:
            # Append row values
            df = pd.concat([df, pd.DataFrame([result_row])], ignore_index=True)
            
        df.to_csv(csv_path, index=False)
        logger.info(f"Leaderboard CSV updated at: {csv_path}")
        
        # Sort leaderboard by errors ascending (fewer errors = higher rank)
        leaderboard_df = df.copy()
        leaderboard_df["errors"] = pd.to_numeric(leaderboard_df["errors"])
        leaderboard_df = leaderboard_df.sort_values(by="errors", ascending=True)
        
        # Format table for Markdown presentation
        display_cols = [
            "experiment", "model", "parameters", "size_mb", "image_size", 
            "ema", "mixup", "cutmix", "tta", "official_accuracy", "errors", "training_time_minutes"
        ]
        
        col_rename = {
            "experiment": "Experiment",
            "model": "Model",
            "parameters": "Params",
            "size_mb": "Model Size",
            "image_size": "Input Size",
            "ema": "EMA",
            "mixup": "MixUp",
            "cutmix": "CutMix",
            "tta": "TTA",
            "official_accuracy": "Official Test Acc",
            "errors": "Errors",
            "training_time_minutes": "Training Time"
        }
        
        md_df = leaderboard_df[display_cols].rename(columns=col_rename)
        
        # Manual dependency-free Markdown table generator
        cols = list(md_df.columns)
        header = "| " + " | ".join(cols) + " |"
        # Center align icons, scores, and sizes
        separator = "| " + " | ".join([":---:" if c in ["EMA", "MixUp", "CutMix", "TTA", "Official Test Acc", "Errors", "Input Size", "Params", "Model Size"] else "---" for c in cols]) + " |"
        rows = []
        for _, r in md_df.iterrows():
            row_vals = []
            for c in cols:
                val = r[c]
                # Format time string
                if c == "Training Time" and val != "N/A":
                    row_vals.append(f"{val} min")
                else:
                    row_vals.append(str(val))
            rows.append("| " + " | ".join(row_vals) + " |")
            
        md_table = "\n".join([header, separator] + rows)
        
        md_path = os.path.join(results_dir, "leaderboard.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# GTSRB Official Benchmark Leaderboard\n\n")
            f.write("This table is automatically generated after each evaluation run.\n\n")
            f.write(md_table)
            f.write("\n")
        logger.info(f"Leaderboard Markdown updated at: {md_path}")
        
    except Exception as e:
        logger.warning(f"Failed to update leaderboard files: {str(e)}")
        
    logger.info("Evaluation pipeline completed successfully!")

if __name__ == "__main__":
    main()
