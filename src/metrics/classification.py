import os
import json
import numpy as np
from typing import List, Dict, Any

def generate_classification_report(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    output_path: str
) -> Dict[str, Any]:
    """Computes precision, recall, and F1-score for each class and saves a JSON report.
    
    Args:
        y_true: List of ground-truth class indices.
        y_pred: List of predicted class indices.
        class_names: Names of the classes.
        output_path: Path where the JSON report will be saved.
        
    Returns:
        Dict containing class-wise and macro/micro averaged metrics.
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    num_classes = len(class_names)
    
    report = {"classes": {}, "macro_avg": {}, "accuracy": 0.0}
    
    precisions = []
    recalls = []
    f1s = []
    
    total_correct = np.sum(y_true_arr == y_pred_arr)
    accuracy = float(total_correct / len(y_true)) if len(y_true) > 0 else 0.0
    report["accuracy"] = accuracy
    
    for c in range(num_classes):
        tp = np.sum((y_true_arr == c) & (y_pred_arr == c))
        fp = np.sum((y_true_arr != c) & (y_pred_arr == c))
        fn = np.sum((y_true_arr == c) & (y_pred_arr != c))
        support = int(np.sum(y_true_arr == c))
        
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        
        report["classes"][class_names[c]] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": support
        }
        
    # Compute averages
    report["macro_avg"] = {
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "f1-score": float(np.mean(f1s)),
        "support": int(len(y_true))
    }
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
        
    print(f"[INFO] Classification report saved to: {output_path}")
    return report
