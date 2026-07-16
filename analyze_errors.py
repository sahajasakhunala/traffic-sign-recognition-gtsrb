import os
import json
import argparse
import collections

SIGN_NAMES = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing for vehicles over 3.5 metric tons",
    11: "Right-of-way at the next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 metric tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve to the left",
    20: "Dangerous curve to the right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 metric tons"
}

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze GTSRB Misclassifications")
    parser.add_argument(
        "--json",
        type=str,
        default="results/misclassified/exp011_resnet50_champion_misclassified.json",
        help="Path to the misclassified samples JSON file."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    project_root = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(project_root, args.json)
    
    if not os.path.exists(json_path):
        print(f"Misclassification JSON file not found at: {json_path}")
        return
        
    with open(json_path, "r", encoding="utf-8") as f:
        misclassified = json.load(f)
        
    total_errors = len(misclassified)
    print(f"==================================================")
    print(f"  Error Analysis Report for Checkpoint")
    print(f"  Total Misclassified Images: {total_errors}")
    print(f"==================================================\n")
    
    # Track statistics
    class_errors = collections.Counter()
    confusion_pairs = collections.Counter()
    
    for item in misclassified:
        true_label = int(item["true_label"])
        pred_label = int(item["predicted_label"])
        
        class_errors[true_label] += 1
        confusion_pairs[(true_label, pred_label)] += 1
        
    # Sort class errors descending
    sorted_class_errors = class_errors.most_common()
    
    # 1. Output error table
    print("### Error Distribution by Traffic Sign Class:\n")
    print("| Rank | Class ID | Traffic Sign Description | Errors Count | % of Total Errors |")
    print("| :---: | :---: | :--- | :---: | :---: |")
    for idx, (class_id, count) in enumerate(sorted_class_errors, 1):
        desc = SIGN_NAMES.get(class_id, "Unknown")
        percentage = (count / total_errors) * 100
        print(f"| {idx} | {class_id:02d} | {desc} | {count} | {percentage:.1f}% |")
        
    print("\n")
    
    # 2. Output confusion analysis
    sorted_confusions = confusion_pairs.most_common(10)
    print("### Top 10 Most Common Confusion Pairs:\n")
    print("| Rank | True Class -> Predicted Class | True Label Description | Predicted Label Description | Confusions |")
    print("| :---: | :---: | :--- | :--- | :---: |")
    for idx, ((true_id, pred_id), count) in enumerate(sorted_confusions, 1):
        true_desc = SIGN_NAMES.get(true_id, "Unknown")
        pred_desc = SIGN_NAMES.get(pred_id, "Unknown")
        print(f"| {idx} | {true_id:02d} ➔ {pred_id:02d} | {true_desc} | {pred_desc} | {count} |")
        
    print("\n==================================================")

if __name__ == "__main__":
    main()
