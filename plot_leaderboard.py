import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(project_root, "results", "leaderboard.csv")
    
    if not os.path.exists(csv_path):
        print(f"Leaderboard CSV not found at: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    if df.empty:
        print("Leaderboard CSV is empty.")
        return
        
    # Clean data for plotting
    plot_data = []
    for idx, row in df.iterrows():
        try:
            # Parse accuracy e.g. "99.76%" -> 99.76
            acc_str = str(row["official_accuracy"]).replace("%", "")
            accuracy = float(acc_str)
            
            # Parse size e.g. "90.0 MB" -> 90.0
            size_str = str(row["size_mb"]).replace(" MB", "").replace("MB", "").strip()
            size_mb = float(size_str)
            
            # Parse params e.g. "23.60M" -> 23.6
            param_str = str(row["parameters"]).replace("M", "").replace("m", "").strip()
            params_m = float(param_str)
            
            # Label
            model_name = row["model"]
            tta_suffix = " + TTA" if row["tta"] == "✅" else ""
            ema_suffix = " + EMA" if row["ema"] == "✅" else ""
            label = f"{model_name}{ema_suffix}{tta_suffix}"
            
            plot_data.append({
                "label": label,
                "size_mb": size_mb,
                "accuracy": accuracy,
                "errors": int(row["errors"])
            })
        except Exception as e:
            print(f"Skipping row {idx} due to parsing error: {e}")
            
    if not plot_data:
        print("No valid data points found for plotting.")
        return
        
    pdf = pd.DataFrame(plot_data)
    
    # Create plot
    plt.figure(figsize=(10, 6), dpi=150)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # Draw scatter points
    scatter = plt.scatter(
        pdf["size_mb"], 
        pdf["accuracy"], 
        c=pdf["errors"], 
        cmap="coolwarm_r", 
        s=120, 
        edgecolors="black", 
        linewidths=1.2, 
        zorder=3
    )
    
    # Add labels to points
    for idx, row in pdf.iterrows():
        plt.annotate(
            row["label"], 
            (row["size_mb"], row["accuracy"]),
            textcoords="offset points",
            xytext=(10, -5),
            ha='left',
            fontsize=9,
            weight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3, ec="gray")
        )
        
    # Colorbar for error count
    cbar = plt.colorbar(scatter)
    cbar.set_label("Number of Test Classification Errors (Lower is Better)", fontsize=10, weight='bold')
    
    # Titles and axis formatting
    plt.title("GTSRB Pareto Frontier: Accuracy vs. Model Complexity", fontsize=14, weight='bold', pad=15)
    plt.xlabel("Model Complexity / Size on Disk (MB)", fontsize=11, weight='bold', labelpad=10)
    plt.ylabel("Official Test Set Accuracy (%)", fontsize=11, weight='bold', labelpad=10)
    
    # Grid and padding
    plt.grid(True, linestyle="--", alpha=0.6)
    
    # Dynamic padding based on ranges
    y_min, y_max = pdf["accuracy"].min(), pdf["accuracy"].max()
    x_min, x_max = pdf["size_mb"].min(), pdf["size_mb"].max()
    
    plt.ylim(max(0.0, y_min - 2.0), min(100.0, y_max + 0.2))
    plt.xlim(max(0.0, x_min - 10.0), x_max + 30.0)
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(project_root, "results", "leaderboard_pareto.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Pareto plot successfully saved to: {output_path}")

if __name__ == "__main__":
    main()
