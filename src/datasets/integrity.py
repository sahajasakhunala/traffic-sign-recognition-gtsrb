import os
from PIL import Image

def verify_dataset(data_dir: str) -> bool:
    """Scans the GTSRB dataset directory to verify its integrity.
    
    Checks that:
    1. The directory exists.
    2. There are 43 class folders (00000 to 00042).
    3. Every folder contains images that can be successfully opened with PIL.
    4. There are no corrupted or empty (0-byte) image files.
    
    Returns:
        True if the dataset is uncorrupted and complete, False otherwise.
    """
    print("=" * 60)
    print("  GTSRB Dataset Integrity Verification")
    print("=" * 60)
    
    if not os.path.exists(data_dir):
        print(f"[ERROR] Directory does not exist: {data_dir}")
        print("Please download the official GTSRB training dataset.")
        print("Download link: https://benchmark.ini.rub.de/gtsrb_dataset.html")
        print("Or from Kaggle: https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign")
        print(f"Extract it so the subfolders 00000 to 00042 are under: {os.path.abspath(data_dir)}")
        return False
    
    subdirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    
    # Check for 43 classes
    expected_folders = [f"{i:05d}" for i in range(43)]
    missing_folders = [f for f in expected_folders if f not in subdirs]
    
    if missing_folders:
        print(f"[WARNING] Missing class folders: {missing_folders}")
        print(f"Found {len(subdirs)} folders instead of 43.")
    
    total_images = 0
    corrupted_files = []
    class_distribution = {}
    
    for folder in subdirs:
        # Check if the folder is named correctly (numeric)
        try:
            class_id = int(folder)
        except ValueError:
            # Skip non-numeric folders (e.g. metadata or CSV folders)
            continue
            
        folder_path = os.path.join(data_dir, folder)
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.ppm', '.png', '.jpg', '.jpeg'))]
        class_distribution[class_id] = len(files)
        
        for file in files:
            file_path = os.path.join(folder_path, file)
            total_images += 1
            
            # Check file size
            if os.path.getsize(file_path) == 0:
                print(f"[ERROR] Empty file found: {file_path}")
                corrupted_files.append(file_path)
                continue
                
            # Try to open and verify image
            try:
                with Image.open(file_path) as img:
                    img.verify()  # verify integrity
            except Exception as e:
                print(f"[ERROR] Corrupted image found: {file_path} - {e}")
                corrupted_files.append(file_path)
                
    print(f"[INFO] Scanned folders      : {len(class_distribution)}")
    print(f"[INFO] Total images checked  : {total_images:,}")
    
    if corrupted_files:
        print(f"[ERROR] Found {len(corrupted_files)} empty or corrupted image files.")
        return False
        
    if len(class_distribution) < 43:
        print("[WARNING] Verification incomplete: Less than 43 class folders found.")
        return False
        
    print("[SUCCESS] All files are valid and uncorrupted! Ready for training.")
    print("-" * 60)
    print("Class Distribution Summary (Top 5 & Bottom 5):")
    sorted_dist = sorted(class_distribution.items(), key=lambda x: x[1], reverse=True)
    print("  Most represented classes:")
    for cid, count in sorted_dist[:5]:
        print(f"    Class {cid:02d}: {count:,} images")
    print("  Least represented classes:")
    for cid, count in sorted_dist[-5:]:
        print(f"    Class {cid:02d}: {count:,} images")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "../../data/raw"
    verify_dataset(path)
