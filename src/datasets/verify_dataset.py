import os
import hashlib
from PIL import Image
import collections
from typing import Dict, List, Tuple

def get_file_md5(file_path: str) -> str:
    """Computes the MD5 checksum of a file to detect duplicates."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def verify_dataset(data_dir: str) -> bool:
    """Performs comprehensive integrity verification on the GTSRB dataset.
    
    Checks:
    - Missing files and empty folders
    - Corrupted/unreadable images
    - Duplicate image files (via MD5 hashing)
    - Image size statistics (min, max, mean dimensions)
    - Class distribution metrics
    """
    print("=" * 60)
    print("  GTSRB Dataset Integrity Verification")
    print("=" * 60)
    
    if not os.path.exists(data_dir):
        print(f"[ERROR] Directory does not exist: {data_dir}")
        return False
        
    subdirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    
    # Check for empty directory
    if not subdirs:
        print(f"[ERROR] No folders found under: {data_dir}")
        return False
        
    # Check for exactly 43 classes
    expected_folders = [f"{i:05d}" for i in range(43)]
    missing_folders = [f for f in expected_folders if f not in subdirs]
    if missing_folders:
        print(f"[WARNING] Missing class folders: {missing_folders}")
        
    total_images = 0
    corrupted_files = []
    empty_folders = []
    
    md5_to_paths = collections.defaultdict(list)
    widths = []
    heights = []
    class_distribution = {}
    
    for folder in subdirs:
        folder_path = os.path.join(data_dir, folder)
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.ppm', '.png', '.jpg', '.jpeg'))]
        
        if not files:
            empty_folders.append(folder)
            continue
            
        try:
            class_id = int(folder)
            class_distribution[class_id] = len(files)
        except ValueError:
            # Skip non-numeric folders (e.g. metadata or CSV directories)
            continue
            
        for file in files:
            file_path = os.path.join(folder_path, file)
            total_images += 1
            
            # Check empty file size
            if os.path.getsize(file_path) == 0:
                corrupted_files.append((file_path, "Empty file (0 bytes)"))
                continue
                
            # MD5 hashing for duplicate detection
            try:
                file_md5 = get_file_md5(file_path)
                md5_to_paths[file_md5].append(file_path)
            except Exception as e:
                corrupted_files.append((file_path, f"Hashing error: {e}"))
                continue
                
            # Verify image can be opened and check dimensions
            try:
                with Image.open(file_path) as img:
                    img.verify()
                # Re-open to read dimensions (verify() closes file)
                with Image.open(file_path) as img:
                    w, h = img.size
                    widths.append(w)
                    heights.append(h)
            except Exception as e:
                corrupted_files.append((file_path, f"PIL open error: {e}"))
                
    # Detect duplicates
    duplicates_count = 0
    for md5, paths in md5_to_paths.items():
        if len(paths) > 1:
            duplicates_count += (len(paths) - 1)
            
    print(f"[INFO] Total scanned classes : {len(class_distribution)}")
    print(f"[INFO] Total scanned images  : {total_images:,}")
    
    if empty_folders:
        print(f"[WARNING] Found {len(empty_folders)} empty folders: {empty_folders}")
        
    if corrupted_files:
        print(f"[ERROR] Found {len(corrupted_files)} corrupted files:")
        for path, reason in corrupted_files[:10]:
            print(f"  - {path}: {reason}")
        if len(corrupted_files) > 10:
            print(f"  ... and {len(corrupted_files) - 10} more.")
            
    if duplicates_count > 0:
        print(f"[WARNING] Found {duplicates_count} duplicate files in the dataset.")
        
    # Print Image Size Stats
    if widths and heights:
        print("-" * 60)
        print("Image Size Statistics:")
        print(f"  Min Dimensions: {min(widths)}x{min(heights)}")
        print(f"  Max Dimensions: {max(widths)}x{max(heights)}")
        print(f"  Mean Width    : {sum(widths)/len(widths):.1f} px")
        print(f"  Mean Height   : {sum(heights)/len(heights):.1f} px")
        
    # Print Class Distributions
    if class_distribution:
        print("-" * 60)
        print("Class Distribution Summary (Top 3 & Bottom 3):")
        sorted_dist = sorted(class_distribution.items(), key=lambda x: x[1], reverse=True)
        print("  Most represented classes:")
        for cid, count in sorted_dist[:3]:
            print(f"    Class {cid:02d}: {count:,} images")
        print("  Least represented classes:")
        for cid, count in sorted_dist[-3:]:
            print(f"    Class {cid:02d}: {count:,} images")
            
    print("=" * 60)
    
    # Return success status
    return len(corrupted_files) == 0 and len(class_distribution) == 43
