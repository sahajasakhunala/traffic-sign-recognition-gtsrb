import os
import zipfile
import shutil
import tempfile

def extract_zip(zip_path, extract_to):
    print(f"[INFO] Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"[SUCCESS] Extracted to {extract_to}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    raw_dir = os.path.join(project_root, "data", "raw")
    
    train_zip = os.path.join(raw_dir, "GTSRB_Final_Training_Images.zip")
    test_zip = os.path.join(raw_dir, "GTSRB_Final_Test_Images.zip")
    
    if not os.path.exists(train_zip):
        print(f"[ERROR] Could not find: {train_zip}")
        print("Please place the downloaded ZIP files in data/raw/ before running this script.")
        return

    # Create temporary directory for extraction
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract Training ZIp
        extract_zip(train_zip, temp_dir)
        
        # Locate the subfolders (00000 to 00042)
        # Official zip structure: GTSRB/Final_Training/Images/00000...
        src_train_images = os.path.join(temp_dir, "GTSRB", "Final_Training", "Images")
        if not os.path.exists(src_train_images):
            # Try alternate path structures if any
            src_train_images = os.path.join(temp_dir, "Final_Training", "Images")
            
        if os.path.exists(src_train_images):
            print("[INFO] Moving training folders to data/raw/ ...")
            for folder in os.listdir(src_train_images):
                src_folder_path = os.path.join(src_train_images, folder)
                dest_folder_path = os.path.join(raw_dir, folder)
                
                # Delete existing destination folder if any to avoid overwrite issues
                if os.path.exists(dest_folder_path):
                    shutil.rmtree(dest_folder_path)
                shutil.move(src_folder_path, dest_folder_path)
            print("[SUCCESS] Training folders rearranged successfully!")
        else:
            print("[ERROR] Could not find training images directory inside extracted zip.")
            
    # Extract Test Zip if present
    if os.path.exists(test_zip):
        with tempfile.TemporaryDirectory() as temp_dir:
            extract_zip(test_zip, temp_dir)
            
            # Official zip structure: GTSRB/Final_Test/Images/00000.ppm...
            src_test_images = os.path.join(temp_dir, "GTSRB", "Final_Test", "Images")
            if not os.path.exists(src_test_images):
                src_test_images = os.path.join(temp_dir, "Final_Test", "Images")
                
            if os.path.exists(src_test_images):
                dest_test_dir = os.path.join(raw_dir, "test")
                print(f"[INFO] Moving test images to {dest_test_dir} ...")
                if os.path.exists(dest_test_dir):
                    shutil.rmtree(dest_test_dir)
                shutil.move(src_test_images, dest_test_dir)
                print("[SUCCESS] Test folders rearranged successfully!")
            else:
                print("[ERROR] Could not find test images directory inside extracted zip.")

    print("[INFO] Setup complete! You can now run scripts/verify_dataset.py")

if __name__ == "__main__":
    main()
