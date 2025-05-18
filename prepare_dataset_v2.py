import os
import kaggle
import shutil
import zipfile
import random

# Configure Kaggle
os.environ['KAGGLE_CONFIG_DIR'] = '/workspaces/Commit-Issues'

def download_dataset():
    # Skip download since we already have the data
    print("Dataset already downloaded and available in waste_categories_data/garbage_classification/")
    return True

def prepare_dataset():
    # Define directories and mappings
    base_dir = 'waste_categories_data/garbage_classification'
    target_dir = 'waste_categories_data/prepared'
    
    # Print available directories for debugging
    print("\nChecking available directories:")
    if os.path.exists('waste_categories_data'):
        print("Contents of waste_categories_data:")
        print(os.listdir('waste_categories_data'))
    else:
        print("waste_categories_data directory not found!")
        return False
    
    # Verify base directory exists
    if not os.path.exists(base_dir):
        print(f"\nError: Source directory {base_dir} not found!")
        return False
        
    categories_mapping = {
        'plastic': ['plastic'],
        'glass': ['brown-glass', 'green-glass', 'white-glass'],
        'metal': ['metal'],
        'paper': ['paper', 'cardboard'],
        'other': ['battery', 'biological', 'clothes', 'shoes', 'trash']
    }
    
    # Create target directories
    os.makedirs(target_dir, exist_ok=True)
    for category in categories_mapping.keys():
        os.makedirs(os.path.join(target_dir, category), exist_ok=True)
    
    print("\nProcessing categories:")
    # Sample and copy images
    for target_cat, source_cats in categories_mapping.items():
        print(f"\nProcessing {target_cat}:")
        images = []
        
        # Collect images from all source categories
        for source_cat in source_cats:
            source_dir = os.path.join(base_dir, source_cat)
            if os.path.exists(source_dir):
                current_images = [f for f in os.listdir(source_dir) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                print(f"  Found {len(current_images)} images in {source_cat}")
                images.extend([(source_cat, img) for img in current_images])
            else:
                print(f"  Warning: Directory not found: {source_dir}")
        
        if not images:
            print(f"  No images found for {target_cat}")
            continue
            
        # Randomly sample images
        sample_size = min(50, len(images))
        selected_images = random.sample(images, sample_size)
        
        # Copy selected images
        successful_copies = 0
        for source_cat, img in selected_images:
            source_path = os.path.join(base_dir, source_cat, img)
            if os.path.exists(source_path):
                target_path = os.path.join(target_dir, target_cat, f"{source_cat}_{img}")
                shutil.copy2(source_path, target_path)
                successful_copies += 1
        
        print(f"  {target_cat}: {successful_copies} images prepared")
    
    return True

if __name__ == "__main__":
    print("Downloading dataset...")
    if download_dataset():
        print("\nPreparing dataset...")
        if prepare_dataset():
            print("\nDataset is ready for training!")
            print("You can now run train_waste_classifier.py to train the model")
        else:
            print("\nError: Failed to prepare dataset!")
