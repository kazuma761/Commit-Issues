import os
import kaggle
import shutil
import zipfile
import random

# Configure Kaggle
os.environ['KAGGLE_CONFIG_DIR'] = '/workspaces/Commit-Issues'

def download_dataset():
    try:
        # Download the dataset
        kaggle.api.dataset_download_files('mostafaabla/garbage-classification', 
                                        path='waste_categories_data',
                                        unzip=True)
        print("Dataset downloaded successfully!")
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        return False
    return True

def prepare_dataset():
    base_dir = 'waste_categories_data/garbage_classification'
    target_dir = 'waste_categories_data/prepared'
    
    # Verify base directory exists
    if not os.path.exists(base_dir):
        print(f"Error: Source directory {base_dir} not found!")
        print("Available directories in waste_categories_data:")
        print(os.listdir('waste_categories_data'))
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
    
    # Sample and copy images
    for target_cat, source_cats in categories_mapping.items():
        images = []
        for source_cat in source_cats:
            source_dir = os.path.join(base_dir, source_cat)
            if os.path.exists(source_dir):
                images.extend([f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        # Randomly sample 50 images (or less if not enough available)
        sample_size = min(50, len(images))
        selected_images = random.sample(images, sample_size)
        
        # Copy selected images
        for img in selected_images:
            for source_cat in source_cats:
                source_path = os.path.join(base_dir, source_cat, img)
                if os.path.exists(source_path):
                    target_path = os.path.join(target_dir, target_cat, img)
                    shutil.copy2(source_path, target_path)
                    break
        
        print(f"{target_cat}: {sample_size} images prepared")

if __name__ == "__main__":
    print("Downloading dataset...")
    if download_dataset():
        print("Preparing dataset...")
        prepare_dataset()
        print("\nDataset is ready for training!")
        print("You can now run train_waste_classifier.py to train the model")
