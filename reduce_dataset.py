import os
import shutil
import random

def reduce_dataset(source_dir, max_images_per_category=50):
    """Reduce the number of images in each category folder"""
    if not os.path.exists(source_dir):
        print(f"Directory {source_dir} does not exist")
        return
        
    print(f"Processing directory: {source_dir}")
    
    # Process each category folder
    for category in os.listdir(source_dir):
        category_path = os.path.join(source_dir, category)
        if not os.path.isdir(category_path):
            continue
            
        # List all images in the category
        images = [f for f in os.listdir(category_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(images) <= max_images_per_category:
            continue
            
        # Randomly select images to remove
        images_to_remove = random.sample(images, len(images) - max_images_per_category)
        
        print(f"Reducing {category} from {len(images)} to {max_images_per_category} images")
        
        # Remove excess images
        for img in images_to_remove:
            img_path = os.path.join(category_path, img)
            os.remove(img_path)

def main():
    # Reduce main datasets
    datasets = [
        'waste_categories_data/garbage_classification',
        'waste_categories_data/prepared',
        'waste_data/DATASET/TRAIN',
        'waste_data/DATASET/TEST',
        'sample_data/TRAIN'
    ]
    
    for dataset in datasets:
        try:
            reduce_dataset(dataset)
        except Exception as e:
            print(f"Error processing {dataset}: {str(e)}")

if __name__ == "__main__":
    main()
