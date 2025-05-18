import os
import shutil
import random
import requests
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import tensorflow as tf

# Define waste categories and image count per category
WASTE_CATEGORIES = ['plastic', 'glass', 'wood', 'metal', 'paper', 'other']
IMAGES_PER_CATEGORY = 50
IMG_SIZE = (128, 128)

def load_and_preprocess_data(data_dir, img_size=IMG_SIZE):
    images = []
    labels = []
    
    for class_idx, category in enumerate(WASTE_CATEGORIES):
        category_dir = os.path.join(data_dir, category)
        if not os.path.exists(category_dir):
            print(f"Warning: Directory not found for category {category}")
            continue
            
        image_files = os.listdir(category_dir)
        selected_files = image_files[:IMAGES_PER_CATEGORY] if len(image_files) >= IMAGES_PER_CATEGORY else image_files
        
        for img_name in selected_files:
            img_path = os.path.join(category_dir, img_name)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.resize(img, img_size)
                img = img / 255.0  # Normalize pixel values
                images.append(img)
                # One-hot encode the labels
                label = np.zeros(len(WASTE_CATEGORIES))
                label[class_idx] = 1
                labels.append(label)
            except Exception as e:
                print(f"Error loading image {img_path}: {str(e)}")
                continue
    
    return np.array(images), np.array(labels)

def train_model():
    # Create and compile the model
    model = create_multiclass_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X, y = load_and_preprocess_data('waste_categories_data')
    
    if len(X) == 0:
        print("No training data found!")
        return
    
    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )
    
    # Train the model
    print("Training the model...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        validation_data=(X_val, y_val),
        epochs=20,
        steps_per_epoch=len(X_train) // 32
    )
    
    # Save the model
    model.save('waste_type_classifier.h5')
    print("Model saved as 'waste_type_classifier.h5'")
    
    return model, history

if __name__ == "__main__":
    # Create directory for categorized waste images
    os.makedirs('waste_categories_data', exist_ok=True)
    for category in WASTE_CATEGORIES:
        os.makedirs(os.path.join('waste_categories_data', category), exist_ok=True)
        
    print("Please add waste images to their respective category folders in 'waste_categories_data' directory")
    print(f"Each category should have {IMAGES_PER_CATEGORY} images")
    print("\nOnce images are added, the script will train the model")
