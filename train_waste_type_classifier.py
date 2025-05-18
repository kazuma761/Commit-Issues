import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
from sklearn.model_selection import train_test_split
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define waste categories
WASTE_CATEGORIES = ['plastic', 'glass', 'metal', 'paper', 'other']
NUM_CATEGORIES = len(WASTE_CATEGORIES)
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 20

def load_and_preprocess_data(data_dir):
    images = []
    labels = []
    
    for class_idx, category in enumerate(WASTE_CATEGORIES):
        category_dir = os.path.join(data_dir, category)
        if not os.path.exists(category_dir):
            print(f"Warning: Directory not found for category {category}")
            continue
            
        print(f"\nLoading {category} images...")
        for img_name in os.listdir(category_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(category_dir, img_name)
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    img = cv2.resize(img, IMG_SIZE)
                    img = img / 255.0  # Normalize pixel values
                    images.append(img)
                    # One-hot encode the labels
                    label = np.zeros(NUM_CATEGORIES)
                    label[class_idx] = 1
                    labels.append(label)
                except Exception as e:
                    print(f"Error loading image {img_path}: {str(e)}")
                    continue
        
        print(f"Loaded {len([l for l in labels if np.argmax(l) == class_idx])} {category} images")
    
    return np.array(images), np.array(labels)

def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CATEGORIES, activation='softmax')
    ])
    return model

def train_model():
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X, y = load_and_preprocess_data('waste_categories_data/prepared')
    
    if len(X) == 0:
        print("No training data found!")
        return
    
    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\nTotal images: {len(X)}")
    print(f"Training images: {len(X_train)}")
    print(f"Validation images: {len(X_val)}")
    
    # Create data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Create and compile the model
    model = create_model()
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train the model
    print("\nTraining the model...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        steps_per_epoch=len(X_train) // BATCH_SIZE
    )
    
    # Save the model
    model.save('waste_type_classifier.h5')
    print("\nModel saved as 'waste_type_classifier.h5'")
    
    return model, history

if __name__ == "__main__":
    train_model()
