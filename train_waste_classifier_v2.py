import os
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from PIL import Image
from waste_type_model import create_multiclass_model, WASTE_CATEGORIES

def load_dataset(data_dir='waste_categories_data/garbage_classification', target_size=(128, 128)):
    images = []
    labels = []
    
    # Load images from each category
    for idx, category in enumerate(WASTE_CATEGORIES):
        category_path = os.path.join(data_dir, category.lower())
        if not os.path.exists(category_path):
            print(f"Warning: Path {category_path} not found")
            continue
            
        print(f"Loading {category} images...")
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            try:
                # Load and preprocess image
                img = Image.open(img_path).convert('RGB')
                img = img.resize(target_size)
                img_array = np.array(img) / 255.0
                
                images.append(img_array)
                labels.append(idx)
            except Exception as e:
                print(f"Error loading {img_path}: {str(e)}")
                
    return np.array(images), np.array(labels)

def main():
    # Set parameters
    IMG_SIZE = (128, 128)
    BATCH_SIZE = 32
    EPOCHS = 10
    
    # Create model
    model = create_multiclass_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Load and preprocess dataset
    X, y = load_dataset(target_size=IMG_SIZE)
    
    if len(X) == 0:
        print("No images found! Please check the dataset directory.")
        return
        
    print(f"Dataset loaded: {len(X)} images")
    
    # Split into train and validation sets
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val)
    )
    
    # Save the trained model
    model.save('waste_type_classifier.h5')
    print("Model saved as waste_type_classifier.h5")
    
    # Print final metrics
    val_acc = history.history['val_accuracy'][-1]
    print(f"Final validation accuracy: {val_acc:.2%}")

if __name__ == "__main__":
    main()
