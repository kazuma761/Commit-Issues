import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
from sklearn.model_selection import train_test_split
from PIL import Image  # Replacing cv2 with PIL

def load_and_preprocess_data(data_dir, img_size=(128, 128)):
    images = []
    labels = []
    
    # Load images from class O (no waste)
    for img_name in os.listdir(os.path.join(data_dir, 'O')):
        img_path = os.path.join(data_dir, 'O', img_name)
        img = Image.open(img_path)
        img = img.resize(img_size)
        img = np.array(img) / 255.0  # Convert to numpy array and normalize
        images.append(img)
        labels.append(0)  # Class O
    
    # Load images from class R (waste)
    for img_name in os.listdir(os.path.join(data_dir, 'R')):
        img_path = os.path.join(data_dir, 'R', img_name)
        img = Image.open(img_path)
        img = img.resize(img_size)
        img = np.array(img) / 255.0
        images.append(img)
        labels.append(1)  # Class R
    
    return np.array(images), np.array(labels)

def create_model(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

def detect_waste(frame, model, threshold=0.5):
    """
    Detect if waste is present in the image
    Returns: (bool, region) where bool indicates if waste was detected
    and region is the area of interest containing the waste
    """
    if isinstance(frame, str):  # If image path is provided
        frame = Image.open(frame)
    elif isinstance(frame, np.ndarray):  # If numpy array is provided
        frame = Image.fromarray(frame)
    
    # Preprocess image
    img_size = (128, 128)
    processed = frame.resize(img_size)
    processed = np.array(processed) / 255.0
    
    # Make prediction
    prediction = model.predict(np.expand_dims(processed, axis=0))[0][0]
    
    # If waste is detected (prediction > threshold), return True and the full image
    return prediction > threshold, np.array(frame)

def load_waste_detection_model(model_path='waste_detection_model.h5'):
    """Load the trained waste detection model"""
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        return model
    else:
        # If model doesn't exist, create a new one
        input_shape = (128, 128, 3)
        model = create_model(input_shape)
        model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
        print(f"Warning: Model file {model_path} not found. Created a new untrained model.")
        return model

if __name__ == "__main__":
    # Set random seed for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    # Parameters
    IMG_SIZE = (128, 128)
    BATCH_SIZE = 32
    EPOCHS = 10
    
    # Load and preprocess data
    data_dir = '/workspaces/Commit-Issues/sample_data/TRAIN'
    X, y = load_and_preprocess_data(data_dir, IMG_SIZE)
    
    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and compile the model
    model = create_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    
    # Train the model
    history = model.fit(X_train, y_train,
                       batch_size=BATCH_SIZE,
                       epochs=EPOCHS,
                       validation_data=(X_val, y_val))
    
    # Save the model
    model.save('waste_detection_model.h5')
    print("Model saved as 'waste_detection_model.h5'")
