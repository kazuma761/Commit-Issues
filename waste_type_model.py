import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
from sklearn.model_selection import train_test_split
from PIL import Image  # Replacing cv2 with PIL

# Define waste categories
WASTE_CATEGORIES = ['plastic', 'glass', 'wood', 'metal', 'paper', 'other']
NUM_CATEGORIES = len(WASTE_CATEGORIES)

def preprocess_image(image):
    """
    Preprocess image using PIL instead of OpenCV
    """
    if isinstance(image, str):  # If image path is provided
        image = Image.open(image)
    elif isinstance(image, np.ndarray):  # If numpy array is provided
        image = Image.fromarray(image)
    
    # Resize image
    image = image.resize((224, 224))
    # Convert to numpy array and normalize
    image_array = np.array(image) / 255.0
    return image_array

def create_multiclass_model(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
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
        layers.Dense(NUM_CATEGORIES, activation='softmax')  # Multiple categories output
    ])
    
    return model

def predict_waste_type(frame, model, img_size=(128, 128)):
    # Preprocess the frame
    processed = Image.fromarray(frame).resize(img_size)
    processed = np.array(processed) / 255.0
    prediction = model.predict(np.expand_dims(processed, axis=0))[0]
    
    # Get the predicted class and confidence
    predicted_class = np.argmax(prediction)
    confidence = prediction[predicted_class]
    
    return WASTE_CATEGORIES[predicted_class], confidence

def load_waste_type_model(model_path='waste_type_classifier.h5'):
    """Load the trained waste type classification model"""
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        return model
    else:
        # If model doesn't exist, create a new one
        model = create_multiclass_model(input_shape=(128, 128, 3))
        model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        print(f"Warning: Model file {model_path} not found. Created a new untrained model.")
        return model

# Note: This is a placeholder model. To properly implement this,
# you would need a dataset with images labeled by waste type
if __name__ == "__main__":
    IMG_SIZE = (128, 128)
    model = create_multiclass_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    print("Model created. Note: This model needs to be trained with a dataset containing different waste categories.")
