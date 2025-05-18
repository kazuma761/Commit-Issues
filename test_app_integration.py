import tensorflow as tf
import numpy as np
import cv2

# Load the model
model = tf.keras.models.load_model('waste_detection_model.h5')

# Define waste categories
WASTE_CATEGORIES = ['clean', 'waste']

# Define a function to process a frame
def process_frame(frame, model, img_size=(128, 128)):
    # Preprocess the frame
    processed = cv2.resize(frame, img_size)
    processed = processed / 255.0
    prediction = model.predict(np.expand_dims(processed, axis=0))[0]
    
    # Get prediction (binary classification)
    confidence = float(prediction)
    waste_type = WASTE_CATEGORIES[1] if confidence > 0.5 else WASTE_CATEGORIES[0]
    
    return confidence, waste_type

# Simulate a video frame (128x128 RGB)
frame = np.random.rand(128, 128, 3).astype(np.float32)

# Process the frame
confidence, waste_type = process_frame(frame, model)

# Print the results
print(f"Waste Type: {waste_type}, Confidence: {confidence:.2f}")
