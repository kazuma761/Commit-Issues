import tensorflow as tf
import numpy as np
import cv2

# Load the model
model = tf.keras.models.load_model('waste_detection_model.h5')

# Create a dummy input image (128x128 RGB)
dummy_image = np.random.rand(128, 128, 3).astype(np.float32)
dummy_image = np.expand_dims(dummy_image, axis=0)  # Add batch dimension

# Normalize the image
dummy_image /= 255.0

# Make a prediction
prediction = model.predict(dummy_image)

# Print the prediction
print("Prediction:", prediction)
