import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model("models/fall_detection_model.h5")

def predict_fall(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(64, 64))  # Resize to match model input size
    img_array = image.img_to_array(img) / 255.0          # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)        # Add batch dimension

    # Make a prediction
    prediction = model.predict(img_array)
    return "Fall Detected" if prediction > 0.5 else "No Fall Detected"

# Test with an image (replace with your test image path)
img_path = "C:/Users/Bhenedix Paul/PycharmProjects/fALL-DETECTION-PROJECT/dataset/images/val/Fall/fall001.jpg"
# Replace with an actual test image path
print(predict_fall(img_path))
