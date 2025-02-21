import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model
model = tf.keras.models.load_model("C:/Users/Bhenedix Paul/PycharmProjects/fALL-DETECTION-PROJECT/models/fall_detection_model.h5")


# Start webcam
cap = cv2.VideoCapture(0)  # Use your default webcam (index 0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame for prediction
    img = cv2.resize(frame, (64, 64))  # Resize to match model input size
    img = img_to_array(img) / 255.0    # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(img)
    label = "Fall Detected" if prediction > 0.5 else "No Fall"

    # Display result on the video feed
    cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1, color=(0, 0, 255), thickness=2)
    cv2.imshow("Fall Detection", frame)

    # Press 'q' to quit the webcam feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
