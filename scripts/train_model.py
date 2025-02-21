import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Dataset directory
data_dir = r"C:\Users\Bhenedix Paul\PycharmProjects\fALL-DETECTION-PROJECT\dataset\images"


# Data preprocessing
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
train_data = datagen.flow_from_directory(
    data_dir + "/train", target_size=(64, 64), batch_size=32, class_mode='binary', subset='training')
val_data = datagen.flow_from_directory(
    data_dir + "/val", target_size=(64, 64), batch_size=32, class_mode='binary', subset='validation')

# CNN model
model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2,2),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile model
from tensorflow.keras.optimizers import Adam

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Train model
from tensorflow.keras.callbacks import EarlyStopping

# Define EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with EarlyStopping
model.fit(train_data, validation_data=val_data, epochs=30, callbacks=[early_stopping])


# Save model
model.save("C:/Users/Bhenedix Paul/PycharmProjects/fALL-DETECTION-PROJECT/models/fall_detection_model.h5")

print("Model trained and saved successfully!")
