import os
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Step 1: Data Preprocessing

# Define paths
train_path = "path/to/cloned/repository/Dataset2/FNA"
test_benign_path = "path/to/cloned/repository/Dataset2/test/benign"
test_malignant_path = "path/to/cloned/repository/Dataset2/test/malignant"

# Image dimensions
img_width, img_height = 150, 150

# Create data generators
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load and preprocess training data
train_set = train_datagen.flow_from_directory(train_path, target_size=(img_width, img_height), batch_size=32, class_mode='binary')

# Load and preprocess test data
test_benign_set = test_datagen.flow_from_directory(test_benign_path, target_size=(img_width, img_height), batch_size=32, class_mode='binary')
test_malignant_set = test_datagen.flow_from_directory(test_malignant_path, target_size=(img_width, img_height), batch_size=32, class_mode='binary')

# Combine both sets for evaluation
test_set = test_datagen.flow_from_directory("path/to/cloned/repository/Dataset2/test", target_size=(img_width, img_height), batch_size=32, class_mode='binary')

# Step 2: Build a Deeper CNN Model

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))  # Dropout for regularization
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 3: Train the Deeper Model

history = model.fit(train_set, steps_per_epoch=len(train_set), epochs=15, validation_data=test_set, validation_steps=len(test_set))

# Rest of the code for plotting, evaluation, and prediction remains the same.
