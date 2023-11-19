import os
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Step 1: Data Preprocessing

# Define paths
train_path = "C:\Users\JBSR-12-2021\Downloads\Dataset2.zip\Dataset2\FNA" #path/to/cloned/repository/Dataset2/FNA
test_benign_path = "C:\Users\JBSR-12-2021\Downloads\Dataset2.zip\Dataset2\FNA\benign" #path/to/cloned/repository/Dataset2/test/benign
test_malignant_path = "C:\Users\JBSR-12-2021\Downloads\Dataset2.zip\Dataset2\FNA\malignant" #path/to/cloned/repository/Dataset2/test/malignant

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

# Step 2: Build CNN Model

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 3: Train the Model

history = model.fit(train_set, steps_per_epoch=len(train_set), epochs=10, validation_data=test_set, validation_steps=len(test_set))

# Step 4: Plot Training and Validation Loss and Accuracy

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Step 5: Evaluate the Model on Test Data

test_loss, test_acc = model.evaluate(test_set, steps=len(test_set))
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_acc:.4f}')

# Step 6: Predict on Unlabeled Test Data

# Load and preprocess unlabeled test data
unlabeled_path = "path/to/cloned/repository/Dataset2/test"
unlabeled_datagen = ImageDataGenerator(rescale=1./255)
unlabeled_set = unlabeled_datagen.flow_from_directory(unlabeled_path, target_size=(img_width, img_height), batch_size=1, class_mode=None, shuffle=False)

# Predict on unlabeled test data
predictions = model.predict(unlabeled_set, steps=len(unlabeled_set))

# Convert probabilities to labels (0: benign, 1: malignant)
predicted_labels = (predictions > 0.5).astype(int)

# Display predictions
for i, label in enumerate(predicted_labels):
    print(f'Image {i+1}: {"Malignant" if label == 1 else "Benign"}')

