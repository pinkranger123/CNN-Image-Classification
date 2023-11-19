import os
from keras.preprocessing import image
from keras.utils import np_utils
import numpy as np

# Define paths to labeled data
data_dir = "C:\Users\JBSR-12-2021\Downloads\Dataset2.zip\Dataset2\FNA"
benign_dir = os.path.join(data_dir, "benign")
malignant_dir = os.path.join(data_dir, "malignant")

# Function to load and preprocess images
def load_and_preprocess_images(directory, label):
    images = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith(".png"):  # Adjust file extension as needed
            img_path = os.path.join(directory, filename)
            img = image.load_img(img_path, target_size=(224, 224))  # Adjust target size as needed
            img_array = image.img_to_array(img)
            images.append(img_array)
            labels.append(label)
    return images, labels

# Load and preprocess benign images
benign_images, benign_labels = load_and_preprocess_images(benign_dir, label=0)  # Assuming 0 for benign

# Load and preprocess malignant images
malignant_images, malignant_labels = load_and_preprocess_images(malignant_dir, label=1)  # Assuming 1 for malignant

# Combine benign and malignant data
all_images = np.concatenate([benign_images, malignant_images])
all_labels = np.concatenate([benign_labels, malignant_labels])

# Convert labels to one-hot encoding
all_labels_one_hot = np_utils.to_categorical(all_labels)

# Normalize image data
all_images_normalized = all_images / 255.0

# Print shapes of preprocessed data
print("Shape of images:", all_images_normalized.shape)
print("Shape of labels:", all_labels_one_hot.shape)
