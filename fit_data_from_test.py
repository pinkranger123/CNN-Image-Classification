import os
import numpy as np
from keras.preprocessing import image
from keras.models import load_model

# Assume 'model' is your trained CNN model
# Replace 'https://github.com/pinkranger123/CNN-Image-Classification' with the actual path to your trained model file

# Load the trained model
model = load_model('path/to/our/model.h1')

# Define the path to the unlabeled test images
test_folder = 'C:\Users\JBSR-12-2021\Downloads\Dataset2.zip\Dataset2\test'

# Function to load and predict on test images
def predict_test_images(model, test_folder):
    predictions = []

    for filename in os.listdir(test_folder):
        if filename.endswith(".png"):  # Adjust file extension as needed
            img_path = os.path.join(test_folder, filename)
            img = image.load_img(img_path, target_size=(224, 224))  # Adjust target size as needed
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            # Normalize image data
            img_array /= 255.0

            # Predict class probabilities
            prediction = model.predict(img_array)

            # Convert probabilities to class (benign or malignant)
            predicted_class = "benign" if prediction[0][0] < 0.5 else "malignant"

            predictions.append((filename, predicted_class))

    return predictions

# Predict on test images
test_predictions = predict_test_images(model, test_folder)

# Display predictions
for filename, predicted_class in test_predictions:
    print(f"{filename}: Predicted class - {predicted_class}")
