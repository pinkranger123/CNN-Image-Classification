# CNN-Image-Classification

# Project 2 - Breast Cancer Image Classification

## Dataset

The project utilizes a dataset located in the "Dataset2" folder, which contains two main subfolders:

- `FNA`: This folder contains labeled images for breast cancer cases.
  - `benign`: Image data for 1074 benign cases.
  - `malignant`: Image data for 650 malignant cases.

## Overview

1. **Data Preprocessing:**
   - Perform preprocessing on the labeled image data inside the 'FNA' files.

2. **Model Training and Validation (CNN):**
   - Train and validate a Convolutional Neural Network (CNN) with the preprocessed images.

3. **Estimate and Plot Training and Validation Metrics:**
   - Estimate and plot the training and validation loss and accuracy functions.

4. **Predictions on Unlabeled Data:**
   - Fit the unlabeled images from the 'test' file to your trained model.
   - Predict whether the images represent benign or malignant cases.

## Objectives

A. **Prediction on Unlabeled Test Data:**
   - The folder “test” inside the "Dataset2" contains 14 unlabeled images.
   - Utilize a CNN (Convolutional Neural Network) or any other deep neural network for training.
   - Predict whether each image represents a benign or malignant case.

B. **Model Evaluation:**
   - Evaluate the model accuracy and loss function.

## Execution Instructions

1. Clone the repository:

   ```bash
   git clone https://github.com/[your-username]/[your-repository].git
   cd [your-repository]

## Install dependencies

pip install -r requirements.txt

## Run the code files:

Preprocessing: python preprocess_data.py
Model Training: python train_model.py
Evaluation and Prediction: CNN_FNA.py

## Directory Structure 

- Dataset2/
  - FNA/
    - benign/
    - malignant/
  - test/
- preprocess data
- train model
- evaluate predict
- requirements 
- README.md



