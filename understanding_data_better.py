import os
import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing import image

# Define paths to labeled data
data_dir = "C:\Users\JBSR-12-2021\Downloads\Dataset2.zip\Dataset2\FNA"
benign_dir = os.path.join(data_dir, "benign")
malignant_dir = os.path.join(data_dir, "malignant")

# Function to load and visualize sample images
def visualize_samples(directory, label, num_samples=5):
    plt.figure(figsize=(15, 3))
    plt.suptitle(f"Sample Images - {label.capitalize()}", fontsize=16)
    
    for i, filename in enumerate(os.listdir(directory)[:num_samples]):
        if filename.endswith(".png"):  # Adjust file extension as needed
            img_path = os.path.join(directory, filename)
            img = image.load_img(img_path, target_size=(224, 224))  # Adjust target size as needed
            
            plt.subplot(1, num_samples, i + 1)
            plt.imshow(img)
            plt.title(f"Sample {i + 1}")
            plt.axis('off')
    
    plt.show()

# Visualize sample benign images
visualize_samples(benign_dir, label='benign')

# Visualize sample malignant images
visualize_samples(malignant_dir, label='malignant')

# Function to plot class distribution
def plot_class_distribution(directory):
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith(".png"):  # Adjust file extension as needed
            labels.append(filename.split('_')[1])  # Extract label from filename
    
    plt.figure(figsize=(8, 6))
    sns.countplot(x=labels)
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.show()

# Plot class distribution for benign and malignant images
plot_class_distribution(benign_dir)
plot_class_distribution(malignant_dir)
