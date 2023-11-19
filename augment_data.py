from keras.preprocessing.image import ImageDataGenerator
import os

# Define paths to labeled data
data_dir = "C:\Users\JBSR-12-2021\Downloads\Dataset2.zip\Dataset2\FNA"
benign_dir = os.path.join(data_dir, "benign")
malignant_dir = os.path.join(data_dir, "malignant")

# Define the ImageDataGenerator for data augmentation
datagen = ImageDataGenerator(
    rotation_range=40,      # Degree range for random rotations
    width_shift_range=0.2,  # Fraction of total width for random horizontal shifts
    height_shift_range=0.2, # Fraction of total height for random vertical shifts
    shear_range=0.2,        # Shear intensity
    zoom_range=0.2,         # Range for random zoom
    horizontal_flip=True,   # Randomly flip inputs horizontally
    fill_mode='nearest'     # Points outside the boundaries of the input are filled according to the given mode
)

# Function to load and augment images
def load_and_augment_images(directory, label, augmentations_per_image=3):
    images = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith(".png"):  # Adjust file extension as needed
            img_path = os.path.join(directory, filename)
            img = image.load_img(img_path, target_size=(224, 224))  # Adjust target size as needed
            img_array = image.img_to_array(img)
            img_array = img_array.reshape((1,) + img_array.shape)  # Reshape to (1, height, width, channels)
            
            # Apply data augmentation
            for _ in range(augmentations_per_image):
                augmented_img = next(datagen.flow(img_array, batch_size=1))[0]
                images.append(augmented_img)
                labels.append(label)
    
    return images, labels

# Load and augment benign images
augmented_benign_images, augmented_benign_labels = load_and_augment_images(benign_dir, label=0)

# Load and augment malignant images
augmented_malignant_images, augmented_malignant_labels = load_and_augment_images(malignant_dir, label=1)

# Combine original and augmented data
all_images = np.concatenate([benign_images, malignant_images, augmented_benign_images, augmented_malignant_images])
all_labels = np.concatenate([benign_labels, malignant_labels, augmented_benign_labels, augmented_malignant_labels])

# Convert labels to one-hot encoding
all_labels_one_hot = np_utils.to_categorical(all_labels)

# Normalize image data
all_images_normalized = all_images / 255.0
