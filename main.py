import os
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models

# Define directories
train_dir = 'boats_dataset/Train'
test_dir = 'boats_dataset/TEST'

# Mapping class names to numeric labels
class_to_label = {}
for i, cls in enumerate(os.listdir(train_dir)):
    class_to_label[cls] = i

# Print class to label mapping
print("Class to label mapping:", class_to_label)

# Define image dimensions
IMG_HEIGHT = 100  # Example height
IMG_WIDTH = 100   # Example width


# Function to load and preprocess images
def load_images(directory):
    image_data = []
    labels = []
    for i, cls in enumerate(os.listdir(directory)):
        class_dir = os.path.join(directory, cls)
        if os.path.isdir(class_dir):  # Check if it's a directory
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                img = Image.open(img_path).resize((IMG_HEIGHT, IMG_WIDTH))
                img = np.array(img) / 255.0  # Normalize pixel values
                image_data.append(img)
                labels.append(class_to_label[cls])
    return np.array(image_data), np.array(labels)


# Load and preprocess training and test images
train_images, train_labels = load_images(train_dir)
test_images, test_labels = load_images(test_dir)

# Function to visualize random images from each class
def visualize(directory):
    plt.figure(figsize=(15, 15))
    num_images_per_class = 9
    for i, cls in enumerate(os.listdir(directory)):
        class_dir = os.path.join(directory, cls)
        img_files = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
        img_names = random.sample(img_files, min(num_images_per_class, len(img_files)))
        for j, img_name in enumerate(img_names):
            img_path = os.path.join(class_dir, img_name)
            try:
                img = Image.open(img_path)
                plt.subplot(len(os.listdir(directory)), num_images_per_class, i * num_images_per_class + j + 1)
                plt.imshow(img)
                plt.title(f"{cls} (Class {class_to_label[cls]})")
                plt.axis('off')
            except Exception as e:
                print(f"Error opening image file: {img_path}")
                print(e)
    plt.tight_layout()
    plt.show()




# Visualize random images from the train set
print("Visualizing random images from the train set:")
visualize(train_dir)

NUM_CLASSES = 10  # Replace 10 with the actual number of classes in your dataset

# Define CNN architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(NUM_CLASSES, activation='softmax')
])


# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
NUM_EPOCHS = 10  # Replace 10 with the desired number of epochs

# Train the model
history = model.fit(train_images, train_labels, epochs=NUM_EPOCHS, validation_split=0.2)


# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print("Test Accuracy:", test_accuracy)
