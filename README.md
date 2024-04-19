# Fashion-Recommendation-System-using-Image-Features

This repository contains code to build a Fashion Recommendation System utilizing image features. The system leverages computer vision and machine learning techniques to analyze fashion items' visual aspects like color, texture, and style and recommend similar or complementary products to users.

## Overview

A Fashion Recommendation System using Image Features aims to provide personalized fashion suggestions to users based on their preferences and browsing history. It analyzes the visual characteristics of fashion items using deep learning models like VGG16, ResNet, or InceptionV3 and recommends similar items from a diverse dataset.

## Features

- Assemble a diverse dataset of fashion items, including various colors, patterns, styles, and categories.
- Implement a preprocessing function to prepare images for feature extraction.
- Choose a pre-trained CNN model (e.g., VGG16) to extract powerful feature representations from images.
- Extract features from fashion images and create a structured dataset of feature vectors.
- Define a metric (e.g., cosine similarity) to measure the similarity between feature vectors.
- Recommend top N fashion items similar to the input image based on feature similarities.
- Visualize the input image and its recommended items for user interactions.

## Requirements

- Python 3.x
- TensorFlow
- Keras
- NumPy
- SciPy
- Matplotlib
- PIL (Python Imaging Library)

## Install dependencies:
pip install -r requirements.txt
# Download the fashion dataset fron the link

https://www.kaggle.com/datasets/santhoshkrishnanr/women-fashion-dataset


# Run the  notebook to preprocess images, extract features, and recommend fashion items

## WORKFLOW

## Fashion Recommendation System using Image Features: Process We Can Follow
Building a fashion recommendation system using image features involves several key steps, leveraging both computer vision and machine learning techniques. Below is a detailed process you can follow to build a fashion recommendation system using image features:

1 . Assemble a diverse dataset of fashion items. This dataset should include a wide variety of
items with different colours, patterns, styles, and categories.
2. Ensure all images are in a consistent format (e.g., JPEG, PNG) and resolution.
3. Implement a preprocessing function to prepare images for feature extraction.
4. Choose a pre-trained CNN model such as VGG16, ResNet, or InceptionV3. These models,
pre-trained on large datasets like ImageNet, are capable of extracting powerful feature
representations from images.
5. Pass each image through the CNN model to extract features.
6. Define a metric for measuring the similarity between feature vectors.
7. Rank the dataset images based on their similarity to the input image and recommend the
top N items that are most similar.
8. Implement a final function that encapsulates the entire process from pre-processing an
input image, extracting features, computing similarities, and outputting recommendations.


So, the process starts with collecting a dataset of images based on fashionable outfits. I found an ideal dataset for this task. You can download the dataset from here.
https://www.kaggle.com/datasets/santhoshkrishnanr/women-fashion-dataset

## Fashion Recommendation System using Image Features with Python

Now, let’s get started with the task of building a fashion recommendation system utilizing image features by importing the necessary Python libraries and the dataset:

Now, we will extract features from all the fashion images:


 here's a breakdown of the provided code with explanations for each part:

```python
# Import necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract features from an image
def extract_features(model, img):
    img_array = keras_image.img_to_array(img)  # Convert image to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess the input according to VGG16 requirements
    features = model.predict(img_array)  # Extract features using the model
    return features.flatten()  # Flatten the feature array

# Function to extract features from all images in a directory
def extract_features_from_all_images(image_paths):
    model = VGG16(weights='imagenet', include_top=False)  # Load pre-trained VGG16 model
    all_features = []
    for img_path in image_paths:
        img = keras_image.load_img(img_path, target_size=(224, 224))  # Load image with resizing
        features = extract_features(model, img)  # Extract features from the image
        all_features.append(features)  # Append the features to the list
    return np.array(all_features)  # Convert the list to a numpy array

# Function to recommend fashion items based on an uploaded image
def recommend_fashion_items_uploaded(input_image_path, all_features, all_image_paths, model, top_n=5):
    # Pre-process the uploaded image
    preprocessed_img = preprocess_uploaded_image(input_image_path)
    
    # Extract features
    features = extract_features(model, preprocessed_img)
    
    # Calculate similarity scores
    similarities = cosine_similarity([features], all_features)
    
    # Get indices of top n similar images
    similar_indices = np.argsort(similarities)[0][-top_n:]
    
    # Filter out the input image index from similar_indices
    similar_indices = [idx for idx in similar_indices if idx != all_image_paths.index(input_image_path)]
    
    # Display the input image
    plt.figure(figsize=(15, 10))
    plt.subplot(1, top_n + 1, 1)
    plt.imshow(keras_image.load_img(input_image_path))
    plt.title("Uploaded Image")
    plt.axis('off')
    
    # Display similar fashion items
    for i, idx in enumerate(similar_indices):
        plt.subplot(1, top_n + 1, i + 2)
        plt.imshow(keras_image.load_img(all_image_paths[idx]))
        plt.title(f"Similar {i + 1}")
        plt.axis('off')
    plt.show()

# Function to preprocess an uploaded image
def preprocess_uploaded_image(file_path):
    img = keras_image.load_img(file_path, target_size=(224, 224))  # Load image with resizing
    return img

# Define the directory containing your images
directory = r'E:\women_fashion\women fashion'

# Get the paths of all images in the directory
def get_image_paths(directory):
    image_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
                image_paths.append(os.path.join(root, file))
    return image_paths

image_paths_list = get_image_paths(directory)

# Extract features from all images
all_features = extract_features_from_all_images(image_paths_list)

# Define the model
model = VGG16(weights='imagenet', include_top=False)

# Example usage:
uploaded_image_path = r'E:\women_fashion\women fashion\dark, elegant, sleeveless dress that reaches down to about mid-calf.jpg'  # Replace with the path to your uploaded image
recommend_fashion_items_uploaded(uploaded_image_path, all_features, image_paths_list, model, top_n=4)
```

Explanation:

1. **Import Libraries**: The code imports necessary libraries such as `os`, `numpy`, `matplotlib.pyplot`, `tensorflow.keras.preprocessing.image`, `VGG16`, `preprocess_input` from `tensorflow.keras.applications.vgg16`, and `cosine_similarity` from `sklearn.metrics.pairwise`.

2. **Extract Features**: The `extract_features` function takes an image and a pre-trained model as input, preprocesses the image, extracts features using the model, and flattens the features to create a feature vector.

3. **Extract Features from All Images**: The `extract_features_from_all_images` function extracts features from all images in a directory by iterating through each image path, loading the image, and calling the `extract_features` function.

4. **Recommend Fashion Items Based on an Uploaded Image**: The `recommend_fashion_items_uploaded` function recommends fashion items based on an uploaded image. It preprocesses the uploaded image, extracts its features, calculates cosine similarity scores between the uploaded image and all images in the dataset, selects the top similar images, and displays them along with the uploaded image.

5. **Preprocess Uploaded Image**: The `preprocess_uploaded_image` function preprocesses an uploaded image by loading it and resizing it to the required input size for the VGG16 model.

6. **Get Image Paths**: The `get_image_paths` function retrieves the paths of all images in a given directory by recursively walking through the directory and filtering files with specific image extensions.

7. **Extract Features from All Images in the Directory**: Features are extracted from all images using the `extract_features_from_all_images` function and stored in the `all_features` variable.

8. **Define the Model**: The VGG16 model is initialized with pre-trained weights and without the top classification layer.

9. **Example Usage**: An example usage of the `recommend_fashion_items_uploaded` function is provided, where an image path is specified, and fashion items similar to the uploaded image are recommended.

This code demonstrates how to build a fashion recommendation system using deep learning-based feature extraction and cosine similarity
