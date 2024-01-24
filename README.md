# Human Detection Model with Nvidia Jetson Nano

## Overview

This project aims to implement a robust human detection model on the Nvidia Jetson Nano. The primary goal is to create an artificial intelligence algorithm that can accurately detect the presence of humans in images captured by the device's built-in camera. This model is a key component of a larger robotics project, specifically a semi-autonomous vacuum designed to collaborate safely with humans.

## Software Requirements

Ensure you have the following software installed before running the project:

- Python 3.8
- TensorFlow
- OpenCV
# Features

## Image Capture

- Captures a single frame from Nvidia Jetson Nano's built-in webcam.
- Utilizes the `take_photo.py` module.
- Saves captured images in the 'capture' folder.
- Webcam resources are released after capturing.

## TensorFlow Integration

- Leverages the TensorFlow library for implementing the Convolutional Neural Network (CNN).
- Utilizes TensorFlow's high-level Keras API for building and training neural networks.
- Compiles the model using the Adam optimizer and categorical cross-entropy loss function.

## Model Training

- Constructs a Convolutional Neural Network (CNN) using TensorFlow's Keras API.
- Involves data preparation by splitting the dataset into training and validation sets.
- Trains the model to distinguish images with and without humans.
- Monitors training progress and evaluates the model's performance on a validation set.

## Model Evaluation

- Evaluates the trained model on a test dataset.
- Analyzes key metrics, including accuracy, to assess the model's effectiveness in detecting humans.

# Running the Human Detection Model
## Step 1: Visit the GitHub Repository

Visit the [GitHub repository](https://github.com/Lemon2311/Human_Detection_Model).

## Step 2: Download ZIP

1. Click on the green "Code" button located on the right side of the repository.
2. Select "Download ZIP" from the dropdown menu.

## Step 3: Extract the Project

1. Locate the downloaded ZIP file on your computer.
2. Extract the contents using a tool like WinRAR or by using the command:

## Step 4: Capture an Image
```bash
python take_photo.py
```

## Step 5: Train the Model
```bash
python main.py
```
## Step 6: Evaluate the Model
After training, the model is automatically evaluated on a test dataset to assess its performance.

## Step 7: View Results
Explore the 'capture' folder to view the captured images and assess the model's effectiveness in detecting humans.

## Step 8: Enjoy!
