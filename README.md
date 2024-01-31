# Human Detection Model

## Overview

This is a simple ai model that should return either 0 or 1 if people are detected into a picture. The aim is to later implement it on a Nvidia Jetson, Nano or Xavier. The primary goal is to create an artificial intelligence algorithm that can accurately detect the presence of humans in images captured by a raspberry pi camera added to the Jetson. This model is a key component of a larger robotics project, specifically a semi-autonomous vacuum designed to collaborate safely with humans.

## Software Requirements

Ensure you have the following software installed before running the project:

- Python 3.8
- TensorFlow
- OpenCV
# Features

## Image Capture

- Captures a single frame from the camera connected to the Nvidia Jetson.
- Utilizes the `take_photo.py` module.
- Saves captured images in the 'capture' folder.
- Webcam resources are released after capturing.

## TensorFlow Integration

- Leverages the TensorFlow library for implementing a Convolutional Neural Network (CNN).
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

## Step 4: Run the model
Runnning the model is as easy as calling
```bash
python main.py
```

## Step 5: Training the Model
If needed the model can be trained more on a custom dataset by running
```bash
python train.py
```
<br><br><br>
More details will be added to the readMe as the project progresses, right now info might be incomplete.
