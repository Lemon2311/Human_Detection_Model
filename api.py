from flask import Flask, request, send_from_directory, send_file
from flask_cors import CORS
import os
import shutil
import cv2

from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('yolov8n.pt') 

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

IMAGE_PATH = 'runs/detect/predict/image.jpg'

# Base directory to delete
BASE_DIR = 'runs/'

# Ensure this path ends with your target directory to avoid deleting unintended content
TARGET_DIR = 'runs'

# Function to delete the target directory and its contents
def delete_directory(path):
    try:
        shutil.rmtree(path)
        print(f"Deleted directory {path}")
    except FileNotFoundError:
        print(f"Directory {path} not found.")

delete_directory(TARGET_DIR)        

# Ensure there's a directory to save received images
IMAGES_DIR = "./"
if not os.path.exists(IMAGES_DIR):
    os.makedirs(IMAGES_DIR)

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return "No image part", 400
    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400
    if file:
        # Run inference on the source
        # You might want to add a more secure way to save files
        # For example, validating filenames or generating a unique name for each file
        filepath = os.path.join(IMAGES_DIR, file.filename)
        file.save(filepath)
        results = model.predict(file.filename, save=True, classes=0) #classes=0 if only people recog is desired

        return "Image received and saved", 200


@app.route('/image')
def get_image():
    # Attempt to serve the image file directly
    try:
        return send_file(IMAGE_PATH, mimetype='image/jpeg')
    except FileNotFoundError:
        return "Image not found", 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

