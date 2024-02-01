from ultralytics import YOLO
import keyboard

# Load a pretrained YOLOv8n model
model = YOLO('yolov8n.pt')
model.classes = [0]  

# Define path to the image file
source = 'image.jpg'

while True:
    # Run inference on the source and show results
    results = model.predict(source, show=True)
    # Till q is pressed
    if keyboard.is_pressed('q'):
        break

print(results)