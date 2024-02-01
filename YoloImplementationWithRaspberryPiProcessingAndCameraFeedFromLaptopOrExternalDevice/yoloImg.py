from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('yolov8n.pt')
model.classes = [0]  

# Define path to the image file
source = 'image.PNG'

# Run inference on the source
results = model.predict(source, show=True)

print(results)