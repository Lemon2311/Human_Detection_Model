from ultralytics import YOLO 

model = YOLO("yolov8n.pt")
 
results = model.predict(source='0',show=True, classes=0)
#conf=0.8 as example can be added as atribute to predict to only output
#predictions with a confidence of 80% or higher

print(results)
