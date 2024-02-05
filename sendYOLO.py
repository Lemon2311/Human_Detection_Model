import cv2
import requests
import numpy as np
import time

url='http://192.168.1.239:5000'
upload_url = url+'/upload'
image_url = url+'/image'
hdm_url = url+'/hdm'

useModel='yolo'

def fetch_and_display_image(image_url):
    response = requests.get(image_url, stream=True)
    if response.status_code == 200:
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        cv2.imshow('Image', image)
    else:
        print("Failed to fetch image")

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the image to a format that can be sent through a request (JPEG)
    ret, buffer = cv2.imencode('.jpg', frame)
    if not ret:
        print("Failed to encode frame")
        break

    # Convert to bytes and send as part of a multipart/form-data request
    img_bytes = buffer.tobytes()
    files = {'image': ('image.jpg', img_bytes, 'image/jpeg')}
    
    if(useModel=='yolo'):
        response = requests.post(upload_url, files=files)
        print(f"Image sent, server response: {response.text}")

        # Fetch and display the image from the server
        fetch_and_display_image(image_url)


    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Wait a bit before next capture to not overwhelm the server and to make the loop more manageable
    time.sleep(1)

# Release the capture and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
