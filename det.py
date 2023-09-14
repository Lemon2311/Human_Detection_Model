import cv2
import numpy as np

# Load the pre-trained OpenPose model (change the paths to your own)
net = cv2.dnn.readNetFromTensorflow('path/to/pose_deploy_linevec.prototxt', 'path/to/pose_iter_XXXXXX.caffemodel')

# Define the body part keypoint mapping
body_parts = {
    0: "Nose",
    1: "Neck",
    2: "Right Shoulder",
    3: "Right Elbow",
    4: "Right Wrist",
    5: "Left Shoulder",
    6: "Left Elbow",
    7: "Left Wrist",
    # ... Add more keypoints as needed
}

# Set the camera source (usually 0 for the built-in camera)
camera_source = 0

# Initialize the camera
cap = cv2.VideoCapture(camera_source)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Prepare the input image for OpenPose
    blob = cv2.dnn.blobFromImage(frame, 1.0, (368, 368), (127.5, 127.5, 127.5), swapRB=True, crop=False)
    
    # Set the input image to the network
    net.setInput(blob)
    
    # Run forward pass to get the keypoints
    output = net.forward()
    
    # Get the height and width of the frame
    frame_height, frame_width, _ = frame.shape
    
    # Iterate through the detected body parts
    for i in range(len(body_parts)):
        prob_map = output[0, i, :, :]  # Probability map of the body part
        
        # Find the local maxima in the probability map
        min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)
        
        # Threshold the probability to filter out weak detections
        if prob > 0.1:
            x = int((frame_width * point[0]) / output.shape[3])
            y = int((frame_height * point[1]) / output.shape[2])
            cv2.circle(frame, (x, y), 5, (0, 255, 255), thickness=-1)
            cv2.putText(frame, body_parts[i], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    
    # Display the frame
    cv2.imshow('Body Part Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
