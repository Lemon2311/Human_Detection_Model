import cv2
import os

def capture_photo(output_folder):
    #folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Initialize webcam capture (0 for default built-in webcam)
    cap = cv2.VideoCapture(0)

    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Error: Couldn't open the camera.")
        exit()

    # Capture a frame from the camera
    ret, frame = cap.read()

    # Check if the frame was captured successfully
    if not ret:
        print("Error: Couldn't capture a frame.")
        exit()

    # Specify the path and filename to save the image in the 'capture' folder
    image_path = os.path.join(output_folder, "image.jpg")

    # Save the captured frame (image) to the 'capture' folder (will overwrite previous image)
    cv2.imwrite(image_path, frame)

    # Release the webcam and close any open windows (if any)
    cap.release()
    cv2.destroyAllWindows()

# Example usage:
#capture_photo("capture")#output_folder