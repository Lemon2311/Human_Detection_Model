import cv2
import os
import time

# Specify the path to the image you want to check and display
image_path = "runs/detect/predict/image.jpg"

# Interval between checks (in seconds)
check_interval = 1

while True:
    # Check if the image exists at the specified path
    if os.path.exists(image_path):
        # Read and display the image
        image = cv2.imread(image_path)
        if image is not None:
            cv2.imshow('Image', image)
            cv2.waitKey(0)  # Wait for any key to be pressed
            cv2.destroyAllWindows()

            # Optional: Delete or move the image after displaying to avoid repeated displays
            # os.remove(image_path)  # Uncomment to delete the image file after displaying

            break  # Exit the loop after displaying the image
        else:
            print("Image exists but could not be read. Check the file format and permissions.")
    else:
        print(f"No image found at {image_path}. Checking again in {check_interval} seconds...")
    
    # Wait for a bit before checking again
    time.sleep(check_interval)
