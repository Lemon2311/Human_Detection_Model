import cv2
import requests
import numpy as np
import threading
from PIL import Image

# Server URLs
url = 'http://192.168.1.239:5000'
hdm_url = url + '/hdm'

# Global variables
current_overlay_color = (128, 128, 128, 128)  # Gray for "Not a Person"
display_text = "Initializing..."
frame_lock = threading.Lock()
send_frame = True

def apply_overlay(image, color):
    """Applies a semi-transparent color overlay to an image."""
    overlay = Image.new('RGBA', image.size, color)
    return Image.alpha_composite(image.convert('RGBA'), overlay)

def update_overlay_and_text(response_text):
    """Updates the overlay color and text based on the server response."""
    global current_overlay_color, display_text
    if response_text == "Not a Person":
        current_overlay_color = (128, 128, 128, 128)  # Gray
        display_text = "Status: Not a Person | 0"
    elif response_text == "Person":
        current_overlay_color = (0, 255, 0, 128)  # Green
        display_text = "Status: Person | 1"

def send_frame_to_server(frame):
    """Sends a frame to the server and waits for a response."""
    global send_frame
    _, buffer = cv2.imencode('.jpg', frame)
    img_bytes = buffer.tobytes()
    files = {'image': ('image.jpg', img_bytes, 'image/jpeg')}
    response = requests.post(hdm_url, files=files)
    update_overlay_and_text(response.text)
    print(f"Server response: {response.text}")
    send_frame = True  # Allow sending a new frame

def main():
    global send_frame
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Only send a new frame if the previous one has been processed
        if send_frame:
            send_frame = False  # Prevent sending a new frame until the current one is processed
            threading.Thread(target=send_frame_to_server, args=(frame,), daemon=True).start()

        # Apply overlay and text to the displayed frame
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_with_overlay = apply_overlay(img, current_overlay_color)
        img_with_overlay = np.array(img_with_overlay)
        cv2.putText(img_with_overlay, display_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Image with Overlay', img_with_overlay)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Quit on 'q' key press
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
