import cv2
from PIL import Image
import numpy as np
import time
import threading

def capture_photo(cap):
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        return None
    return frame

def apply_overlay(image, color):
    overlay = Image.new('RGBA', image.size, color)
    return Image.alpha_composite(image.convert('RGBA'), overlay)

# Dummy function that you will replace with your actual function.
# This function simulates taking some time (e.g., 0.5 seconds) to return a value.
def check_condition():
    while True:
        global current_color_overlay, display_text
        # Simulate a delay in function execution, like the ai model taking time to run.
        time.sleep(5)
        #logic underneath to return 1 or 0. Commented logic underneath toggles between 1 and 0 every 5 seconds.
        #if current_color_overlay == base_gray_overlay:
        #    current_color_overlay = green_overlay
        #    display_text = "1"
        #else:
        #    current_color_overlay = base_gray_overlay
        #    display_text = "0"
        current_color_overlay = green_overlay
        display_text = "1"
        #logic above simulates the model returning 1 after a delay of 5 seconds, simulating the model taking time to run.

        #underneath theres an example for changing the output/ui overlay based on model output
        #func being a function that runs the model and returns the output
        #to use this example comment out the logic above and uncomment the logic underneath
        #and add the function func running the model and returning the output

        #if(func==1):
        #    current_color_overlay = green_overlay
        #    display_text = "1"
        #else:
        #    current_color_overlay = base_gray_overlay
        #    display_text = "0"

def main():
    global current_color_overlay, display_text, base_gray_overlay, green_overlay
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    base_gray_overlay = (128, 128, 128, 51)  # RGBA, 20% opacity
    green_overlay = (0, 255, 0, 51)          # RGBA, 20% opacity
    current_color_overlay = base_gray_overlay
    display_text = "0"

    # Start the condition checking function in a separate thread
    condition_thread = threading.Thread(target=check_condition, daemon=True)
    condition_thread.start()

    while True:
        captured_frame = capture_photo(cap)
        if captured_frame is None:
            break
        img = Image.fromarray(cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB))
        img = apply_overlay(img, base_gray_overlay)
        if current_color_overlay:
            img = apply_overlay(img, current_color_overlay)

        overlayed_image = np.array(img)
        overlayed_image_with_text = cv2.cvtColor(overlayed_image, cv2.COLOR_RGB2BGR)
        cv2.putText(overlayed_image_with_text, display_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Image with Overlay', overlayed_image_with_text)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
