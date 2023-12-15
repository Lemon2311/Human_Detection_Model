import cv2
import os

def crop_and_save_images(input_folder, output_folder, target_size):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        try:
            # Read the image using OpenCV
            img = cv2.imread(input_path)

            # Get the dimensions of the image
            height, width, _ = img.shape

            # Calculate the cropping coordinates
            left = (width - target_size) // 2
            top = (height - target_size) // 2
            right = left + target_size
            bottom = top + target_size

            # Crop the image
            cropped_img = img[top:bottom, left:right]

            # Save the cropped image to the output folder
            cv2.imwrite(output_path, cropped_img)

            print(f"Cropped and saved {filename}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Example usage:
input_folder = "input_photos"
output_folder = "output_photos"
target_size = 700  # Set your desired target size here
crop_and_save_images(input_folder, output_folder, target_size)
