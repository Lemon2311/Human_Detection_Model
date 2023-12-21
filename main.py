from tenserflow.keras.models import load_model
from take_photo import capture_photo
from crop import crop_and_save_images
from human_detection_model import predict_image

# Load the model
loaded_model = load_model('path/to/my_model.h5')

capture_photo("capture")
crop_and_save_images("capture", "cropped", 1400)#crop size needs to be adjusted

new_image_class = predict_image(loaded_model, 'capture/image.jpg')
# Print the predicted class
print("Predicted class:", "Person" if new_image_class[0] == 0 else "Dog")