from tensorflow.keras.models import load_model
from take_photo import capture_photo
from human_detection_model import predict_image

# Load the model
loaded_model = load_model('my_model.h5')

#capture_photo("capture")
#crop size needs to be adjusted / might not be needed
#crop_and_save_images("capture", "cropped", 1400) outdated

new_image_class = predict_image(loaded_model, 'capture/image2.jpg')
# Print the predicted class
print("Predicted class:", "Person" if new_image_class[0] == 0 else "Not a Person")
