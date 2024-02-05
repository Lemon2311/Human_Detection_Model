from tensorflow.keras.models import load_model
from human_detection_model import predict_image

def run(path):
    # Load the model
    loaded_model = load_model('my_model.h5')

    #capture_photo("capture")
    #crop size needs to be adjusted / might not be needed
    #crop_and_save_images("capture", "cropped", 1400) outdated

    new_image_class = predict_image(loaded_model, path)
    # Print the predicted class
    return new_image_class

# example: run('capture/image2.jpg')
