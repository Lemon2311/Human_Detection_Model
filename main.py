from take_photo import capture_photo
from crop import crop_and_save_images
#from human_detection_model import predict_image, train_model


capture_photo("capture")
crop_and_save_images("capture", "cropped", 1400)
