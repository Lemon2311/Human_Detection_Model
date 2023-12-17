import numpy as np
import os
import cv2
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Function to process and load images
def load_and_process_images(path):
    x_data = [] 
    for img in os.listdir(path):
        if img.endswith('png'):
            pic = cv2.imread(os.path.join(path, img))
            pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
            pic = cv2.resize(pic, (80, 80))
            x_data.append([pic])
    return x_data

# Function to save and load numpy arrays
def save_and_load_np_array(data, path, file_name):
    np.save(os.path.join(path, file_name), np.array(data))
    return np.load(os.path.join(path, file_name))

# CNN Model Creation
def create_cnn_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    return model

# Training Function
def train_model(path_1, path_0):
    x_data_1 = load_and_process_images(path_1)
    x_data_0 = load_and_process_images(path_0)

    x_data_1_np = save_and_load_np_array(x_data_1, path_1, 'features.npy')
    x_data_0_np = save_and_load_np_array(x_data_0, path_0, 'features.npy')

    x_data = np.concatenate((x_data_1_np, x_data_0_np), axis=0)
    x_data = x_data.reshape(x_data.shape[0], x_data.shape[-1], x_data.shape[-1], 1)

    y_data_1 = np.ones(np.shape(x_data_1)[0])
    y_data_0 = np.zeros(np.shape(x_data_0)[0])
    y_data = np.concatenate((y_data_1, y_data_0), axis=0)

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    N_in = np.shape(x_train)[-2]
    N_channel = np.shape(x_train)[-1]
    model = create_cnn_model((N_in, N_in, N_channel))
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(x_train, y_train, batch_size=16, epochs=20, verbose=1, validation_data=(x_test, y_test))

    results = model.evaluate(x_test, y_test, batch_size=16)
    print("Test loss, Test accuracy:", results)

    return model, history

# Function for predicting on a single image
def predict_image(model, image_path, image_size=(80, 80)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, image_size)
    img = img.reshape(1, image_size[0], image_size[1], 1)
    img = img / 255.0

    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)
    return predicted_class
