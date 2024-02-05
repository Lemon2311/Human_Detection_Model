import numpy as np
import os
import cv2
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf

def define_paths():
    path_home = "C:\\Users\\th2ke\\Music\\EXPO\\Human_Detection_Model-master\\Converted_Data"
    path_1 = os.path.join(path_home, '1')
    path_0 = os.path.join(path_home, '0')
    return path_home, path_1, path_0
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
# Change the output layer to have 1 node and 'sigmoid' activation
    model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Change the number of nodes to 1 and use 'sigmoid' activation
])

    return model

# Training Function modified to take input folder as an argument containing as many datasets as we want
def train_model(input_folder):
    x_data_list = []
    y_data_list = []

    # List all the subdirectories in the input folder
    classes = [d for d in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, d))]
    class_labels = {class_name: idx for idx, class_name in enumerate(classes)}

    for class_name, idx in class_labels.items():
        class_path = os.path.join(input_folder, class_name)
        x_data = load_and_process_images(class_path)
        x_data_np = save_and_load_np_array(x_data, class_path, 'features.npy')

        x_data_list.append(x_data_np)
        y_data_list.append(np.full(len(x_data_np), idx))


    # Define paths
    path_home, path_1, path_0 = define_paths()
    
    # Load and process images for class 1 (humans)
    x_data_1_np = load_and_process_images(path_1)
    
    # Load and process images for class 0 (non-humans)
    x_data_0_np = load_and_process_images(path_0)
    
    # Visualize an image from x_data_1_np (optional)
    #plt.imshow(x_data_1_np[0][:, :, 0])
    #plt.show()
    
    # Concatenate images with humans and images without humans
    x_data = np.concatenate((x_data_1_np, x_data_0_np), axis=0)
    x_data = x_data.reshape(x_data.shape[0], x_data.shape[-1], x_data.shape[-1], 1)
    
    # Generate labels (1 = Humans, 0 = Non-humans)
    y_data_1 = np.ones(np.shape(x_data_1_np)[0])
    y_data_0 = np.zeros(np.shape(x_data_0_np)[0])
    y_data = np.concatenate((y_data_1, y_data_0), axis=0)
    
    # Split into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

    # Create the CNN model
    input_shape = x_train.shape[1:]
    model = create_cnn_model(input_shape)

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(x_train, y_train, batch_size=16, epochs=20, verbose=1, validation_data=(x_test, y_test))

    # Evaluate the model
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

#Example usage:
def example():
    # Step 2: Processing and Loading Images
    # Process images for each category
    person_images = load_and_process_images('data/person')

    # Step 3: Saving and Loading Processed Data
    # Save processed images and load them back
    save_and_load_np_array(person_images, 'processed_data', 'person_images.npy')

    # Step 4: Creating and Training the Model
    # Train the model using the data in 'processed_data' folder
    model, history = train_model('processed_data')

    # Optionally, plot the training history
    plot_training_history(history)

    # Save your model
    model.save('my_model.h5')

    # Step 5: Predicting on a New Image
    # Let's say you have a new image called 'new_image.png'
    new_image_class = predict_image(model, 'path/to/new_image.png')

    # Print the predicted class
    print("Predicted class:", 0 if new_image_class[0] == 0 else 1)

# Plot training history
import matplotlib.pyplot as plt

def plot_training_history(history):
    # Plot training & validation accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.show()
