import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import os
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split



#creating datasets
import cv2
#setting the path to the directory containing the pics
# In this section create your own paths based on where you have stored the dataset
path_home = 'C:/Users/888254/Documents/EXPO/EXPO_2023_2024/EXPO21/code'
path_1 = 'C:/Users/888254/Documents/EXPO/EXPO_2023_2024/EXPO21/code/human_detection_dataset/1'
path_0 = 'C:/Users/888254/Documents/EXPO/EXPO_2023_2024/EXPO21/code/human_detection_dataset/0'

# THIS CREATES NUMPY ARRAYS THAT STORE PHOTOS
#appending the pics to the training data list
#import os
x_data_1 = [] # It contains the picture in the folder 1 (humans are present)
for img in os.listdir(path_1):
    if img.find('png')!=-1:
        pic = cv2.imread(os.path.join(path_1,img))
        pic = cv2.cvtColor(pic,cv2.COLOR_BGR2GRAY)
        pic = cv2.resize(pic,(80,80))
        x_data_1.append([pic])
# and it will save them into an array file called features.npy
np.save(os.path.join(path_1,'features'),np.array(x_data_1))   
#loading the saved file once again and store it in x_data_1_p
x_data_1_np = np.load(os.path.join(path_1,'features.npy')) 

#appending the pics to the training data list
x_data_0 = [] # It contains the picture in the folder 1 (humans are not present)
for img in os.listdir(path_0):
    if img.find('png')!=-1:
        pic = cv2.imread(os.path.join(path_0,img))
        #pic = cv2.cvtColor(pic,cv2.COLOR_BGR2RGB)
        #pic = cv2.cvtColor(pic,cv2.COLOR_BGR2RGB)
        pic = cv2.cvtColor(pic,cv2.COLOR_BGR2GRAY)
        pic = cv2.resize(pic,(80,80))
        x_data_0.append([pic])
# and it will save them into an array file called features.npy
np.save(os.path.join(path_1,'features'),np.array(x_data_0))   
#loading the saved file once again and store it in x_data_0_p
x_data_0_np = np.load(os.path.join(path_1,'features.npy'))  

# YOU CAN PRINT ANY IMAGE STOREd EITHER IN x_data_0_np and x_data_1_np
# THIS IS JUST AN EXAMPLE TO VISUALIZE IMAGES INSIDE DATASET
plt.imshow(x_data_1_np[0,0,:,:])
plt.show()


#This concatenate both images with humans and images without humans
x_data = np.concatenate((x_data_1_np,x_data_0_np), axis=0)
x_data = x_data.reshape(x_data.shape[0],x_data.shape[-1],x_data.shape[-1],1)

#This generates the labes, where 1 = Humans and 0 = Non humans
y_data_1 = np.ones(np.shape(x_data_1)[0])
y_data_0 = np.zeros(np.shape(x_data_0)[0])
y_data = np.concatenate((y_data_1,y_data_0), axis=0)

#This generates dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)


# Converts to categorical data (one-hot encoding)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Establish image shapes according to data sizes
N_in = np.shape(x_train)[-2]
N_channel = np.shape(x_train)[-1]

# Create a CNN model
model = tf.keras.Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(N_in, N_in, N_channel)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))  # Use softmax for multi-class classification

# Define an optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the model
history = model.fit(x=x_train, y=y_train, batch_size=16, epochs=20, verbose=1,
                    validation_data=(x_test, y_test))

# Evaluate the model
results = model.evaluate(x_test, y_test, batch_size=16)
print("Test loss, Test accuracy:", results)

#Test comment 1