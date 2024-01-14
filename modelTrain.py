# Importing required libraries
from tensorflow import keras
import numpy as np
import cv2
import os
import warnings

# Disable TensorFlow warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Defining input and output parameters
output_classes = 2 # number of classes
input_shape = (150, 150, 3) # input image shape

# Loading the dataset
data_dir = "photos"
classes = ["lee_sin", "garen"]
train_data = []

#Convert run photos to JPG
runPhotosFolderPath = "runPhotos/"
runPhotosFolderFiles = os.listdir(runPhotosFolderPath)
for runPhotosFolderFile in runPhotosFolderFiles:
	photoPath = runPhotosFolderPath + runPhotosFolderFile.strip()
	if "jpg" not in runPhotosFolderFile:
		print(photoPath)
		#open image in png format
		png_img = cv2.imread(photoPath)
		  
		runPhotosFolderFileParts = runPhotosFolderFile.split(".")
		runPhotosFolderFileNewName = runPhotosFolderFileParts[0] + ".jpg"
		cv2.imwrite(runPhotosFolderPath + runPhotosFolderFileNewName, png_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
		
		os.remove(photoPath)

# Iterating through the classes and adding images to training data
for i in range(len(classes)):
    path = os.path.join(data_dir, classes[i])
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img))
        img_array = cv2.resize(img_array, (input_shape[0], input_shape[1]))
        train_data.append([img_array, i])

# Shuffling the training data
np.random.shuffle(train_data)

# Splitting the data into training and testing sets
x_train = []
y_train = []
for features, label in train_data:
    x_train.append(features)
    y_train.append(label)

x_train = np.array(x_train).reshape(-1, input_shape[0], input_shape[1], input_shape[2])
y_train = np.array(y_train)

# Normalizing the data
x_train = x_train / 255.0

# Defining the CNN model
model = keras.Sequential([
    keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation="relu", input_shape=input_shape),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation="relu"),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dense(output_classes, activation="softmax")
])

# Compiling the model
model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# Training the model
model.fit(x_train, y_train, epochs=10, batch_size=32)

# Saving the model
model.save('game_classifier.h5')
