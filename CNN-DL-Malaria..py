import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import plaidml.keras
plaidml.keras.install_backend()
import tensorflow as tf
import keras

# import other stuff
from keras import backend as K


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.image import imread

my_data_dir = '/home/fabio/Documents/TF_2_Notebooks_and_Data/DATA/cell_images'

# CONFIRM THAT THIS REPORTS BACK 'test', and 'train'
print(os.listdir(my_data_dir))

test_path = my_data_dir + '/test/'
train_path = my_data_dir + '/train/'

print(os.listdir(test_path))
print(os.listdir(train_path + 'parasitized')[0])

para_cell = train_path + 'parasitized/' + os.listdir(train_path + 'parasitized')[0]

para_img = imread(para_cell)
#plt.imshow(para_img)
#plt.show()

print(para_img.shape)

dim1 = []
dim2 = []
for image_filename in os.listdir(test_path + '/uninfected'):
    img = imread(test_path + '/uninfected' + '/' + image_filename)
    d1, d2, colors = img.shape
    dim1.append(d1)
    dim2.append(d2)

#sns.jointplot(dim1,dim2)
#plt.show()

image_shape = (130,130,3)

#from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator
image_gen = ImageDataGenerator(rotation_range=20, # rotate the image 20 degrees
                               width_shift_range=0.10, # Shift the pic width by a max of 5%
                               height_shift_range=0.10, # Shift the pic height by a max of 5%
                               rescale=1/255, # Rescale the image by normalzing it.
                               shear_range=0.1, # Shear means cutting away part of the image (max 10%)
                               zoom_range=0.1, # Zoom in by 10% max
                               horizontal_flip=True, # Allo horizontal flipping
                               fill_mode='nearest' # Fill in missing pixels with the nearest filled value
                              )

image_gen.flow_from_directory(train_path)
image_gen.flow_from_directory(test_path)

#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D

from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D

#https://stats.stackexchange.com/questions/148139/rules-for-selecting-convolutional-neural-network-hyperparameters
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3,3),input_shape=image_shape, activation='relu',))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3,3),input_shape=image_shape, activation='relu',))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3,3),input_shape=image_shape, activation='relu',))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())


model.add(Dense(128))
model.add(Activation('relu'))

# Dropouts help reduce overfitting by randomly turning neurons off during training.
# Here we say randomly turn off 50% of neurons.
model.add(Dropout(0.5))

# Last layer, remember its binary so we use sigmoid
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

#from tensorflow.keras.callbacks import EarlyStopping
from keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss',patience=2)

batch_size = 16
train_image_gen = image_gen.flow_from_directory(train_path,
                                               target_size=image_shape[:2],
                                                color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='binary')

test_image_gen = image_gen.flow_from_directory(test_path,
                                               target_size=image_shape[:2],
                                               color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='binary',shuffle=False)

train_image_gen.class_indices
import warnings
warnings.filterwarnings('ignore')

results = model.fit_generator(train_image_gen,epochs=20,
                              validation_data=test_image_gen,
                             callbacks=[early_stop])

from tensorflow.keras.models import load_model
model.save('malaria_detector.h5')

losses = pd.DataFrame(model.history.history)
losses[['loss','val_loss']].plot()

model.metrics_names
from tensorflow.keras.preprocessing import image

# https://datascience.stackexchange.com/questions/13894/how-to-get-predictions-with-predict-generator-on-streaming-test-data-in-keras
pred_probabilities = model.predict_generator(test_image_gen)
predictions = pred_probabilities > 0.5
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(test_image_gen.classes,predictions))

my_image = image.load_img(para_cell,target_size=image_shape)
type(my_image)
my_image = image.img_to_array(my_image)
my_image = np.expand_dims(my_image, axis=0)

model.predict(my_image)