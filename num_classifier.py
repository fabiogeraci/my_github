import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import plaidml.keras
plaidml.keras.install_backend()
import tensorflow as tf
print('Tensorflow Version'+' = '+ tf.__version__)
import keras

# import other stuff
from keras import backend as K

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os, cv2
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, roc_auc_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Lambda, LeakyReLU, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

DATADIR = '/home/fabio/Documents/TF_2_Notebooks_and_Data/04-CNNs/images'
CATEGORIES = ['digit_0','digit_1','digit_2', 'digit_3','digit_4','digit_5','digit_6','digit_7','digit_8','digit_9']

dataset = []

def create_training_data():
    for categories in CATEGORIES:
        path = os.path.join(DATADIR, categories)
        n_class = CATEGORIES.index(categories)
        for images in os.listdir(path):
            try:
                image_array = cv2.imread(os.path.join(path,images), cv2.IMREAD_GRAYSCALE)

                new_array = cv2.resize(image_array,(32, 32))
                dataset.append([new_array, n_class])
            except Exception as e:
                pass

create_training_data()

#for train data
features = []
labels = []

for feature, label in dataset:
    features.append(feature)
    labels.append(label)

#train data
feature_reshape = np.array(features).reshape(-1,32,32,1)
ytrain_one_hot = to_categorical(labels, num_classes=10)
print("Shape of feature_reshape {} Shape of ytrain_one_hot {}".format(feature_reshape.shape,ytrain_one_hot.shape))

x_train, x_valid, y_train, y_valid = train_test_split(feature_reshape, ytrain_one_hot,
                                                  test_size=.25, random_state=0,
                                                  stratify=ytrain_one_hot)

model = Sequential()
model.add(Conv2D(64, (3, 3), padding='same', input_shape=(32, 32, 1)))
model.add(BatchNormalization(momentum=0.5, epsilon=1e-5, gamma_initializer="uniform"))
model.add(LeakyReLU(alpha=0.1))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer="uniform"))
model.add(LeakyReLU(alpha=0.1))

model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(BatchNormalization(momentum=0.2, epsilon=1e-5, gamma_initializer="uniform"))
model.add(LeakyReLU(alpha=0.1))
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer="uniform"))
model.add(LeakyReLU(alpha=0.1))

model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.2))

model.add(Conv2D(256, (3, 3), padding='same'))
model.add(BatchNormalization(momentum=0.2, epsilon=1e-5, gamma_initializer="uniform"))
model.add(LeakyReLU(alpha=0.1))
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer="uniform"))
model.add(LeakyReLU(alpha=0.1))

model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(256))
model.add(LeakyReLU(alpha=0.1))

model.add(BatchNormalization())
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss="categorical_crossentropy",
              optimizer=RMSprop(lr=2e-3),
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale = 1./255.,
                                   rotation_range = 10,
                                   width_shift_range = 0.25,
                                   height_shift_range = 0.25,
                                   shear_range = 0.1,
                                   zoom_range = 0.25,
                                   horizontal_flip = False)

valid_datagen = ImageDataGenerator(rescale=1./255)

def lr_decay(epoch):#lrv
    return 2e-3 * 0.99 ** epoch

es = EarlyStopping(monitor='val_loss', verbose=1, patience=10)

history = model.fit(train_datagen.flow(np.array(x_train), np.array(y_train), batch_size=32),
                    steps_per_epoch=100,
                    epochs=30,
                    callbacks=[LearningRateScheduler(lr_decay), es],
                    validation_data=valid_datagen.flow(np.array(x_valid),np.array(y_valid)),
                    validation_steps=50, verbose=1)

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(val_accuracy))
plt.plot(epochs, accuracy, 'go', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'go', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='validation loss')
plt.title('training and validation loss')
plt.legend()
plt.show()

predicted = np.argmax(model.predict(x_valid/255), axis=-1)
print(predicted)

a = np.array([np.argmax(y, axis=None, out=None) for y in y_valid])
print(classification_report(predicted, a))

confusion_matrix = confusion_matrix(predicted, a)
plt.figure(figsize=(7,7))
sns.heatmap(confusion_matrix, fmt='.0f', annot=True, linewidths=0.2, linecolor='purple')
plt.xlabel('predicted value')
plt.ylabel('Truth value')
plt.show()