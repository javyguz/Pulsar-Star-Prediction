# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 21:43:15 2019

@author: 73830
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Flatten, Conv2D, MaxPool2D, Dense, Dropout
from keras.utils import np_utils
import random
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

img_width,img_height = 28,28
num_classes=10
batch_size = 128
num_epochs = 10
(X_train, Y_train),(X_test,Y_test) = mnist.load_data()
X_train_reshaped = mp.expand_dims(X_train, axis = 3)
X_test_reshaped = np.expand_dims(X_test, axis=3)

print('Dimensiones de los datos de entrenamiento X_train:',X_train.shape)
print('Dimensiones de los datos de valicadion X_test:',X_test.shape)
print('Dimensiones de las etiquetas de entrenamiento Y_train:',Y_train.shape)
print('Dimensiones de las etiquetas de entrenamiento Y_test:',Y_test.shape)


X_train_reshaped = X_train.reshape(X_train.shape[0],img_width*img_height)
X_test_reshaped = X_test.reshape(X_test.shape[0],img_width*img_height)

print('Dimensiones de los datos de entrenamiento re-dimensionados: ',X_train_reshaped.shape)
print('Dimensiones de los datos de validación re-dimensionados: ', X_test_reshaped.shape)

Y_train_one_hot = np_utils.to_categorical(Y_train,num_classes)
Y_test_one_hot = np_utils.to_categorical(Y_test,num_classes)

print('Reprsentación de la 1° etiqueta de entrenamiento en formato one-hot: ',Y_train_one_hot[0,:])

X_train_norm = X_train_reshaped.astype('float32')
X_test_norm = X_test_reshaped.astype('float32')

for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(X_train[i,:,:],cmap='gray')
    plt.axis('off')

num_neurons_input = 500
num_neurons_hidden_1 = 250
num_neurons_hidden_2 = 125
prob_drop_out = 0.25

model = Sequential()

model.add(Conv2D(200,
            kernel_size=(3,3),
            activation='relu',
            padding='same', input_shape(img_width=')))

model.add(MaxPool2D(5,5))

model.add(Conv2D(180,
            kernel_size=(3,3),
            activation='relu',
            padding='same'))

model.add(Conv2D(140,
            kernel_size=(3,3),
            activation='relu',
            padding='same'))

model.add(Conv2D(100,
            kernel_size=(3,3),
            activation='relu',
            padding='same'))

model.add(Conv2D(50,
            kernel_size=(3,3),
            activation='relu',
            padding='valid'))


model.add(Flatten())
model.add(Dense(180,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dropout(rate=0.5))

model.add(Dense(10,activation='Softmax'))
model.compile(loss='categorical_crossentropy',optimizer = 'adam', metrics = ['accuracy'])






