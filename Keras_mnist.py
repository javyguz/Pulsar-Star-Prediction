# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 20:58:06 2019

@author: Javier Urrecha
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.utils import np_utils
import random
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
img_width, img_height = 28, 28
num_classes = 10
batch_size = 128
num_epochs = 10
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
print('Dimensiones de los datos de entrenamiento X_train', X_train.shape)
print('Dimensiones de los datos de validación X_test', X_test.shape)
print('Dimensiones de las etiquetas de entrenamiento Y_train', Y_train.shape)
print('Dimensiones de las etiquetas de entrenamiento Y_test', Y_test.shape)

X_train_reshaped = X_train.reshape(X_train.shape[0], img_width*img_height)
X_test_reshaped = X_test.reshape(X_test.shape[0], img_width*img_height)

print('Dimensiones de los datos de entrenamiento re-dimensionados: ', X_train_reshaped.shape)
print('Dimensiones de los datos de validación re-dimensionales: ', X_test_reshaped.shape)

Y_train_one_hot = np_utils.to_categorical(Y_train, num_classes)
Y_test_one_hot = np_utils.to_categorical(Y_test, num_classes)

X_train_norm = X_train_reshaped.astype('float32') /255
X_test_norm = X_test_reshaped.astype('float32') /255

for i in range(9):
    plt.subplot(3, 3, i +1)
    plt.imshow(X_train[i, :, :], cmap = 'gray')
    plt.axis('off')
    
num_neurons_input = 500
num_neurons_hidden_1 = 250
num_neurons_hidden_2 = 125
prob_drop_out= 0.25

#Se crea el modelo como un modelo secuencial
model = Sequential()

#Se añade la capa de entradas
model.add(Dense(units = num_neurons_input,
                input_dim = img_width*img_height,
                activation = 'relu',
                use_bias=True,
                kernel_initializer = 'glorot_uniform'))

#Se añade un regularizador para evitar sobreajuste de curvas
model.add(Dropout(prob_drop_out))
model.add(Dropout(prob_drop_out))
#Se añade la 2da capa oculta

#Se añade la 1ra capa oculta
model.add(Dense(units = num_neurons_hidden_1,
                activation = 'relu',
                use_bias = True,
                kernel_initializer='glorot_uniform'))

#Se añade un regularizador para evitar el sobreajuste de curvas
model.add(Dense(units = num_neurons_hidden_2,
                activation = 'relu',
                use_bias = True,
                kernel_initializer='glorot_uniform'))
#Se añade la capa de salida
model.add(Dense(units=num_classes, activation = 'softmax'))

model.compile(loss='categorical_crossentropy', optimizer= 'sgd', metrics = ['accuracy'])

history = model.fit(X_train_norm,
                    Y_train_one_hot,
                    batch_size = batch_size,
                   epochs = num_epochs,
                   verbose = 1,
                   validation_data = (X_test_norm, Y_test_one_hot))

#Summarize history for accuracy
plt.figure(1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Precisión del modelo')
plt.ylabel('Precisión en %')
plt.xlabel('Iteración')
plt.legend(['datos de entrenamiento', 'datos de validación'], loc='upper left')
plt.show()

random_index = random.randint(0, np.round(X_test_norm.shape[0]/2))
predictions = model.predict_classes(X_test_norm[random_index:random_index+9])
plt.figure(figsize=(8,8)) #Se crea la figura para graficar
i=0
for i in range(random_index, random_index+9):
        plt.subplot(3, 3, i+1-random_index)
        plt.imshow(X_test[i], cmap='gray')
        plt.gca().get_xaxis().set_ticks([])
        plt.gca().get_yaxis().set_ticks([])
        plt.ylabel('prediction = &d' % predictions[i - random_index], fontsize = 15)
        

        

    
    
    
    
    
    
    
    
    