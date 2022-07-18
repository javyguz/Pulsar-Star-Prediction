# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 11:41:56 2019

@author: Javier Urrecha
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek

df = pd.read_csv('pulsar_stars.csv/pulsar_stars.csv')
print(df.shape)

train , test = train_test_split(data,test_size = .3,random_state = 100)
train_X = train[[x for x in train.columns if x not in ["target_class"]]]
train_Y = train[["target_class"]]
test_X  = test[[x for x in test.columns if x not in ["target_class"]]]
test_Y  = test[["target_class"]]

target_count = train_Y.target_class.value_counts()
print('Class 0:', target_count[0])
print('Class 1:', target_count[1])
print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')
target_count.plot(kind='bar', title='Count (target)');
smote_enn = SMOTEENN(random_state=0)
train_X_res, train_Y_res = smote_enn.fit_resample(train_X, train_Y)

smote_tomek = SMOTETomek(random_state=0)
train_X_res, train_Y_res = smote_tomek.fit_resample(train_X, train_Y)
train_Y_res=pd.DataFrame(train_Y_res, columns=['target_class'])

target_count_res = train_Y_res.target_class.value_counts()
print('Class 0:', target_count_res[0])
print('Class 1:', target_count_res[1])
print('Proportion:', round(target_count_res[0] / target_count_res[1], 2), ': 1')

target_count_res.plot(kind='bar', title='Count (target)');

X = df.drop(['target_class'], axis=1)
targets = df['target_class'].values
print(targets.shape)
print(X.shape)
print("df", df.shape)
print("X", X.shape)
y= targets
print("y", y.shape)

def sigmoid(z):
    return (1/(1+np.exp(-z)))

def theta(w, x):
    return w.T.dot(x)
def loss(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

w1 = -0.1
w2 = -0.1
w3 = 3
w4 = -0.3
w5 = -0.03
w6 = 0.03
w7 = -0.03
w8 = -0.0001

w = np.array([w1, w2, w3, w4, w5, w6, w7, w8], dtype='float64')
w = w.reshape((w.shape[0], 1))

def dj(x, w, y):
    z=theta(w,x)
    y1=y.reshape(y.shape[0],1)
    e = sigmoid(z).T - y1
    return np.dot(x, (e)) / y.shape[0]

def optimizar_LMS(x, y, num_iter, alpha, w):
    
    # se generan los vectores
    j = np.zeros(num_iter)   
    # Se optimiza el modelo por el numero de iteraciones
    for i in range(num_iter):
       
        # Calculamos la hipótesis
        preds = sigmoid(theta(w,x))
        # Actualizamos los pesos
        w = w - alpha * dj(x, w, y)
        # Guardamos el costo
        j[i] = loss(preds, y)
    return w, j

w, j = optimizar_LMS(X.T, y, 100000, 0.01,  w)
print("j", j)
preds = sigmoid(theta(w,X.T))
print(w)
(preds != y).mean()

array=[preds.T, y]
print(array)



