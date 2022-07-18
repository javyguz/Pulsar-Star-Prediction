# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 11:46:10 2019

@author: 73830
"""

import sklearn
import numpy as np
import matplotlib.pyplot as plt

iris = sklearn.datasets.load_iris()
X = iris.data[:, :4]
x1= iris.data[:, :1]
y = (iris.target != 0) * 1
x1_min, x1_max = X[:,0].min(), X[:,0].max(),
x2_min, x2_max = X[:,1].min(), X[:,1].max(),
X_b =  np.insert(X, 0, 1, axis=1)
def sigmoid(z):
    return (1/(1+np.exp(-z)))

def theta(w, x):
    return w.T.dot(x)
def loss(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

w0 = 1
w1 = 1
w2 = 1
w3 = 1
w4 = 1

w = np.array([w0, w1, w2, w3, w4], dtype='float64')
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

w, j = optimizar_LMS(X_b.T, y, 100000, 0.01,  w)
print("j", j)
preds = sigmoid(theta(w,X_b.T))
print(w)
(preds != y).mean()

array=[preds, y]
print(array)