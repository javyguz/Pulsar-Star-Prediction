# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 08:44:04 2019

@author: 73830
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN
from sklearn.metrics import confusion_matrix, roc_curve, auc, log_loss
from sklearn import metrics
import sklearn.metrics as skm
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from sklearn.ensemble import RandomForestClassifier
def theta(w, x):
    
    """
        params:
            w: [np_array] a vector of weights with dimensions (nx1), where n represents the number of weights.
            x: [np_array] a matrix of feature variables with dimensions (nxm), 
                where n represents the number of feature variables and m the number of training examples
        returns:
            theta: [np_array] a vector of the inner product of w and x with dimensions (1xm)
    """
        
    return w.T.dot(x)

def sigmoid(z):
    
    """
        params:
            z: [np_array] a vector of the inner product of w and x with dimensions (1xm)
        returns:
            sigmoid: [np_array] a vector of the estimations performed by the model
    """
    
    return (1/(1+np.exp(-z)))

def J(w, x, y):
    """
        params:
            w: [np_array] a vector of weights with dimensions (nx1), where n represents the number of weights.
            x: [np_array] a matrix of feature variables with dimensions (nxm), 
                where n represents the number of feature variables and m the number of training examples
            y: [np_array] a vector of target variables with dimensions (mx1), 
                where m represents the number of target variables
        returns:
            J: [double] the loss function
    """
    
    z=theta(w,x)
    h=sigmoid(z)
    return (1/(x.shape[1]))*np.sum((-y * np.log(h) - (1 - y) * np.log(1 - h)))

def dJ(w, x, y):
    
    """
        params:
            w: [np_array] a vector of weights with dimensions (nx1), where n represents the number of weights.
            x: [np_array] a matrix of feature variables with dimensions (nxm), 
                where n represents the number of feature variables and m the number of training examples
            y: [np_array] a vector of target variables with dimensions (mx1), 
                where m represents the number of target variables
        returns:
            dJ: [double] the derivative of the loss function
    """

    z=theta(w,x)
    e = sigmoid(z).T - y
    
    return (1 / (x.shape[1]))*(np.dot(x, e))

def optimizar_LMS(x_train, y_train, x_test, y_test, num_iter, alpha, w=None):

    """
        We calculate gradient descent for minimizing the MSE to obtain the best linear hypothesis.
            params:
                x_train: [np_array] a matrix of feature variables with dimensions (nxm), 
                    where n represents the number of feature variables and m the number of training examples
                x_test: [np_array] a matrix of feature variables with dimensions (nxm), 
                    where n represents the number of feature variables and m the number of validating examples
                y_train: [np_array] a vector of target variables with dimensions (mx1), 
                    where m represents the number of training target variables
                y_test: [np_array] a vector of target variables with dimensions (mx1), 
                    where m represents the number of validating target variables
                num_iter: [int] an integer indicating the number of iterations of the Gradient Descent algorithm
                alpha: [double] learning rate constant specifying the magnitude update step
                w: [np_array] vector that contains the initial weights to start optimzing the model with dimensions (n x 1)

            return:
                J_train: [np_array] a vector (num_iter x 1) containing all cost function evaluations during training
                J_test: [np_array] a vector (num_iter x 1) containing all cost function evaluations during training
                w: [np_array] a vector of the final optimized weights with dimensions (nx1)
    """

    if w is None:
        # Inicializamos los pesos aleatoriamente
        w = np.random.randn(x_train.shape[0], 1)

    # se generan los vectores
    it = np.arange(0, num_iter)
    J_train = np.zeros(num_iter)
    J_test = np.zeros(num_iter)

    # Se optimiza el modelo por el numero de iteraciones
    for i in range(num_iter):

        # Actualizamos los pesos
        w = w - alpha * dJ(w, x_train, y_train)

        # Guardamos los costo
        J_train[i] = J(w, x_train, y_train)

        J_test[i] = J(w, x_test, y_test)

    return w, J_train, J_test

def proba(w,x):
    """
        params:
            w: [np_array] a vector of weights with dimensions (nx1), where n represents the number of weights.
            x: [np_array] a matrix of feature variables with dimensions (nxm), 
                where n represents the number of feature variables and m the number of training examples
        returns:
            probability: [np_array] a vector with the probabilities of belonging to class 1, with dimensions (1xm)
    """
    proba=np.zeros(x.shape[1])
    n=0
    for i in range(0,x.shape[1]):
        y_hat = sigmoid(theta(w, x[:,i]))
        proba[n]=y_hat
        n+=1

    return proba

def classify(w,x_test,porcentaje):
    """
        params:
            w: [np_array] a vector of weights with dimensions (nx1), where n represents the number of weights.
            x: [np_array] a matrix of feature variables with dimensions (nxm), 
                where n represents the number of feature variables and m the number of training examples
        returns:
            prediction: [np_array] a vector with the predicted classes, with dimensions (1xm)
    """
    prediction = []
    for i in range(0,x_test.shape[1]):
        y_hat = sigmoid(theta(w, x_test[:,i]))           
        if y_hat>porcentaje:
            prediction.append(1)
        else:
            prediction.append(0)
    return prediction

def norm(arr,min,max):
    
    """
        params:
            arr: [np_array] a vector of training/validating examples with dimensions (mx1), 
            where m represents the number training or validating examples
            min: [double] the minimum value of the array
            max: [double] the maximum value of the array  
        returns:
            normalization: [np_array] the normalized vector (mx1)
    """
    return (arr - min) / (max - min)

# Indicamos la dirección y nombre del archivo y se imprime la cabecera
data = pd.read_csv('HTRU_2.csv')
data.head()

#Se utiliza la función train_test_split para separar los datos en 70% de entrenamiento y 30% de validación
data_train , data_test = train_test_split(data,test_size = .3,random_state = 100)

#Se seleccionan las primeras ocho columnas para las características y la última columna para la clase,
#tanto en entrenamiento como en validación
X_train = data_train[[x for x in data_train.columns if x not in ["target_class"]]]
Y_train = data_train[["target_class"]]
X_test  = data_test[[x for x in data_test.columns if x not in ["target_class"]]]
Y_test  = data_test[["target_class"]]

#Se cuentan los datos de la clase para ver si el conjunto de entrenamiento de la clase está balanceado
target_count = Y_train.target_class.value_counts()
print('Class 0:', target_count[0])
print('Class 1:', target_count[1])
print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')
target_count.plot(kind='bar', title='Count (target)');

#Se convierten los conjuntos de entrenamiento en vectores
X_train_list = X_train.values.tolist()
X_train = np.array(X_train_list, dtype='float64')

Y_train_list = Y_train.values.tolist()
Y_train = np.array(Y_train_list, dtype='float64')

#Se utiliza la función SMOTEEEN para balancear los datos de entrenamiento
smote_enn = SMOTEENN(random_state=0)
X_train_res, Y_train_res = smote_enn.fit_resample(X_train, Y_train)

#Se verifica que los datos de entrenamiento estén balanceados
Y_train_res=pd.DataFrame(Y_train_res, columns=['target_class'])
target_count_res = Y_train_res.target_class.value_counts()
print('Class 0:', target_count_res[0])
print('Class 1:', target_count_res[1])
print('Proportion:', round(target_count_res[0] / target_count_res[1], 2), ': 1')

target_count_res.plot(kind='bar', title='Count (target)');

#Se convierten los conjuntos en vectores
Y_train_list = Y_train_res.values.tolist()

y_train = np.array(Y_train_list, dtype='float64')

x_train=X_train_res
#Se convierten los conjuntos de validación en vectores
X_test_list = X_test.values.tolist()
X_test = np.array(X_test_list, dtype='float64')
x_test=X_test


Y_test_list = Y_test.values.tolist()
Y_test = np.array(Y_test_list, dtype='float64')
y_test=Y_test

#Se normalizan los conjuntos de datos de características
x_train_norm = np.zeros((x_train.shape[0],x_train.shape[1]))
x_test_norm = np.zeros((x_test.shape[0],x_test.shape[1]))

for i in range (0,x_train.shape[1]):
    x_train_norm[:,i]=norm(x_train[:,i],min(x_train[:,i]), max(x_train[:,i]))
    
for i in range (0,x_test.shape[1]):
    x_test_norm[:,i]=norm(x_test[:,i],min(x_train[:,i]), max(x_train[:,i]))

print("x_train", x_train.shape)
print("y_train", y_train.shape)

print("y_test" ,y_test.shape)
print("x_test", x_test.shape)

print("x_train_norm" ,x_train_norm.shape)
print("x_test_norm", x_test_norm.shape)

#Encontramos el valor de k que maximice la el f1score

min_k=1
max_k=100
k=np.linspace(min_k, max_k, max_k)
KNN_precision=np.zeros(max_k)
KNN_sensitivity=np.zeros(max_k)
KNN_f1score=np.zeros(max_k)
n=0

for i in range (min_k,max_k+1):
    KNN = KNeighborsClassifier(n_neighbors=i)
    KNN.fit(x_train_norm,y_train)
    y_KNN_pred = KNN.predict(x_test_norm)
    confusion_KNN = confusion_matrix(y_test,y_KNN_pred)
    KNN_precision[n]=(confusion_KNN[1,1]/(confusion_KNN[1,1]+confusion_KNN[0,1]))
    KNN_sensitivity[n]=(confusion_KNN[1,1]/(confusion_KNN[1,1]+confusion_KNN[1,0]))
    KNN_f1score[n]=2*(KNN_precision[n]*KNN_sensitivity[n])/(KNN_precision[n]+KNN_sensitivity[n])
    n+=1
    
print(KNN_f1score)
#Imprime el mejor número de k
k=np.array(k, dtype='int64')
print('Mejor valor de K:', k[KNN_f1score.argmax()])

#Graficamos el f1score en función del valor de k
plt.figure()
plt.scatter(k, KNN_f1score, alpha = 1, color='r')
plt.scatter(k[KNN_f1score.argmax()], KNN_f1score[KNN_f1score.argmax()], alpha = 1, color='b', label='Best K {}'.format('%d' % k[KNN_f1score.argmax()]))
plt.title("f1score in function of K")
plt.xlabel('K')
plt.ylabel('f1score') 
plt.legend()
plt.show()

#Entrenamos el modelo con el mejor valor de K
KNN = KNeighborsClassifier(n_neighbors=k[KNN_f1score.argmax()])
KNN.fit(x_train_norm,y_train)

#Encontramos las predicciones y probabilidad de clase 1 para KNN con el valor de k que máximiza la precisión
y_KNN_pred = KNN.predict(x_test_norm)
y_KNN_pred_proba = KNN.predict_proba(x_test_norm)[:,1]

#Calculamos la matrix de confusión
confusion_KNN = confusion_matrix(y_test,y_KNN_pred)
print('Confusion matrix\n',confusion_KNN)

#Se calculan las métricas de evaluación
accuracy=((confusion_KNN[0,0]+confusion_KNN[1,1])/(confusion_KNN[0,0]+confusion_KNN[0,1]+confusion_KNN[1,0]+confusion_KNN[1,1]))
precision=(confusion_KNN[1,1]/(confusion_KNN[1,1]+confusion_KNN[0,1]))
sensitivity=(confusion_KNN[1,1]/(confusion_KNN[1,1]+confusion_KNN[1,0]))
specificity=(confusion_KNN[0,0]/(confusion_KNN[0,0]+confusion_KNN[0,1]))
f1score=2*(precision*sensitivity)/(precision+sensitivity)

print('\nAcurracy:',accuracy)
print('Precision:',precision)
print('Sensitivity:',sensitivity)
print('Specificity:',specificity)
print('F1-score:',f1score)

#Obtenemos el false positve rate y el true positive rate con librerías
fpr_KNN, tpr_KNN, thresholds_KNN = metrics.roc_curve(y_test, y_KNN_pred_proba)

#Calculamos el área bajo la curva ROC utilizando librerías
auc_KNN=skm.roc_auc_score(y_test,y_KNN_pred_proba)

#Encontramos el umbral óptimo
optimal_thresholds_KNN_idx=np.argmax(tpr_KNN-fpr_KNN)

optimal_thresholds_KNN=thresholds_KNN[optimal_thresholds_KNN_idx]
print('\nOptimal threshold:',optimal_thresholds_KNN)
print('AUC:',auc_KNN)

KNN_loss = log_loss(y_test, y_KNN_pred)
print("\nLog loss:",KNN_loss)

#Graficamos la curva ROC utilizando librerías
plt.figure()
plt.plot(fpr_KNN, tpr_KNN,"r", linewidth=3, label='AUC {}'.format('%.4f' % auc_KNN))
plt.scatter((fpr_KNN[optimal_thresholds_KNN_idx]),tpr_KNN[optimal_thresholds_KNN_idx], alpha = 1, color='b', label='Best Threshold {}'.format('%.4f' % thresholds_KNN[optimal_thresholds_KNN_idx]))
plt.title("ROC Curve\nK Nearest Neighbors\nK=42")
plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity') 
plt.legend()
plt.show()