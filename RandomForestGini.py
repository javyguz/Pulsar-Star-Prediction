# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 01:48:05 2019

@author: 73830
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN
from sklearn import metrics
import sklearn.metrics as skm
from keras.models import Sequential
from keras.layers.core import Dense
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, auc, log_loss
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

#Se crea el algoritmo utilizando el criterio del Gini impurity y 100 árboles 
RF = RandomForestClassifier(criterion='gini', oob_score=True, random_state=1, n_estimators=123)
RF.fit(X_train, Y_train)

#Se realizan las predicciones, se encuentran las probabilidades y se calcula la matriz de confunsión
y_RF_pred = RF.predict(X_test)
y_RF_pred_proba = RF.predict_proba(X_test)[:,1]
confusion_RF=confusion_matrix(Y_test, y_RF_pred)

print('Confusion matrix\n',confusion_RF)

#Se calculan las métricas de evaluación
accuracy=((confusion_RF[0,0]+confusion_RF[1,1])/(confusion_RF[0,0]+confusion_RF[0,1]+confusion_RF[1,0]+confusion_RF[1,1]))
precision=(confusion_RF[1,1]/(confusion_RF[1,1]+confusion_RF[0,1]))
sensitivity=(confusion_RF[1,1]/(confusion_RF[1,1]+confusion_RF[1,0]))
specificity=(confusion_RF[0,0]/(confusion_RF[0,0]+confusion_RF[0,1]))
f1score=2*(precision*sensitivity)/(precision+sensitivity)

print('\nAcurracy:',accuracy)
print('Precision:',precision)
print('Sensitivity:',sensitivity)
print('Specificity:',specificity)
print('F1-score:',f1score)

#Obtenemos el false positve rate y el true positive rate con librerías
fpr_RF, tpr_RF, thresholds_RF = metrics.roc_curve(Y_test, y_RF_pred_proba)

#Calculamos el área bajo la curva ROC utilizando librerías
auc_RF=skm.roc_auc_score(Y_test,y_RF_pred_proba)

#Encontramos el umbral óptimo
optimal_thresholds_RF_idx=np.argmax(tpr_RF-fpr_RF)

optimal_thresholds_RF=thresholds_RF[optimal_thresholds_RF_idx]
print('\nOptimal threshold:',optimal_thresholds_RF)
print('AUC:',auc_RF)

RF_loss = log_loss(Y_test, y_RF_pred)
print("\nLog loss:", RF_loss)

#Graficamos la curva ROC utilizando librerías
plt.figure()
plt.plot(fpr_RF, tpr_RF,"r", linewidth=3, label='AUC {}'.format('%.4f' % auc_RF))
plt.scatter((fpr_RF[optimal_thresholds_RF_idx]),tpr_RF[optimal_thresholds_RF_idx], alpha = 1, color='b', label='Best Threshold {}'.format('%.4f' % thresholds_RF[optimal_thresholds_RF_idx]))
plt.title("ROC Curve\nRandom Forest\nCriterion:Gini Impurity Trees:100")
plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity') 
plt.legend()
plt.show()

min_trees=64
max_trees=128
trees_range=(max_trees-min_trees)+1
trees = np.linspace(min_trees, max_trees, trees_range)

trees=np.array(trees, dtype='int64')

auc_RF_gini=np.zeros(trees_range)

auc_RF_entropy=np.zeros(trees_range)

n=0

for i in trees:
    RF = RandomForestClassifier(criterion='gini', oob_score=True, random_state=1, n_estimators=i)
    RF.fit(X_train, Y_train)
    y_RF_gini_pred_proba = RF.predict_proba(X_test)[:,1]

    #Calculamos el área bajo la curva ROC utilizando librerías
    auc_RF_gini[n]=skm.roc_auc_score(Y_test,y_RF_gini_pred_proba)
    n+=1
    
n=0

for i in trees:
    RF = RandomForestClassifier(criterion='entropy', oob_score=True, random_state=1, n_estimators=i)
    RF.fit(X_train, Y_train)
    y_RF_entropy_pred_proba = RF.predict_proba(X_test)[:,1]

    #Calculamos el área bajo la curva ROC utilizando librerías
    auc_RF_entropy[n]=skm.roc_auc_score(Y_test,y_RF_gini_pred_proba)
    n+=1

print("\nMax AUC using Gini impurity: ",auc_RF_gini.max())
print("Max AUC using Entropy Information Gain: ",auc_RF_entropy.max())

#Finding the optimal number of trees
print("\nNumber of trees for highest AUC: ",trees[auc_RF_gini.argmax()])

#Running algorithm with the criterion and n_estimators that yield the highest AUC
RF = RandomForestClassifier(criterion='gini', oob_score=True, random_state=1, n_estimators=trees[auc_RF_gini.argmax()])
RF.fit(X_train, Y_train)

y_RF_pred = RF.predict(X_test)
y_RF_pred_proba = RF.predict_proba(X_test)[:,1]
confusion_RF=confusion_matrix(Y_test, y_RF_pred)

print('Confusion matrix\n',confusion_RF)

#Se calculan las métricas de evaluación
accuracy=((confusion_RF[0,0]+confusion_RF[1,1])/(confusion_RF[0,0]+confusion_RF[0,1]+confusion_RF[1,0]+confusion_RF[1,1]))
precision=(confusion_RF[1,1]/(confusion_RF[1,1]+confusion_RF[0,1]))
sensitivity=(confusion_RF[1,1]/(confusion_RF[1,1]+confusion_RF[1,0]))
specificity=(confusion_RF[0,0]/(confusion_RF[0,0]+confusion_RF[0,1]))
f1score=2*(precision*sensitivity)/(precision+sensitivity)

print('\nAcurracy:',accuracy)
print('Precision:',precision)
print('Sensitivity:',sensitivity)
print('Specificity:',specificity)
print('F1-score:',f1score)

#Obtenemos el false positve rate y el true positive rate con librerías
fpr_RF, tpr_RF, thresholds_RF = metrics.roc_curve(Y_test, y_RF_pred_proba)

#Calculamos el área bajo la curva ROC utilizando librerías
auc_RF=skm.roc_auc_score(Y_test,y_RF_pred_proba)

#Encontramos el umbral óptimo
optimal_thresholds_RF_idx=np.argmax(tpr_RF-fpr_RF)

optimal_thresholds_RF=thresholds_RF[optimal_thresholds_RF_idx]
print('\nOptimal threshold:',optimal_thresholds_RF)
print('AUC:',auc_RF)

RF_loss = log_loss(Y_test, y_RF_pred)
print("\nLog loss:",RF_loss)

#Graficamos la curva ROC utilizando librerías
plt.figure()
plt.plot(fpr_RF, tpr_RF,"r", linewidth=3, label='AUC {}'.format('%.4f' % auc_RF))
plt.scatter((fpr_RF[optimal_thresholds_RF_idx]),tpr_RF[optimal_thresholds_RF_idx], alpha = 1, color='b', label='Best Threshold {}'.format('%.4f' % thresholds_RF[optimal_thresholds_RF_idx]))
plt.title("ROC Curve\nRandom Forest\nCriterion:Gini Impurity Trees:{0}".format(trees[auc_RF_gini.argmax()]))
plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity') 
plt.legend()
plt.show()