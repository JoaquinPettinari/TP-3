import pandas as pd
from decision_tree import DecisionTree
from sklearn.model_selection import train_test_split
from random_forest import RandomForest
import numpy as np
import time
from sklearn import datasets
from utils import obtener_accuracy, obtener_metricas, obtener_precision

atributos = ["Account Balance","Payment Status of Previous Credit","Value Savings/Stocks","Length of current employment","Sex & Marital Status","Guarantors","Creditability"]

TrainingData = pd.read_csv("TrainingCompleto.csv", skiprows=1, header=None, names=atributos)
X_Training = TrainingData.iloc[:, :-1].values
Y_Training = TrainingData.iloc[:, -1].values.reshape(-1,1)

TestData = pd.read_csv("TestCompleto.csv", skiprows=1, header=None, names=atributos)
X_Test = TestData.iloc[:, :-1].values
Y_Test = TestData.iloc[:, -1].values.reshape(-1,1)

arbol_Training = DecisionTree(atributos, min_samples_split=3, max_depth=3)
#arbol_Training.fit(X_Training,Y_Training)

actual_Training = TrainingData.iloc[:, -1].tolist()
#prediccion_Training = arbol_Training.predict(X_Training)

#arbol_Test = DecisionTree(atributos, min_samples_split=3, max_depth=3)
#arbol_Test.fit(X_Test,Y_Test)
#prediccion_Test = arbol_Test.predict(X_Test)

#TP_Training, FP_Training, TN_Training, FN_Training = obtener_metricas(prediccion_Training, actual_Training)
#accuracy_Training = obtener_accuracy(TP_Training, FP_Training, TN_Training, FN_Training)
#precision_Training = obtener_precision(TP_Training, FP_Training)

actual_Test = TestData.iloc[:, -1].tolist()
"""
TP_Test, FP_Test, TN_Test, FN_Test = obtener_metricas(prediccion_Test, actual_Test)
accuracy_Test = obtener_accuracy(TP_Test, FP_Test, TN_Test, FN_Test)
precision_Test = obtener_precision(TP_Test, FP_Test)

print("Training: ")
print("Accuracy: ", accuracy_Training)
print("Precision: ", precision_Training)

print("Test: ")
print("Accuracy: ", accuracy_Test)
print("Precision: ", precision_Test)
"""

random_Forest = RandomForest(atributos, n_trees=6, max_depth=10)
random_Forest.fit(X_Training,Y_Training)

y_prediction = random_Forest.predict(X_Training)
TP_Training, FP_Training, TN_Training, FN_Training = obtener_metricas(y_prediction, actual_Training)
accuracy_Test = obtener_accuracy(TP_Training, FP_Training, TN_Training, FN_Training)
precision_Test = obtener_precision(TP_Training, FP_Training)
print(accuracy_Test)
print(precision_Test)