from cProfile import label
from imagenes import obtener_conjuntos
from r2 import mostrar_puntos_en_plano, obtener_puntos, obtener_valores_de_columna, perceptron, plot_decision_boundary, trazar_linea
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from utils import obtener_conjuntos_de_datos
import numpy as np

"""
trazar_linea()
puntos = obtener_puntos()
mostrar_puntos_en_plano(puntos)
theta = perceptron(TP3_1_X, TP3_1_Y, 0.5)
plot_decision_boundary(TP3_1_X, theta)
"""

conjuntos_training_X_Random, conjuntos_training_Y_Random, conjuntos_test_X_Random, conjuntos_test_Y_Random = obtener_conjuntos()
print("Con valores Random")
clf = svm.SVC()
clf.fit(conjuntos_training_X_Random, conjuntos_training_Y_Random)
pred_conjunto_test_entero = clf.predict(conjuntos_test_X_Random)

print(confusion_matrix(conjuntos_test_Y_Random, pred_conjunto_test_entero, labels=[0,1,2]))
print(f"Accuracy: {100*accuracy_score(conjuntos_test_Y_Random, pred_conjunto_test_entero)}%")
print(f"Precision: {100*precision_score(conjuntos_test_Y_Random, pred_conjunto_test_entero, average='micro')}%")

print("-------------------")

print("Valores foto entera")
conjuntos_training_X_Entero, conjuntos_training_Y_Entero, conjuntos_test_X_Entero, conjuntos_test_Y_Entero = obtener_conjuntos(con_valores_random=False)
pred_conjunto_test_entero = clf.predict(conjuntos_test_X_Entero)

print(confusion_matrix(conjuntos_test_Y_Entero, pred_conjunto_test_entero, labels=[0,1,2]))
print(f"Accuracy: {100*accuracy_score(conjuntos_test_Y_Entero, pred_conjunto_test_entero)}%")
print(f"Precision: {100*precision_score(conjuntos_test_Y_Entero, pred_conjunto_test_entero, average='micro')}%")
