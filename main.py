import cv2
from imagenes import dibujar_vacas, obtener_conjuntos_de_imagenes
from perceptron import perceptron
from r2 import mostrar_puntos_en_plano, obtener_puntos, obtener_puntos_mal_clasificados, trazar_linea_con_pesos, trazar_linea_inicial
from sklearn import svm
import numpy as np
from utils import obtener_conjuntos_de_datos, obtener_conjuntos_training_test, print_metricas, separar_conjunto


cantidad_puntos = 20
print("-------------------")
TP3_1 = obtener_puntos(cantidad_puntos)
trazar_linea_inicial()
mostrar_puntos_en_plano(TP3_1)
print("Perceptron simple, bien calificado")
TP3_1_pesos = perceptron(TP3_1, cantidad_puntos)
trazar_linea_con_pesos(TP3_1_pesos)
mostrar_puntos_en_plano(TP3_1)

print("-------------------")
TP3_2 = obtener_puntos_mal_clasificados(TP3_1)

trazar_linea_inicial()
mostrar_puntos_en_plano(TP3_2)
print("Perceptron simple, mal calificado")
TP3_2_pesos = perceptron(TP3_2, cantidad_puntos)
trazar_linea_con_pesos(TP3_2_pesos)
mostrar_puntos_en_plano(TP3_2)
print("-------------------")
#kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’} or callable, default=’rbf’
svm_TP3_1 = svm.SVC(kernel='poly', C=100)
TP3_1_X_Training, TP3_1_Y_Training, TP3_1_X_Test, TP3_1_Y_Test = obtener_conjuntos_training_test(TP3_1)
svm_TP3_1.fit(TP3_1_X_Training, TP3_1_Y_Training)
pred_TP3_1_Test = svm_TP3_1.predict(TP3_1_X_Test)

print("SVM EN TP3_1: ")
print_metricas(TP3_1_Y_Test, pred_TP3_1_Test, labels=[1,-1])

svm_TP3_2 = svm.SVC(kernel='poly', C=100)
TP3_2_X_Training, TP3_2_Y_Training, TP3_2_X_Test, TP3_2_Y_Test = obtener_conjuntos_training_test(TP3_2)

svm_TP3_2.fit(TP3_2_X_Training, TP3_2_Y_Training)
pred_TP3_2_Test = svm_TP3_2.predict(TP3_2_X_Test)
print("-------------------")
print("SVM EN TP3_2: ")
print_metricas(TP3_2_Y_Test, pred_TP3_2_Test, labels=[1,-1])

print("-------------------")
print("Punto 2")
conjuntos_training_X_Random, conjuntos_training_Y_Random, conjuntos_test_X_Random, conjuntos_test_Y_Random = obtener_conjuntos_de_imagenes()
print("Foto con valores Random")
clf = svm.SVC(kernel='poly', C=1000)
clf.fit(conjuntos_training_X_Random, conjuntos_training_Y_Random)
pred_conjunto_test_entero = clf.predict(conjuntos_test_X_Random)
print_metricas(conjuntos_test_Y_Random, pred_conjunto_test_entero)

print("-------------------")
print("Foto con todos los valores")
conjuntos_training_X_foto_entera, conjuntos_training_Y_foto_entera, conjuntos_test_foto_entera, conjuntos_test_Y_foto_entera = obtener_conjuntos_de_imagenes(con_valores_random=False)
pred_conjunto_test_entero = clf.predict(conjuntos_test_foto_entera)
print_metricas(conjuntos_test_Y_foto_entera, pred_conjunto_test_entero)
print("-------------------")

dibujar_vacas(clf)