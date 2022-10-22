import cv2
from imagenes import obtener_conjuntos_de_imagenes
from perceptron import perceptron
from r2 import mostrar_puntos_en_plano, obtener_puntos, obtener_puntos_mal_clasificados, trazar_linea_con_pesos, trazar_linea_inicial
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
import numpy as np
from utils import obtener_conjuntos_de_datos, obtener_conjuntos_training_test, separar_conjunto

cantidad_puntos = 20
print("-------------------")
TP3_1 = obtener_puntos(cantidad_puntos)
trazar_linea_inicial()
mostrar_puntos_en_plano(TP3_1)

TP3_1_pesos = perceptron(TP3_1, cantidad_puntos)
trazar_linea_con_pesos(TP3_1_pesos)
mostrar_puntos_en_plano(TP3_1)

print("-------------------")
TP3_2 = obtener_puntos_mal_clasificados(TP3_1)

trazar_linea_inicial()
mostrar_puntos_en_plano(TP3_2)

TP3_2_pesos = perceptron(TP3_2, cantidad_puntos)
trazar_linea_con_pesos(TP3_2_pesos)
mostrar_puntos_en_plano(TP3_2)

#kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’} or callable, default=’rbf’
svm_TP3_1 = svm.SVC(kernel='poly', C=100)
TP3_1_X_Training, TP3_1_Y_Training, TP3_1_X_Test, TP3_1_Y_Test = obtener_conjuntos_training_test(TP3_1)
svm_TP3_1.fit(TP3_1_X_Training, TP3_1_Y_Training)
pred_TP3_1_Test = svm_TP3_1.predict(TP3_1_X_Test)

print("SVM EN TP3_1: ")
print(confusion_matrix(TP3_1_Y_Test, pred_TP3_1_Test, labels=[1,-1]))
print(f"Accuracy: {100*accuracy_score(TP3_1_Y_Test, pred_TP3_1_Test)}%")
print(f"Precision: {100*precision_score(TP3_1_Y_Test, pred_TP3_1_Test, average='micro')}%")

svm_TP3_2 = svm.SVC(kernel='poly', C=100)
TP3_2_X_Training, TP3_2_Y_Training, TP3_2_X_Test, TP3_2_Y_Test = obtener_conjuntos_training_test(TP3_2)

svm_TP3_2.fit(TP3_2_X_Training, TP3_2_Y_Training)
pred_TP3_2_Test = svm_TP3_2.predict(TP3_2_X_Test)
print("SVM EN TP3_2: ")
print(confusion_matrix(TP3_2_Y_Test, pred_TP3_2_Test, labels=[1,-1]))
print(f"Accuracy: {100*accuracy_score(TP3_2_Y_Test, pred_TP3_2_Test)}%")
print(f"Precision: {100*precision_score(TP3_2_Y_Test, pred_TP3_2_Test, average='micro')}%")

"""
print("-------------------")
print("Punto 2")
conjuntos_training_X_Random, conjuntos_training_Y_Random, conjuntos_test_X_Random, conjuntos_test_Y_Random = obtener_conjuntos_de_imagenes()
print(conjuntos_training_X_Random)
print("Con valores Random")
clf = svm.SVC(kernel='linear', C=1000)
clf.fit(conjuntos_training_X_Random, conjuntos_training_Y_Random)
pred_conjunto_test_entero = clf.predict(conjuntos_test_X_Random)

print(confusion_matrix(conjuntos_test_Y_Random, pred_conjunto_test_entero, labels=[0,1,2]))
print(f"Accuracy: {100*accuracy_score(conjuntos_test_Y_Random, pred_conjunto_test_entero)}%")
print(f"Precision: {100*precision_score(conjuntos_test_Y_Random, pred_conjunto_test_entero, average='micro')}%")

print("-------------------")

print("Valores foto entera")
conjuntos_training_X_Entero, conjuntos_training_Y_Entero, conjuntos_test_X_Entero, conjuntos_test_Y_Entero = obtener_conjuntos_de_imagenes(con_valores_random=False)
pred_conjunto_test_entero = clf.predict(conjuntos_test_X_Entero)

print(confusion_matrix(conjuntos_test_Y_Entero, pred_conjunto_test_entero, labels=[0,1,2]))
print(f"Accuracy: {100*accuracy_score(conjuntos_test_Y_Entero, pred_conjunto_test_entero)}%")
print(f"Precision: {100*precision_score(conjuntos_test_Y_Entero, pred_conjunto_test_entero, average='micro')}%")
print("-------------------")

img = cv2.imread("imagenes/cow.jpg", cv2.IMREAD_COLOR)
colores = {0: [255, 0, 0], 1: [0, 255, 0], 2: [0, 0, 255]}

ancho_foto = len(img[0])
altura_foto = len(img)
nueva_vaca = np.zeros((altura_foto, ancho_foto, 3),np.uint8)

for i in range(altura_foto):
    for j in range(ancho_foto):
        prediccion = clf.predict([img[i][j]])
        nueva_vaca[i,j] = colores[prediccion[0]]

cv2.imwrite("nueva_vaca.png",nueva_vaca)

"""