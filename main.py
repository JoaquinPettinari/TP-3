import cv2
from imagenes import obtener_conjuntos
from perceptron import perceptron
from r2 import mostrar_puntos_en_plano, obtener_puntos, trazar_linea
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
import numpy as np


puntos = obtener_puntos()


pesos = perceptron(puntos)
#pesos = [0.5, 1.70470572, -0.64141028]
print(pesos)
trazar_linea(pesos, puntos)
mostrar_puntos_en_plano(puntos)

"""
#trazar_linea(pesos[0], pesos[1] )
print(pesos)
print("-------------------")
print("Punto 2")
conjuntos_training_X_Random, conjuntos_training_Y_Random, conjuntos_test_X_Random, conjuntos_test_Y_Random = obtener_conjuntos()
print("Con valores Random")
clf = svm.SVC(kernel='linear', C=1000)
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