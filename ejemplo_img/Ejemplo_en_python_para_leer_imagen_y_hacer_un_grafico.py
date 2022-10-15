# -*- coding: utf-8 -*-
"""
@author: Juan
"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import svm

# Crear una ventana para mostrar la imagen
cv.namedWindow("Imagen", 0)  
  
# Leer la imagen
imagen = cv.imread("imagen.png", 0)
print(imagen)
print("Dimensiones ", imagen.shape)

# Mostrar la imagen
cv.imshow("Imagen", imagen)
# Necesario para que opencv la muestre. 0 - espera hasta que se 
# presione una tecla, > 0 - la cant de miliseg que se muestra la imagen
cv.waitKey(0)  
cv.destroyAllWindows()

# Para graficar datos
# Une los puntos con una línea de color naranja
# plt.plot([2, 3, 5], [5, 3, 2], color ='tab:orange')
# Dibuja pelotitas en las coordenadas 
# plt.plot([2, 3, 5], [5, 3, 2], "o", color ='tab:orange')
# Dibuja pelotitas unidas con una línea en las coordenadas
plt.plot([2, 3, 5], [5, 3, 2], 'o-', linewidth=2, markersize=12)

plt.draw()  
# Establecer limites del gráfico
plt.xlim([0, 10])  
plt.ylim([0, 10])  
# Poner un título en el gráfico
plt.title('Ejemplo') 
# Mostrar el gráfico
plt.show() 

# SVM
X = [[0, 0], [1, 1]]
y = [0, 1]
clf = svm.SVC()
clf.fit(X, y)
print("Predicción para el ejemplo de prueba (2, 2) es ", clf.predict([[2., 2.]]))
print("Los vectores soportes son ", clf.support_vectors_)
