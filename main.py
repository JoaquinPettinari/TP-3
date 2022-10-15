from r2 import mostrar_puntos_en_plano, obtener_puntos, obtener_valores_de_columna, perceptron, plot_decision_boundary, trazar_linea
import cv2
import random

from utils import obtener_conjunto_entrenamiento, obtener_conjunto_test

cielo = cv2.imread("imagenes/cielo.jpg", cv2.IMREAD_COLOR)
cielo = random.choices(cielo, k=100)
training_cielo = obtener_conjunto_entrenamiento(cielo[0])
test_cielo = obtener_conjunto_test(cielo[0])
print(len(training_cielo))
print(len(test_cielo))
"""
trazar_linea()
puntos = obtener_puntos()
mostrar_puntos_en_plano(puntos)
theta = perceptron(TP3_1_X, TP3_1_Y, 0.5)
plot_decision_boundary(TP3_1_X, theta)
"""
