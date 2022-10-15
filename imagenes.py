from sklearn import svm
import numpy as np
import cv2
import random

from utils import flattenList, obtener_conjuntos_de_datos, obtener_y_de_imagen

def obtener_conjuntos():

    conjuntos_training_X = []
    conjuntos_training_Y = []
    conjuntos_test_X = []
    conjuntos_test_Y = []


    for i, image in enumerate(["cielo.jpg", "pasto.jpg", "vaca.jpg"]):
        img = cv2.imread("imagenes/" + image, cv2.IMREAD_COLOR)
        img = random.choices(img, k=100)
        # Divide el conjunto en entrenamiento y test
        # Se le pone img[0] para que agarre el dato de la matrix, porque random devuelve una tupla
        training_cielo, test_cielo = obtener_conjuntos_de_datos(img[0])
        # Punto 2. a
        # Valores Y de las imagenes
        training_Y = obtener_y_de_imagen(len(training_cielo), i)
        test_Y = obtener_y_de_imagen(len(test_cielo), i)

        conjuntos_training_X.append(training_cielo)
        conjuntos_test_X.append(test_cielo)
        conjuntos_training_Y.append(training_Y)
        conjuntos_test_Y.append(test_Y)
        #Flatteo las listas así tengo todos los valores en una lista sola
        if(i == 2):
            conjuntos_training_X = flattenList(conjuntos_training_X)
            conjuntos_training_Y = flattenList(conjuntos_training_Y)
            conjuntos_test_X = flattenList(conjuntos_test_X)
            conjuntos_test_Y = flattenList(conjuntos_test_Y)
    
    return conjuntos_training_X, conjuntos_training_Y, conjuntos_test_X, conjuntos_test_Y