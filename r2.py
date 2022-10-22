import numpy as np
import matplotlib.pyplot as plt
import random
from random import sample

from utils import separar_conjunto

def trazar_linea_con_pesos(w):
    m = -w[1]/w[2]
    b = -w[0]/w[2]
    plt.plot([0, 5], [b, m*5+b], color ='red')

def trazar_linea_inicial():    
    plt.plot([0,5], [1,6])

def formula_y(x):
    return x + 1

def obtener_puntos(cantidad_puntos):
    puntos = np.empty((cantidad_puntos,3))
    for i in range(cantidad_puntos):
        for j in range(3):
            if(j == 2):
                x1 = puntos[i,0]
                x2 = puntos[i,1]
                puntos[i,j] = 1 if x2 <= formula_y(x1) else -1
            else:
                puntos[i,j] = random.uniform(0, 5)
    return puntos

def obtener_puntos_mal_clasificados(puntos):
    X, y = separar_conjunto(puntos)
    # Deuvelve y al azar. Algunos se van a quedar bien calificados y otros no
    np.random.shuffle(y)
    # Vuelve a unir los conjuntos pero con los y mezclados
    return np.column_stack((X, y))


def mostrar_puntos_en_plano(puntos):
    # Conjuntos de x
    # Columna de Y
    X, y = separar_conjunto(puntos)    
    # Plot de todos los Xi con Y = -1
    plt.plot(X[:, 0][y == -1], X[:, 1][y == -1], 'r^')
    # Plot de todos los Xi con Y = 1
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], 'bs')
    plt.xlabel("x")
    plt.ylabel("y")
    # Limita el rango del plano
    plt.xlim([0,5])
    plt.ylim([0,5])
    plt.show()