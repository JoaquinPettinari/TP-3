import numpy as np
import matplotlib.pyplot as plt
import random

def trazar_linea(w, puntos):
    #plt.plot([0,5], [2,7])
    x=[]
    y=[]
    for punto in puntos:
        x.append(punto[0])
        y.append(-w[1]/w[2]*punto[0]-w[0]/w[2])
    #print(x,y)
    plt.plot(x,y)


def formula_y(x):
    return x + 2

def obtener_puntos():
    puntos = np.empty((20,3))
    for i in range(20):
        for j in range(3):
            if(j == 2):
                x1 = puntos[i,0]
                x2 = puntos[i,1]
                puntos[i,j] = 1 if x2 <= formula_y(x1) else -1
            else:
                puntos[i,j] = random.uniform(0, 5)
    return puntos

def mostrar_puntos_en_plano(puntos):
    #Conjuntos de x
    X = puntos[:,[0,1]]
    #Columna de Y
    y = puntos[:,-1]
    # Plot de todos los Xi con Y = -1
    plt.plot(X[:, 0][y == -1], X[:, 1][y == -1], 'r^')
    # Plot de todos los Xi con Y = 1
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], 'bs')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
