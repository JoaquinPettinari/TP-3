import numpy as np

def obtener_conjuntos_training_test(conjunto):
    conjunto_X, conjunto_Y = separar_conjunto(conjunto)
    X_Training, X_Test = obtener_conjuntos_de_datos(conjunto_X)
    Y_Training, Y_Test = obtener_conjuntos_de_datos(conjunto_Y)
    return X_Training, Y_Training, X_Test, Y_Test

def obtener_conjuntos_de_datos(conjunto):
    p80 = len(conjunto) * 0.80
    p20 = len(conjunto) * 0.20
    return conjunto[:int(p80)], conjunto[-int(p20):]

def flattenList(list):
    return [item for sublist in list for item in sublist]

def separar_conjunto(conjunto):
    X = np.array(conjunto)[:,[0,1]]
    y = np.array(conjunto)[:,-1]
    return X, y

def clasificar_imagen(conjunto, clase):
    nuevo_conjunto = np.empty((len(conjunto), 4))
    for i in range(len(conjunto)):
        for j in range(4):
            if(j == 3):
                nuevo_conjunto[i,j] = clase
            else:
                nuevo_conjunto[i,j] = conjunto[i,j]
    return nuevo_conjunto

def poner_clase_de_imagen(conjunto, clase):
    return [clase for i in range(len(conjunto))]