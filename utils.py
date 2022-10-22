import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

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

def print_metricas(test_Y, conjunto_prediccion, labels=[0,1,2]):
    print(confusion_matrix(test_Y, conjunto_prediccion, labels=labels))
    print(f"Accuracy: {100*accuracy_score(test_Y, conjunto_prediccion)}%")
    print(f"Precision: {100*precision_score(test_Y, conjunto_prediccion, average='micro')}%")