# File name: RandomForest.py

import numpy as np
from arbol_decision import arbol_decision, predecir
from collections import Counter

#clase_primaria = 'Juega'
clase_primaria = 'Creditability'

def RandomForest(dataset,numero_de_arboles):
    # Lista para ir guardando cada árbol
    arboles_random_forest = []
    
    # For para la cantidad de arboles a crear
    for i in range(numero_de_arboles):
        # Crea los datos aleatorios del bootstraping
        # De esta forma usa datos random del conjunto
        conjunto_bootstrap = dataset.sample(frac=1,replace=True)
        # Se agrega a la lista el arbol que se crea con el conjunto de datos          #Lista de atributos sin la clase primaria
        arboles_random_forest.append(arbol_decision(conjunto_bootstrap,conjunto_bootstrap,dataset.drop(labels=[clase_primaria],axis=1).columns))
    
    return arboles_random_forest

def RandomForest_predecir(fila_dict,random_forest,default='p'):    
    predicciones = []
    #Por cada arbol que hay 
    for arbol in random_forest:
        # Se obtiene la predicción de la fila en cada árbol
        predicciones.append(predecir(fila_dict,arbol,default))
    #Devuelve la predicción mas común [1,0,1,1,0] -> [(1,3)] -> [(1,3)][0][0] -> 1
    return Counter(predicciones).most_common(1)[0][0]

def RF_obtener_clase_predecida(conjunto,random_forest, clase_primaria):
    # Devuelve una lista con las clases predecidas
    predicciones = []
    for i in range(len(conjunto)):
        #Se queda con los valores de la fila sin la clase primaria y lo transforma a un dict
        fila_dict = conjunto.iloc[i,:].drop(clase_primaria).to_dict()
        predicciones.append(RandomForest_predecir(fila_dict,random_forest,default='p'))
    return predicciones
        