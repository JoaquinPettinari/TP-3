# File name: RandomForest.py

import numpy as np
from arbol_decision import ID3, predecir
from collections import Counter

#clase_primaria = 'Juega'
clase_primaria = 'Creditability'

def bootstraping(conjunto):
    # Función que crea valores aleatorios para todas las columnas menos la Creditability
    for i in range(len(conjunto)):        
        for k in range(len(conjunto.keys()) - 1):
            atributo = conjunto.keys()[k]
            valores_unicos = np.unique(conjunto[atributo])
            conjunto[atributo][i] = np.random.choice(valores_unicos)
    return conjunto

def RandomForest_Train(dataset,numbero_de_arboles):
    # Lista para ir guardando cada árbol
    arboles_random_forest = []
    
    # For para la cantidad de arboles a crear
    for i in range(numbero_de_arboles):
        # Crea los datos aleatorios del bootstraping
        # De esta forma usa los mismos datos pero mezclados
        #conjunto_bootstrap = dataset.sample(frac=1,replace=True)
        # Con esta función crea random 100% dentro de los valores de la columna
        conjunto_bootstrap = bootstraping(dataset)
        # Se agrega a la lista el arbol que se crea con el conjunto de datos
        arboles_random_forest.append(ID3(conjunto_bootstrap,conjunto_bootstrap,dataset.drop(labels=[clase_primaria],axis=1).columns))
    
    return arboles_random_forest

def RandomForest_predecir(fila_dict,random_forest,default='p'):    
    predictions = []
    #Por cada arbol que hay 
    for tree in random_forest:
        # Se obtiene la predicción de la fila en cada árbol
        predictions.append(predecir(fila_dict,tree,default))
    #Devuelve la predicción mas común [1,0,1,1,0] -> [(1,3)] -> [(1,3)][0][0] -> 1
    return Counter(predictions).most_common(1)[0][0]

def RF_obtener_clase_predecida(conjunto,random_forest, clase_primaria):
    # Devuelve una lista con las clases predecidas
    predicciones = []
    for i in range(len(conjunto)):
        #Se queda con los valores de la fila sin la clase primaria y lo transforma a un dict
        fila_dict = conjunto.iloc[i,:].drop(clase_primaria).to_dict()
        predicciones.append(RandomForest_predecir(fila_dict,random_forest,default='p'))
    return predicciones
        