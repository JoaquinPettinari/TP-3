# File name: RandomForest.py

import numpy as np
from arbol_decision import ID3, predecir
import scipy.stats as sps

def bootstraping(conjunto):
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
        arboles_random_forest.append(ID3(conjunto_bootstrap,conjunto_bootstrap,dataset.drop(labels=['Creditability'],axis=1).columns))
    
    return arboles_random_forest

def RandomForest_predecir(query,random_forest,default='p'):
    predictions = []
    for tree in random_forest:
        predictions.append(predecir(query,tree,default))
    print(predictions)
    print(sps.mode(predictions, keepdims=True))
    return sps.mode(predictions, keepdims=True)[0][0]

def RF_obtener_clase_predecida(conjunto,random_forest, clase_primaria):
    predicciones = []
    for i in range(len(conjunto)):
        query = conjunto.iloc[i,:].drop(clase_primaria).to_dict()
        predicciones.append(RandomForest_predecir(query,random_forest,default='p'))
    #accuracy = sum(predictions == data['Creditability'])/len(data)*100
    #print('The prediction accuracy is: ',sum(data['predictions'] == data['target'])/len(data)*100,'%')
    return predicciones
        