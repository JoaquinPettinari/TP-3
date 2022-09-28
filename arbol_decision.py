import pandas as pd
import numpy as np
import math

#clase_primaria = "Juega"
clase_primaria = "Creditability"


# Calcula la entropía de una columna
def entropia(columna_objetivo):  
    elementos,cantidad = np.unique(columna_objetivo,return_counts = True)
    return np.sum([(-cantidad[i]/np.sum(cantidad))*np.log2(cantidad[i]/np.sum(cantidad)) for i in range(len(elementos))])

def info_ganancia(conjunto,nombre_atributo_divisor,clase_objetivo= clase_primaria):
    
    #Entropía de la clase primaria
    entropia_total = entropia(conjunto[clase_objetivo])
    
    #Valores únicos del atributo
    valores,cantidad= np.unique(conjunto[nombre_atributo_divisor],return_counts=True)

    # Fórmula de la ganancia h(s) - E de valores(A)
    peso_entropia = np.sum([(cantidad[i]/np.sum(cantidad))*entropia(conjunto.where(conjunto[nombre_atributo_divisor]==valores[i]).dropna()[clase_objetivo]) for i in range(len(valores))])
    
    # Función ganancia
    return entropia_total - peso_entropia

def ID3(conjunto,conjunto_original,atributos,nombre_atributo_objetivo=clase_primaria ,clase_nodo_padre = None):
    # (En el conjunto) Si en la columna del atributo objetivo hay un solo valor corta la recursión y devuelve el valor
    if len(np.unique(conjunto[nombre_atributo_objetivo])) <= 1:        
        return np.unique(conjunto[nombre_atributo_objetivo])[0]
    #Si los atributos vienen vacíos, Devuelve el valor del padre    
    elif len(atributos) ==0:        
        return clase_nodo_padre
    # Si no cumple, se construye el árbol    
    else:
        # Guarda el valor del nodo 
        # return_counts = Devuelve la cantidad de veces que aparece un valor único en el array
        # En python si haces [0,1,2][1] -> 1
        clase_nodo_padre = np.unique(conjunto[nombre_atributo_objetivo])[np.argmax(np.unique(conjunto[nombre_atributo_objetivo],return_counts=True)[1])]
        # Selecciona cual es la mejor división para el conjunto
        # Devuelve los valores de la información de la ganancia para los atributos del conjunto 
        valores_atributo = [info_ganancia(conjunto,feature,nombre_atributo_objetivo) for feature in atributos] 
        indice_mejor_atributo = np.argmax(valores_atributo)
        mejor_atributo = atributos[indice_mejor_atributo]
        #Crea la estructura del arbol. 
        tree = {mejor_atributo:{}}
        #Elimina el atributo con mejor información de ganancia en el         
        atributos = [i for i in atributos if i != mejor_atributo]
        # Iteración por cada hijo del nodo padre.
        for hijo in np.unique(conjunto[mejor_atributo]):
            # Divide el conjunto a través del valor del atributo con mas información de ganancia y con eso crea el subconjunto
            # El where es como un filter
            sub_conjunto = conjunto.where(conjunto[mejor_atributo] == hijo).dropna()
            # Llama a la recursión con los nuevos valores
            subtree = ID3(sub_conjunto,conjunto_original,atributos,nombre_atributo_objetivo,clase_nodo_padre)
            #Agrega el subarbol al dict
            tree[mejor_atributo][hijo] = subtree
        return(tree)            
    
def predecir(fila,arbol,default = 1): 
    #Por cada atributo de la fila
    for atributo in list(fila.keys()):
        # SI existe en el arbol
        if atributo in list(arbol.keys()):
            try:
                #Devuelve el valor que encuentra en el arbol
                valor = arbol[atributo][fila[atributo]] 
            except:
                return default
            # Si es un dict (no valor exacto) hace la recursión con eso
            if isinstance(valor,dict):
                return predecir(fila,valor)
            # Si no, devuelve el valor
            else:
                return math.trunc(valor)


def obtener_clase_predecida(data,arbol):
    conjunto_sin_clase_primaria = data.iloc[:,:-1].to_dict(orient = "records")
    #Lista de precciones 
    predicted = []    
    # Itera cada columna y la predice
    for i in range(len(data)):
        # Agrega la predicción al árbol
        predicted.append(predecir(conjunto_sin_clase_primaria[i],arbol,1))
    return predicted
    