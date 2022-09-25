"""
Make the imports of python packages needed
"""
import pandas as pd
import numpy as np
import math


#Import the dataset and define the feature as well as the target datasets / columns#
#Import all columns omitting the fist which consists the names of the animals

#clase_primaria = "PlayGolf"
clase_primaria = "Creditability"


# Calcula la entropía de una columna
def entropia(columna_objetivo):  
    elementos,cantidad = np.unique(columna_objetivo,return_counts = True)
    return np.sum([(-cantidad[i]/np.sum(cantidad))*np.log2(cantidad[i]/np.sum(cantidad)) for i in range(len(elementos))])

def info_ganancia(conjunto,nombre_atributo_divisor,clase_objetivo= clase_primaria):
    
    #Entropía de la clase primaria
    entropia_total = entropia(conjunto[clase_objetivo])
    
    #Calculate the values and the corresponding counts for the split attribute 
    valores,cantidad= np.unique(conjunto[nombre_atributo_divisor],return_counts=True)

    # Fórmula de la ganancia h(s) - E de valores(A)
    peso_entropia = np.sum([(cantidad[i]/np.sum(cantidad))*entropia(conjunto.where(conjunto[nombre_atributo_divisor]==valores[i]).dropna()[clase_objetivo]) for i in range(len(valores))])
    
    # Función ganancia
    return entropia_total - peso_entropia
       
###################

###################


def ID3(conjunto,conjunto_original,atributos,nombre_atributo_objetivo=clase_primaria ,clase_nodo_padre = None):
    # (En el conjunto) Si en la columna del atributo objetivo hay un solo valor corta la recursión y devuelve el valor
    if len(np.unique(conjunto[nombre_atributo_objetivo])) <= 1:        
        return np.unique(conjunto[nombre_atributo_objetivo])[0]
    #Si los atributos vienen vacíos, Devuelve el valor del padre    
    elif len(atributos) ==0:        
        return clase_nodo_padre
    # Si no cumple, se construye el árbol    
    else:
        #Guarda el valor del nodo - return_counts = Devuelve la cantidad de veces que aparece un valor único en el array
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
    
def predecir(query,tree,default = 1):
    """
    Prediction of a new/unseen query instance. This takes two parameters:
    1. The query instance as a dictionary of the shape {"feature_name":feature_value,...}

    2. The tree 


    We do this also in a recursive manner. That is, we wander down the tree and check if we have reached a leaf or if we are still in a sub tree. 
    Since this is a important step to understand, the single steps are extensively commented below.

    1.Check for every feature in the query instance if this feature is existing in the tree.keys() for the first call, 
    tree.keys() only contains the value for the root node 
    --> if this value is not existing, we can not make a prediction and have to 
    return the default value which is the majority value of the target feature

    2. First of all we have to take care of a important fact: Since we train our model with a database A and then show our model
    a unseen query it may happen that the feature values of these query are not existing in our tree model because non of the
    training instances has had such a value for this specific feature. 
    For instance imagine the situation where your model has only seen animals with one to four
    legs - The "legs" node in your model will only have four outgoing branches (from one to four). If you now show your model
    a new instance (animal) which has for the legs feature the vale 5, you have to tell your model what to do in such a 
    situation because otherwise there is no classification possible because in the classification step you try to 
    run down the outgoing branch with the value 5 but there is no such a branch. Hence: Error and no Classification!
    We can address this issue with a classification value of for instance (999) which tells us that there is no classification
    possible or we assign the most frequent target feature value of our dataset used to train the model. Or, in for instance 
    medical application we can return the most worse case - just to make sure... 
    We can also return the most frequent value of the direct parent node. To make a long story short, we have to tell the model 
    what to do in this situation.
    In our example, since we are dealing with animal species where a false classification is not that critical, we will assign
    the value 1 which is the value for the mammal species (for convenience).

    3. Address the key in the tree which fits the value for key --> Note that key == the features in the query. 
    Because we want the tree to predict the value which is hidden under the key value (imagine you have a drawn tree model on 
    the table in front of you and you have a query instance for which you want to predict the target feature 
    - What are you doing? - Correct:
    You start at the root node and wander down the tree comparing your query to the node values. Hence you want to have the
    value which is hidden under the current node. If this is a leaf, perfect, otherwise you wander the tree deeper until you
    get to a leaf node. 
    Though, you want to have this "something" [either leaf or sub_tree] which is hidden under the current node
    and hence we must address the node in the tree which == the key value from our query instance. 
    This is done with tree[keys]. Next you want to run down the branch of this node which is equal to the value given "behind"
    the key value of your query instance e.g. if you find "legs" == to tree.keys() that is, for the first run == the root node.
    You want to run deeper and therefore you have to address the branch at your node whose value is == to the value behind key.
    This is done with query[key] e.g. query[key] == query['legs'] == 0 --> Therewith we run down the branch of the node with the
    value 0. Summarized, in this step we want to address the node which is hidden behind a specific branch of the root node (in the first run)
    this is done with: result = [key][query[key]]

    4. As said in the 2. step, we run down the tree along nodes and branches until we get to a leaf node.
    That is, if result = tree[key][query[key]] returns another tree object (we have represented this by a dict object --> 
    that is if result is a dict object) we know that we have not arrived at a root node and have to run deeper the tree. 
    Okay... Look at your drawn tree in front of you... what are you doing?...well, you run down the next branch... 
    exactly as we have done it above with the slight difference that we already have passed a node and therewith 
    have to run only a fraction of the tree --> You clever guy! That "fraction of the tree" is exactly what we have stored
    under 'result'.
    So we simply call our predict method using the same query instance (we do not have to drop any features from the query
    instance since for instance the feature for the root node will not be available in any of the deeper sub_trees and hence 
    we will simply not find that feature) as well as the "reduced / sub_tree" stored in result.

    SUMMARIZED: If we have a query instance consisting of values for features, we take this features and check if the 
    name of the root node is equal to one of the query features.
    If this is true, we run down the root node outgoing branch whose value equals the value of query feature == the root node.
    If we find at the end of this branch a leaf node (not a dict object) we return this value (this is our prediction).
    If we instead find another node (== sub_tree == dict objct) we search in our query for the feature which equals the value 
    of that node. Next we look up the value of our query feature and run down the branch whose value is equal to the 
    query[key] == query feature value. And as you can see this is exactly the recursion we talked about
    with the important fact that for each node we run down the tree, we check only the nodes and branches which are 
    below this node and do not run the whole tree beginning at the root node 
    --> This is why we re-call the classification function with 'result'
    """
    
    
    #1.
    for key in list(query.keys()):
        if key in list(tree.keys()):
            #2.
            try:
                result = tree[key][query[key]] 
            except:
                return default
  
            #3.
            result = tree[key][query[key]]
            #4.
            if isinstance(result,dict):
                return predecir(query,result)

            else:
                return math.trunc(result)

        
        
"""
Check the accuracy of our prediction.
The train_test_split function takes the dataset as parameter which should be divided into
a training and a testing set. The test function takes two parameters, which are the testing data as well as the tree model.
"""
###################

###################

def obtener_clase_predecida(data,tree):
    #Create new query instances by simply removing the target feature column from the original dataset and 
    #convert it to a dictionary
    queries = data.iloc[:,:-1].to_dict(orient = "records")
    #Create a empty DataFrame in whose columns the prediction of the tree are stored
    predicted = []
    
    #Calculate the prediction accuracy
    for i in range(len(data)):
        predicted.append(predecir(queries[i],tree,1.0))
    
    return predicted
    