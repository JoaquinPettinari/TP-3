import numpy as np
from nodo import Nodo
from collections import Counter

class DecisionTree():
    def __init__(self, nombre_columnas, min_de_observaciones=2, max_profundidad_del_arbol=2):
        ''' constructor '''
        
        # initialize the root of the tree 
        self.root = None
        
        # stopping conditions
        self.min_samples_split = min_de_observaciones
        self.max_profundidad_del_arbol = max_profundidad_del_arbol
        self.nombre_columnas = nombre_columnas
        
    # Función recursiva para construir el árbol
    def contruir_arbol(self, conjunto, profundidad=0):
        #Valores de X = valores de los atributos. Y = Clase primaria
        X, Y = conjunto[:,:-1], conjunto[:,-1]   
        # El shape devuelve una tupla de los valores declarados abajo (x,y)    
        cant_filas_en_un_conjunto, cant_de_atributos = np.shape(X)
        # Validación para del árbol, que no se pase de la profundidad y tenga un mínimo de datos
        if cant_filas_en_un_conjunto>=self.min_samples_split and profundidad<=self.max_profundidad_del_arbol:
            # Busca el mejor valor para hacer la división del árbol. Devuelve un objeto con los valores y la división del árbol
            mejor_particion = self.obtener_mejor_particion(conjunto, cant_de_atributos)            
            # Validación de error
            if(mejor_particion == {}):
                pass
            # Mira si el peso es positivo
            elif mejor_particion["info_ganancia"]>0:                
                # Recursividad del árbol izquierdo 
                subarbol_izquierdo = self.contruir_arbol(mejor_particion["conjunto_izquierdo"], profundidad+1)
                # Recursividad del árbol derecho
                subarbol_derecho = self.contruir_arbol(mejor_particion["conjunto_derecho"], profundidad+1)
                # Devuelve el nodo decisión.
                return Nodo(mejor_particion["indice_atributo"], mejor_particion["threshold"], 
                            subarbol_izquierdo, subarbol_derecho, mejor_particion["info_ganancia"])
                
        # Calcula valor de la hoja
        valor_de_la_hoja = self.calculate_leaf_value(Y)
        # return leaf node
        return Nodo(valor=valor_de_la_hoja)
    
    # Función para obtener la mejor partición. Devuelve un objeto
    def obtener_mejor_particion(self, conjunto, cant_atributos):
        mejor_particion = {}
        max_info_ganancia = -float("inf")
        # Iteración para la cantidad de atributos(iteración por columna)
        for indice_atributo in range(cant_atributos):
            # (Por iteración) Obtiene los valores de la columna
            valores_del_atributo = conjunto[:, indice_atributo]
            #Valores únicos en la columna
            posibles_threshold = np.unique(valores_del_atributo)
            
            # loop over all the feature values present in the data
            for threshold in posibles_threshold:
                # Obtiene la partición
                conjunto_izquierdo, conjunto_derecho = self.dividir(conjunto, indice_atributo, threshold)
                # Verificación de que los hijos no sean nulos
                if len(conjunto_izquierdo)>0 and len(conjunto_derecho)>0:
                    y, left_y, right_y = conjunto[:, -1], conjunto_izquierdo[:, -1], conjunto_derecho[:, -1]
                    # Obtener ganancia
                    info_ganancia_actual = self.obtener_ganancia(y, left_y, right_y)
                    # Actualiza la mejor partición si hace falta
                    if info_ganancia_actual>max_info_ganancia:
                        mejor_particion["indice_atributo"] = indice_atributo
                        mejor_particion["threshold"] = threshold
                        mejor_particion["conjunto_izquierdo"] = conjunto_izquierdo
                        mejor_particion["conjunto_derecho"] = conjunto_derecho
                        mejor_particion["info_ganancia"] = info_ganancia_actual
                        max_info_ganancia = info_ganancia_actual
                        
        # return best split
        return mejor_particion
    
    def dividir(self, conjunto, indice_atributo, valor):
        ''' function to split the data '''
        # Crea los conjuntos por (List Comprehension). Devuelve fila si cumple con la condición
        conjunto_izquierdo = np.array([row for row in conjunto if row[indice_atributo]<=valor])
        conjunto_derecho = np.array([row for row in conjunto if row[indice_atributo]>valor])
        return conjunto_izquierdo, conjunto_derecho
    
    def obtener_ganancia(self, parent, l_child, r_child):
        ''' function to compute information gain '''
        
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        return self.indice_gini(parent) - (weight_l*self.indice_gini(l_child) + weight_r*self.indice_gini(r_child))
        
    # Calcular indice gini
    def indice_gini(self, y):                
        valores_de_la_clase = np.unique(y)
        gini = 0
        for valor in valores_de_la_clase:            
            # Frecuencia relativa
            p_clase = len(y[y == valor]) / len(y)
            gini += p_clase**2
        return 1 - gini
        
    def calculate_leaf_value(self, Y):
        ''' function to compute leaf node '''
        
        Y = list(Y)
        return max(Y, key=Y.count)
    
    def print_tree(self, tree=None, indent=" "):
        ''' function to print the tree '''
        
        if not tree:
            tree = self.root

        if tree.valor is not None:
            print(tree.valor)

        else:
            print("X_"+self.nombre_columnas[tree.indice_atributo])
            print("%sIzq:" % (indent), end="")
            self.print_tree(tree.izquierdo, indent + indent)
            print("%sDer:" % (indent), end="")
            self.print_tree(tree.derecho, indent + indent)
    
    def entrenar(self, X, Y):
        # Concatena la matriz de la clase primaria y los atributos
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.contruir_arbol(dataset)
    
    def predict(self, X):
        ''' function to predict new dataset '''
        
        return [self.make_prediction(x, self.root) for x in X]
        
    
    def make_prediction(self, x, tree:Nodo):
        ''' function to predict a single data point '''        
        if tree.valor!=None: return tree.valor
        feature_val = x[tree.indice_atributo]
        if feature_val<=tree.threshold:
            return self.make_prediction(x, tree.izquierdo)
        else:
            return self.make_prediction(x, tree.derecho)