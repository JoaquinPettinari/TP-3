class Nodo():
    def __init__(self, indice_atributo=None, threshold=None, izquierdo=None, derecho=None, info_ganancia=None, valor=None):
        ''' constructor ''' 
        
        # for decision node
        self.indice_atributo = indice_atributo
        self.threshold = threshold
        self.izquierdo = izquierdo
        self.derecho = derecho
        self.info_ganancia = info_ganancia
        
        # for leaf node
        self.valor = valor