import pandas as pd
import math
from collections import Counter
import matplotlib.pyplot as plt
from pprint import pprint

class ID_3:
    def __init__(self, df, clasePrimaria, atributos):
        # print("\n Given Play Golf Dataset:\n\n", df)
        # print('Target Attribute is   ➡ ', clasePrimaria)
        self.df = df
        self.clasePrimaria = clasePrimaria
        self.atributos = atributos
    
    def entropia_de_un_conjunto(self, conjunto):
        # Contador que calcula la frecuencia relativa de el conjunto Counter({'1': 9, '0': 5 })
        contador = Counter(x for x in conjunto)   
        # Numero total de datos (Ejemplo arriba es 14)
        num_instancias = len(conjunto)*1.0          
        
        # Se mapea el contador / el total para obtener la probabilidad. 9/14 y 5/14 -> [0.35714, 0.64285]
        probs = [x / num_instancias for x in contador.values()]        
        # Call Entropy
        return self.entropy(probs) 
    
    def entropy(self, probs):
        # Calculo de la entropia recibe lista con 2 valores y aplica fórmula de -sum prob*log(prob)
        return sum( [-prob*math.log(prob, 2) for prob in probs])