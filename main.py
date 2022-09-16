import pandas as pd
from set import Set
from id3 import ID_3

df = pd.read_csv("example.csv")
#Nombre clase primaria
clasePrimaria = df.keys()[-1]

# Lista de otros atributos
nombres_atributos = list(df.keys())
nombres_atributos.remove(clasePrimaria) 

arbol = ID_3(df, clasePrimaria, nombres_atributos)
arbol.imprimirArbol()

#Entropia de los valores 0,1 Creditability
entropia_total = arbol.entropia_de_un_conjunto(df[clasePrimaria])
#print(entropia_total)
