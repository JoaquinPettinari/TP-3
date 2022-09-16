import pandas as pd
from set import Set
from id3 import ID_3

df = pd.read_csv("example.csv")
clasePrimaria = df.keys()[-1]

nombres_atributos = list(df.keys())
arbol = ID_3(df, clasePrimaria, nombres_atributos)
entropia_total = arbol.entropia_de_un_conjunto(df[clasePrimaria])
print(entropia_total)
