from set import Set
from id3 import ID3_C
import numpy as np

datos = np.genfromtxt('Training.csv', delimiter=",", dtype="str")

X     = datos[:, :-1]
Y     = datos[:, -1]

arbol = ID3_C()
arbol.entrenar(X, Y)
print(len(X))
print(Y)

salida = arbol.predecir(X)
print(salida)
print('Porcentaje de aciertos: ', 100 * sum(Y == salida)/X.shape[0])