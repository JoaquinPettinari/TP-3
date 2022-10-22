import numpy as np
import random

from utils import separar_conjunto

def perceptron(puntos, cantidad_puntos):
	w = np.array([0,0,0])
	X,y =separar_conjunto(puntos)	
	entradas = [[1,x1,x2] for x1,x2 in X]

	#Columna de Y
	salidas = np.array([[int(punto)] for punto in y])
	eta = 0.1
	error = 1
	i = 0
	cota = 20000
	while error > 0 and i < cota:
		# Indice para valor al azar
		indice_random = random.randrange(cantidad_puntos)
		x_al_azar=entradas[indice_random]
		y_al_azar=salidas[indice_random][0]
		#Multiplicaciones de w0+w1*x1+w2*x2
		exitacion = multiplicar_listas(x_al_azar, w) 
		tita = signo(exitacion)
		#Fórmula
		n = eta*(y_al_azar - tita)			
		#Multiplica el x con las cuentas anteriores. 2 * (4,5,6) -> 8,10,12
		w_delta = [(n*valor) for valor in x_al_azar]
		#(1,2,3) + (4,5,6) -> (5,7,9)
		w = np.add(w, w_delta)
		error = calcular_error(entradas, y ,w)
		i+=1
	print("Error: ", error)
	return w

def signo(exitacion):
    return 1 if (exitacion >= 0) else -1	

def multiplicar_listas(lista1, lista2):
	resultado = 0
	for i in range(len(lista1)):
		#w0*x0+w1*x1...+wn*xn
		resultado += lista1[i] * lista2[i]
	return resultado

def calcular_error(X,y, w):
	error = 0
	for i in range(len(X)):
		h = multiplicar_listas(X[i], w)
		o = signo(h)
		error += abs(y[i] - o)
	return error
