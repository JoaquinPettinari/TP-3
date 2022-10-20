import numpy as np
import random

def perceptron(puntos):
	w = np.array([0,0,0])
	entradas = [[1,x1,x2] for x1,x2 in puntos[:,[0,1]]]

	#Columna de Y
	y = puntos[:,-1]
	salidas = np.array([[int(punto)] for punto in y])
	eta = 0.1
	error = 1
	i = 0
	cota = 20000
	while(error > 0 & i < cota):
		if(i > cota): break
		indice_random = random.randrange(20)
		x_al_azar=entradas[indice_random]
		y_al_azar=salidas[indice_random][0]
		exitacion = multiplicar_listas(x_al_azar, w) #Multiplicaciones de w0+w1*x1+w2*x2
		tita = signo(exitacion)
		#Ajustar el error en base a la tasa de aprendizaje 
		#print(y_al_azar - tita)
		n = eta*(y_al_azar - tita)			
		#Multiplica el x con las cuentas anteriores
		w_delta = [(n*valor) for valor in x_al_azar]
		w = np.add(w, w_delta)
		error = calcular_error(entradas, y ,w)
		i+=1
	print("Error: ", error)
	return w

def signo(exitacion):
    return 1 if (exitacion >= 0) else 0	

def multiplicar_listas(lista1, lista2):
	resultado = 0
	for i in range(len(lista1)):
		resultado += lista1[i] * lista2[i]
	return resultado

def calcular_error(X,y, w):
	error = 0
	for i in range(len(X)):
		h = multiplicar_listas(X[i], w)
		o = signo(h)
		error = abs(y[i] - o)
	return error
