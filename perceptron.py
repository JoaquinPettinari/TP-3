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
	cota = 10000
	while(error > 0 & i < cota):
		if(i > cota): print(i); break
		indice_random = random.randrange(20)
		x_al_azar=entradas[indice_random]
		y_al_azar=salidas[indice_random]
		
		exitacion = multiplicar_listas(x_al_azar, w) #Multiplicaciones de w0+w1*x1+w2*x2
		#print(exitacion)
		tita = signo(exitacion)
		#Ajustar el error en base a la tasa de aprendizaje 
		n = eta*(y_al_azar - tita)			
		#Multiplica el x con las cuentas anteriores
		w_delta = [(n*valor)[0] for valor in x_al_azar]
		w = np.add(w, w_delta)
		#print(w)
		error = calcular_error(entradas, y ,w)
		i+=1
	print(w)
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
		#print(y[i])
		error = abs(y[i] - o)
	return error
