import numpy as np
import matplotlib.pyplot as plt
import random

def trazar_linea():
    plt.plot([0,8], [2,10])

def formula_y(x):
    print(x)
    return x + 2

def obtener_puntos():
    puntos = np.empty((20,3))
    for i in range(20):
        for j in range(3):
            if(j == 2):
                x1 = puntos[i,0]
                x2 = puntos[i,1]
                puntos[i,j] = 1 if x2 <= formula_y(x1)  else -1
            else:
                puntos[i,j] = random.uniform(0, 10)
    return puntos












def step_func(z):
    return 1.0 if (z > 0) else -1.0

def obtener_valores_de_columna(columna, matriz):
    return [i[columna] for i in matriz]

def perceptron(TP_X, y, lr):        
    X = np.matrix(TP_X)
    # n-> número de atributos
    _, n = X.shape
    
    # Initializing parapeters(theta) to zeros.
    # +1 in n+1 for the bias term.
    theta = np.zeros((n+1,1))
    print(X)
    
    # Empty list to store how many examples were 
    # misclassified at every iteration.
    
    # looping for every example.
    for idx, x_i in enumerate(X):
        
        # Insering 1 for bias, X0 = 1.
        print(x_i)
        print(idx)
        x_i = np.insert(x_i, 0, 1).reshape(-1,1)
        print(x_i.T)
        print(theta)
        """
        a = np.array([[1,2],[3,4]]) 
        b = np.array([[11,12],[13,14]]) 
        np.dot(a,b) -> [[1*11+2*13, 1*12+2*14],[3*11+4*13, 3*12+4*14]]

        .T = [[1,2,3]] -> [[1][2][3]]
        """
        # Calculating prediction/hypothesis.
        y_hat = step_func(np.dot(x_i.T, theta))
        print(np.dot(x_i.T, theta))
        # Updating if the example is misclassified.
        if (np.squeeze(y_hat) - y[idx]) != 0:
            theta += lr*((y[idx] - y_hat)*x_i)
            
            # Incrementing by 1.
    
        # Appending number of misclassified examples
        # at every iteration.
    print("-----------------")
    return theta

def plot_decision_boundary(TP_X, theta):
    X_0 = obtener_valores_de_columna(0, TP_X)

    x1 = [min(X_0), max(X_0)]
    m = -theta[1]/theta[2]
    print(m)    
    c = -theta[0]/theta[2]
    print(c)
    x2 = m*x1 + c
    
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.xlim([0,10])
    plt.ylim([0,10])
    plt.plot(x1, x2)
    plt.show()
