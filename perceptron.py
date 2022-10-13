import numpy as np

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
