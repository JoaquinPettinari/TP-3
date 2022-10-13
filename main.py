from perceptron import obtener_valores_de_columna, perceptron
from sklearn import datasets
import matplotlib.pyplot as plt
from pandas import DataFrame
import numpy as np

TP3_1_X = [
    [10.02488611,  5.08333171],
    [ 6.86675981,  5.96880721],
    [ 2.23989814,  4.28353092],
    [ 7.46134859,  4.58877252],
    [ 1.57845107,  8.26537513],
    [ 7.90593303,  4.20733751],
    [ 0.1661053,  9.58255841],
    [ 1.95947097,  9.11096293],
    [ 9.669524,  4.51458008],
    [ 1.22871378,  9.54865641],
    [ 6.92799963,  5.40733662],
    [ 8.05409731,  5.55928774],
    [ 1.72877019,  9.23272803],
    [ 1.35649936,  9.91166608],
    [ 8.31001394,  5.75179727],
    [ 0.86038413,  7.82138602],
    [ 7.35736969,  5.13258805],
    [ 0.47871581,  9.33930476],
    [ 0.23053498,  7.05967085],
    [ 7.32133123,  4.70066552],
]
TP3_1_Y = [-1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, -1]

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
    print(x1)
    print(x2)
    plt.plot(x1, x2)
    plt.show()

df = DataFrame(dict(x1=obtener_valores_de_columna(0, TP3_1_X), x2=obtener_valores_de_columna(1, TP3_1_X), y=TP3_1_Y))
colores = {-1:'red', 1:'blue'}
_, subplot = plt.subplots()
#print(TP3_1_X)
# Agrupa por los valores de Y
valores_agrupados = df.groupby('y')
for valor_Y, grupo in valores_agrupados:
    grupo.plot(ax=subplot, x='x1', y='x2', label=valor_Y, color=colores[valor_Y], kind='scatter')

theta = perceptron(TP3_1_X, TP3_1_Y, 0.5)
plot_decision_boundary(TP3_1_X, theta)
