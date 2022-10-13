from perceptron import perceptron
from sklearn import datasets
import matplotlib.pyplot as plt
from pandas import DataFrame

TP3_1_X = [
    [10.02488611,  5.08333171],
    [ 6.86675981,  5.96880721],
    [ 0.23989814,  9.28353092],
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
    [ 7.63415083,  4.70066552],
]
TP3_1_Y = [0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0]

def plot_decision_boundary(X, theta):
    
    # X --> Inputs
    # theta --> parameters
    
    # The Line is y=mx+c
    # So, Equate mx+c = theta0.X0 + theta1.X1 + theta2.X2
    # Solving we find m and c
    x1 = [min(X[:,0]), max(X[:,0])]
    m = -theta[1]/theta[2]
    c = -theta[0]/theta[2]
    x2 = m*x1 + c
    
    plt.plot(X[:, 0][TP3_1_Y==0], X[:, 1][TP3_1_Y==0], "r^")
    plt.plot(X[:, 0][TP3_1_Y==1], X[:, 1][TP3_1_Y==1], "bs")
    plt.xlabel("feature 1")
    plt.ylabel("feature 2")
    plt.title("Perceptron Algorithm")
    plt.plot(x1, x2, 'y-')
    plt.show()

X, y = datasets.make_blobs(n_samples=20, centers=2, n_features=2, center_box= (0.5,10))
df = DataFrame(dict(x=[i[0] for i in TP3_1_X], y=[i[1] for i in TP3_1_X], label=TP3_1_Y))
colores = {0:'red', 1:'blue'}
print(df)
_, ax = plt.subplots()
print(ax)
grouped = df.groupby('label')
print(grouped)
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colores[key])
plt.show()
"""
print(df)
theta, miss_l = perceptron(X, y, 0.5, 100)
plot_decision_boundary(X, theta)
"""