from cProfile import label
from imagenes import obtener_conjuntos
from r2 import mostrar_puntos_en_plano, obtener_puntos, obtener_valores_de_columna, perceptron, plot_decision_boundary, trazar_linea
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from utils import obtener_conjuntos_de_datos


conjuntos_training_X, conjuntos_training_Y, conjuntos_test_X, conjuntos_test_Y = obtener_conjuntos()

clf = svm.SVC()
clf.fit(conjuntos_training_X, conjuntos_training_Y)
pred_conjunto_test = clf.predict(conjuntos_test_X)
print(confusion_matrix(conjuntos_test_Y, pred_conjunto_test, labels=[0,1,2]))
print(f"Accuracy: {100*accuracy_score(conjuntos_test_Y, pred_conjunto_test)}%")
print(f"Precision: {100*precision_score(conjuntos_test_Y, pred_conjunto_test, average='micro')}%")


"""
trazar_linea()
puntos = obtener_puntos()
mostrar_puntos_en_plano(puntos)
theta = perceptron(TP3_1_X, TP3_1_Y, 0.5)
plot_decision_boundary(TP3_1_X, theta)
"""
