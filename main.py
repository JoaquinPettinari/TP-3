import pandas as pd
from id3 import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from random_forest import RandomForest
import numpy as np
import time
from sklearn import datasets

atributos = ["Account Balance","Payment Status of Previous Credit","Value Savings/Stocks","Length of current employment","Sex & Marital Status","Guarantors","Creditability"]

TrainingData = pd.read_csv("TrainingCompleto.csv", skiprows=1, header=None, names=atributos)
X_Training = TrainingData.iloc[:, :-1].values
Y_Training = TrainingData.iloc[:, -1].values.reshape(-1,1)
actual_Training = TrainingData.iloc[:, -1].tolist()
# Load data

TestData = pd.read_csv("TestCompleto.csv", skiprows=1, header=None, names=atributos)
X_Test = TestData.iloc[:, :-1].values
Y_Test = TestData.iloc[:, -1].values.reshape(-1,1)

"""
Increasing the number of trees until the accuracy begins to stop showing improvement. 
This may take a long time, but will not over-fit the training data.
"""
data = datasets.load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=1234)

start = time.time()
"""
Increasing the number of trees until the accuracy begins to stop showing improvement. 
This may take a long time, but will not over-fit the training data.
"""
rm = RandomForest(n_trees=5, max_depth=10)
rm.fit(X_Training, actual_Training)

y_prediction = rm.predict(X_Test)
#print('Accuracy is: %.2f%%' % (np.sum(y_test == y_prediction) / len(y_test) * 100))

end = time.time()  # ----------------------------------------------
print('\n ----------\n Execution Time: {%f}' % ((end - start) / 1000) + ' seconds.')

"""

arbol_Training = DecisionTreeClassifier(atributos, min_samples_split=3, max_depth=3)
arbol_Training.fit(X_Training,Y_Training)
prediccion_Training = arbol_Training.predict(X_Training)

arbol_Test = DecisionTreeClassifier(atributos, min_samples_split=3, max_depth=3)
arbol_Test.fit(X_Test,Y_Test)
prediccion_Test = arbol_Test.predict(X_Test)

actual_Training = TrainingData.iloc[:, -1].tolist()
TP_Training, FP_Training, TN_Training, FN_Training = arbol_Training.obtener_metricas(prediccion_Training, actual_Training)
accuracy_Training = arbol_Training.obtener_accuracy(TP_Training, FP_Training, TN_Training, FN_Training)
precision_Training = arbol_Training.obtener_precision(TP_Training, FP_Training)

actual_Test = TestData.iloc[:, -1].tolist()
TP_Test, FP_Test, TN_Test, FN_Test = arbol_Test.obtener_metricas(prediccion_Test, actual_Test)
accuracy_Test = arbol_Test.obtener_accuracy(TP_Test, FP_Test, TN_Test, FN_Test)
precision_Test = arbol_Test.obtener_precision(TP_Test, FP_Test)

print("Training: ")
print("Accuracy: ", accuracy_Training)
print("Precision: ", precision_Training)

print("Test: ")
print("Accuracy: ", accuracy_Test)
print("Precision: ", precision_Test)
"""