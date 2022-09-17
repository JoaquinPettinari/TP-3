import pandas as pd
from id3 import DecisionTreeClassifier

atributos = ["Account Balance","Payment Status of Previous Credit","Value Savings/Stocks","Length of current employment","Sex & Marital Status","Guarantors","Creditability"]

TrainingData = pd.read_csv("TrainingCompleto.csv", skiprows=1, header=None, names=atributos)
X_Training = TrainingData.iloc[:, :-1].values
Y_Training = TrainingData.iloc[:, -1].values.reshape(-1,1)

TestData = pd.read_csv("TestCompleto.csv", skiprows=1, header=None, names=atributos)
X_Test = TestData.iloc[:, :-1].values
Y_Test = TestData.iloc[:, -1].values.reshape(-1,1)

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
