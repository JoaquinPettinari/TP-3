import pandas as pd
from decision_tree import DecisionTree
from random_forest import RandomForest
from utils import obtener_accuracy, obtener_metricas, obtener_precision

atributos = ["Account Balance","Payment Status of Previous Credit","Value Savings/Stocks","Length of current employment","Sex & Marital Status","Guarantors","Creditability"]

TrainingData = pd.read_csv("TrainingCompleto.csv", skiprows=1, header=None, names=atributos)
X_Training = TrainingData.iloc[:, :-1].values
Y_Training = TrainingData.iloc[:, -1].values.reshape(-1,1)

TestData = pd.read_csv("TestCompleto.csv", skiprows=1, header=None, names=atributos)
X_Test = TestData.iloc[:, :-1].values
Y_Test = TestData.iloc[:, -1].values.reshape(-1,1)

arbol_Training = DecisionTree(atributos, min_samples_split=3, max_depth=3)
arbol_Training.fit(X_Training,Y_Training)
actual_Training = TrainingData.iloc[:, -1].tolist()
prediccion_Training = arbol_Training.predict(X_Training)


arbol_Test = DecisionTree(atributos, min_samples_split=3, max_depth=3)
arbol_Test.fit(X_Test,Y_Test)
actual_Test = TestData.iloc[:, -1].tolist()
prediccion_ID3_Test = arbol_Test.predict(X_Test)

TP_ID3_Training, FP_ID3_Training, TN_ID3_Training, FN_ID3_Training = obtener_metricas(prediccion_Training, actual_Training)
accuracy_ID3_Training = obtener_accuracy(TP_ID3_Training, FP_ID3_Training, TN_ID3_Training, FN_ID3_Training)
precision_ID3_Training = obtener_precision(TP_ID3_Training, FP_ID3_Training)


TP_ID3_Test, FP_ID3_Test, TN_ID3_Test, FN_ID3_Test = obtener_metricas(prediccion_ID3_Test, actual_Test)
accuracy_ID3_Test = obtener_accuracy(TP_ID3_Test, FP_ID3_Test, TN_ID3_Test, FN_ID3_Test)
precision_ID3_Test = obtener_precision(TP_ID3_Test, FP_ID3_Test)

print("ID3 - Training")
print("Arbol: ", arbol_Training.print_tree())
print("Matriz confusion: ")
print(TP_ID3_Training, FP_ID3_Training)
print(FN_ID3_Training, TN_ID3_Training)
print("Accuracy: ", accuracy_ID3_Training)
print("Precision: ", precision_ID3_Training)
print("\n")
print("ID3 - Test")
print("Arbol: ", arbol_Test.print_tree())
print("Matriz confusion: ")
print(TP_ID3_Test, FP_ID3_Test)
print(FN_ID3_Test, TN_ID3_Test)
print("Accuracy: ", accuracy_ID3_Test)
print("Precision: ", precision_ID3_Test)
print("\n ------------------- \n" )


random_Forest_Training = RandomForest(atributos, n_trees=6, max_depth=10)
random_Forest_Training.fit(X_Training,Y_Training)
Y_RF_Prediction_Training = random_Forest_Training.predict(X_Training)

TP_RF_Training, FP_RF_Training, TN_RF_Training, FN_RF_Training = obtener_metricas(Y_RF_Prediction_Training, actual_Training)
accuracy_RF_Training = obtener_accuracy(TP_RF_Training, FP_RF_Training, TN_RF_Training, FN_RF_Training)
precision_RF_Training = obtener_precision(TP_RF_Training, FP_RF_Training)

random_Forest_Test = RandomForest(atributos, n_trees=6, max_depth=10)
random_Forest_Test.fit(X_Test, Y_Test)
Y_RF_Prediction_Test = random_Forest_Test.predict(X_Test)

TP_RF_Test, FP_RF_Test, TN_RF_Test, FN_RF_Test = obtener_metricas(Y_RF_Prediction_Test, actual_Test)
accuracy_RF_Test = obtener_accuracy(TP_RF_Test, FP_RF_Test, TN_RF_Test, FN_RF_Test)
precision_RF_Test = obtener_precision(TP_RF_Test, FP_RF_Test)

print("Random Forest - Training")
print("Matriz confusion: ")
print(TP_RF_Training, FP_RF_Training) 
print(FN_RF_Training,TN_RF_Training)
print("Accuracy: ", accuracy_RF_Training)
print("Precision: ",precision_RF_Training)
print("\n")
print("Random Forest - Test")
print("Matriz confusion: ")
print(TP_RF_Test, FP_RF_Test)
print(FN_RF_Test, TN_RF_Test)
print("Accuracy: ", accuracy_RF_Test)
print("Precision: ", precision_RF_Test)