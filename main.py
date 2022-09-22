import pandas as pd
from arbol_decision import ID3, getActualClass
from utils import obtener_metricas, obtener_accuracy, obtener_precision
from pprint import pprint

Training_Data = pd.read_csv("TrainingCompleto.csv")
Test_Data = pd.read_csv("TestCompleto.csv")
primaryClass = Training_Data.keys()[-1]

ID3_Training_Tree = ID3(Training_Data,Training_Data,Training_Data.columns[:-1])
ID3_Test_Tree = ID3(Test_Data,Test_Data,Test_Data.columns[:-1])

prediccion_Y_Training = Training_Data.iloc[:, -1].tolist()
prediccion_Y_Test = Test_Data.iloc[:, -1].tolist()


print("ID3 - Training: ")
pprint(ID3_Training_Tree)
actual_Y_Training = getActualClass(Training_Data,ID3_Training_Tree)
TP_ID3_Training, FP_ID3_Training, TN_ID3_Training, FN_ID3_Training = obtener_metricas(prediccion_Y_Training, actual_Y_Training)
print("Matriz: ")
print(TP_ID3_Training, FP_ID3_Training)
print(FN_ID3_Training, TN_ID3_Training)
print("ACCURACY: ", obtener_accuracy(TP_ID3_Training, FP_ID3_Training, TN_ID3_Training, FN_ID3_Training))

print("ID3 - Test")
pprint(ID3_Test_Tree)
actual_Y_Test = getActualClass(Test_Data,ID3_Test_Tree)
TP_ID3_Test, FP_ID3_Test, TN_ID3_Test, FN_ID3_Test = obtener_metricas(prediccion_Y_Test, actual_Y_Test)
print("Matriz: ")
print(TP_ID3_Test, FP_ID3_Test)
print(FN_ID3_Test, TN_ID3_Test)
print("ACCURACY: ", obtener_accuracy(TP_ID3_Test, FP_ID3_Test, TN_ID3_Test, FN_ID3_Test))