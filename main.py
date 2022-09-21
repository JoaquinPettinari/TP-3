import pandas as pd
from arbol_decision import id3, imprimir_arbol, evaluate
from utils import obtener_metricas, obtener_accuracy, obtener_precision

Training_Data = pd.read_csv("Test6.csv")
Test_Data = pd.read_csv("Test6.csv")

actual_Y_Training = Training_Data.iloc[:, -1].tolist()
primaryClass = Training_Data.keys()[-1]
tree = id3(Training_Data, primaryClass)

imprimir_arbol(tree)
prediccion_ID3_Training = evaluate(tree, Test_Data, primaryClass)
print("--------")
print(actual_Y_Training)
print(prediccion_ID3_Training)
TP_ID3_Training, FP_ID3_Training, TN_ID3_Training, FN_ID3_Training = obtener_metricas(prediccion_ID3_Training, actual_Y_Training)
print(TP_ID3_Training, FP_ID3_Training)
print(FN_ID3_Training, TN_ID3_Training)
print(obtener_accuracy(TP_ID3_Training, FP_ID3_Training, TN_ID3_Training, FN_ID3_Training))
