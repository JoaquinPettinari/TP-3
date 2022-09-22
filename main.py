import pandas as pd
from arbol_decision import ID3, getPrediccionClase
from random_forest import RandomForest_Predict, getPrediccionClasses, RandomForest_Train
from utils import obtener_metricas, obtener_accuracy, obtener_precision
from pprint import pprint

Training_Data = pd.read_csv("TrainingCompleto.csv")
Test_Data = pd.read_csv("TestCompleto.csv")
primaryClass = Training_Data.keys()[-1]

ID3_Training_Tree = ID3(Training_Data,Training_Data,Training_Data.columns[:-1])
ID3_Test_Tree = ID3(Test_Data,Test_Data,Test_Data.columns[:-1])

actual_Y_Training = Training_Data.iloc[:, -1].tolist()
actual_Y_Test = Test_Data.iloc[:, -1].tolist()


print("ID3 - Training: ")
pprint(ID3_Training_Tree)
prediccion_Y_Training = getPrediccionClase(Training_Data,ID3_Training_Tree)
TP_ID3_Training, FP_ID3_Training, TN_ID3_Training, FN_ID3_Training = obtener_metricas(actual_Y_Training, prediccion_Y_Training)
print("Matriz: ")
print(TP_ID3_Training, FP_ID3_Training)
print(FN_ID3_Training, TN_ID3_Training)
print("ACCURACY: ", obtener_accuracy(TP_ID3_Training, FP_ID3_Training, TN_ID3_Training, FN_ID3_Training))

print("ID3 - Test")
pprint(ID3_Test_Tree)
prediccion_Y_Test = getPrediccionClase(Test_Data,ID3_Test_Tree)
TP_ID3_Test, FP_ID3_Test, TN_ID3_Test, FN_ID3_Test = obtener_metricas(actual_Y_Test, prediccion_Y_Test)
print("Matriz: ")
print(TP_ID3_Test, FP_ID3_Test)
print(FN_ID3_Test, TN_ID3_Test)
print("ACCURACY: ", obtener_accuracy(TP_ID3_Test, FP_ID3_Test, TN_ID3_Test, FN_ID3_Test))

print("------------------------------------------------------------------------------------------------------------")
print("Random Forest - Training")
RF_Training = RandomForest_Train(Training_Data, 20)
RF_Test = RandomForest_Train(Test_Data, 20)

pred_Y_RF_Training = getPrediccionClasses(Training_Data, RF_Training)
TP_RF_TRAINING, FP_RF_TRAINING, TN_RF_TRAINING, FN_RF_TRAINING = obtener_metricas(actual_Y_Training, pred_Y_RF_Training)
print("Matriz: ")
print(TP_RF_TRAINING, FP_RF_TRAINING)
print(FN_RF_TRAINING , TN_RF_TRAINING)
print("ACCURACY: ", obtener_accuracy(TP_RF_TRAINING, FP_RF_TRAINING, TN_RF_TRAINING, FN_RF_TRAINING))
print("PRECISION: ", obtener_precision(TP_RF_TRAINING, FP_RF_TRAINING))

    
print("Random Forest - Test")
pred_Y_RF_Test = getPrediccionClasses(Test_Data, RF_Test)
TP_RF_Test, FP_RF_Test, TN_RF_Test, FN_RF_Test = obtener_metricas(actual_Y_Test, pred_Y_RF_Test)
print("Matriz: ")
print(TP_RF_Test, FP_RF_Test)
print(FN_RF_Test, TN_RF_Test)
print("ACCURACY: ", obtener_accuracy(TP_RF_Test, FP_RF_Test, TN_RF_Test, FN_RF_Test))
print("PRECISION: ", obtener_precision(TP_RF_Test, FP_RF_Test))