import pandas as pd
from id3 import DecisionTreeClassifier

col_names = ["Account Balance","Payment Status of Previous Credit","Value Savings/Stocks","Length of current employment","Sex & Marital Status","Guarantors","Creditability"]
data = pd.read_csv("TrainingCompleto.csv", skiprows=1, header=None, names=col_names)
X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values.reshape(-1,1)

classifier = DecisionTreeClassifier(col_names, min_samples_split=3, max_depth=3)
classifier.fit(X,Y)
prediccion_Y = classifier.predict(X)
#classifier.print_tree()

actual_Y = data.iloc[:, -1].tolist()
print(prediccion_Y)
print(actual_Y)
TP, FP, TN, FN = classifier.obtener_metricas(prediccion_Y, actual_Y)
accuracy = classifier.obtener_accuracy(TP, FP, TN, FN)
from sklearn.metrics import accuracy_score
print(accuracy)
print(accuracy_score(actual_Y, prediccion_Y))

