from xml.etree.ElementInclude import include
import pandas
import sys
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Punto 1:
atributeColumns = ["Account Balance", "Payment Status of Previous Credit","Purpose", "Value Savings/Stocks", "Length of current employment","Instalment per cent", "Most valuable available asset", "Occupation", "Duration in Current address", "Sex & Marital Status", "Guarantors"]
df = pandas.read_excel("prestamos_bancarios_alemanes_1994.xls")

# Dada una lista, key y value devuelve las columnas que repiten
def getColumnsEqualsToValue(list, key , value):
    return list[list[key].isin([value])]

# Devuelve una lista de training. Hardcodeado el 20% de la lista
def getCutList(list, percentage, head):
    trainingLength = int(len(list) * percentage)
    return list.head(trainingLength) if head else list.tail(trainingLength)

# Usa el atributo objetivo y calcula cuantos hay del total. Ejemplo 300/600
def getCreditabilityProb(list, total):
    return len(list) / total

# Devuelve un valor (0,x). Para calcular la probabilidad de ese value en una key
def probabilityOfMatches(list, key, value, k):
    return (len(getColumnsEqualsToValue(list, key, value)) + 1) / (len(list) + k)

def getKValue(column):
    return len(df.drop_duplicates(subset=[column]))

def getMetrics(tn, fp, fn, tp, predictionCreditabilities, actualCreditabilities):
    # Punto 6
    print("------------")
    print("CONFUSION MATRIX:")    
    print(tp, fp)
    print(fn, tn)
    showConfusionMatrix(actualCreditabilities, predictionCreditabilities)    
    
    # Punto 7
    accuracyBySkLearn = accuracy_score(actualCreditabilities, predictionCreditabilities)
    accuracyByFormula = (tp + tn) / (tp + tn + fp + fn)
    print("------------")
    print("ACCURACY: ", accuracyBySkLearn, accuracyByFormula)

    precisionBySKLearn = precision_score(actualCreditabilities, predictionCreditabilities)
    precisionByFormula = (tp / (tp + fp))
    print("------------")
    print("PRECISION: ", precisionBySKLearn, precisionByFormula)

    F1ScoreBySKLearn = f1_score(actualCreditabilities, predictionCreditabilities)
    F1ScoreByFormula = ((2 * tp) / ((2 * tp) + fp + fn))
    print("------------")
    print("F1 Score: ", F1ScoreBySKLearn, F1ScoreByFormula)

    trueRatePositiveByFormula = tp / (tp + fn)
    falseRatePositiveByFormula = fp / (fp + tn)
    print("------------")
    print("True Rate Positive: ", trueRatePositiveByFormula)
    print("False Rate Positive: ", falseRatePositiveByFormula)

# Dada una lista de valores devuelve la probabilidad de credibilidad y no credibilidad
def getClassProb(rowValue, creditabilityList, noCreditabilityList ):
    # Esto es para obtener la cantidad de "Creditability" (1 o 0) sobre el total de todos los datos 300/600. Esto devuelve 0.5. 
    # Pero en caso que haya 350/600 y 250/600 queda automatizado
    totalLengthLists = len(creditabilityList) + len(noCreditabilityList)
    pNoCreditability = getCreditabilityProb(noCreditabilityList, totalLengthLists)
    pCreditability = getCreditabilityProb(creditabilityList, totalLengthLists)

    for i, primaryColumn in enumerate(atributeColumns):
        
        # Hacer una función que reciba la lista de noCredibilidad o credibilidad y que devuelva la probabilidad / 300. Así no repite código
        pNoCreditability *= probabilityOfMatches(noCreditabilityList, primaryColumn, rowValue[i], getKValue(primaryColumn))
        pCreditability *= probabilityOfMatches(creditabilityList, primaryColumn, rowValue[i], getKValue(primaryColumn))

    return pNoCreditability, pCreditability

def getRelativeProb(noCreditability, creditability):
    probTotal = noCreditability + creditability
    return creditability / probTotal

"""
def getConfusionMatrix(actualCreditabilities, predictionCreditabilities):
    return confusion_matrix(actualCreditabilities, predictionCreditabilities).ravel()
"""

def getConfusionMatrix(predictionCreditabilities, actualCreditabilities):
    TP = 0
    FP = 0 
    TN = 0 
    FN = 0
    
    for i, actual in enumerate(actualCreditabilities):
        prediction = predictionCreditabilities[i]
        if prediction == 1 and actual == 1:
            TP += 1
        elif prediction == 0 and actual == 0:
            TN += 1
        elif prediction == 1 and actual == 0:
            FP += 1
        else:
            FN += 1   
    return [TP, FP, TN, FN]
# Genera la matriz de confusion
def calculateActualClasses(atributtesValues, threshold):
    actualCreditabilities = []
    for rowValues in atributtesValues:
        pTotalNoCredibility, pTotalCredibility = getClassProb(rowValues, creditabilityList_Training, noCreditabilityList_Training)

        creditabilityProb = getRelativeProb(pTotalNoCredibility, pTotalCredibility)
        actualCreditabilities.append(1 if creditabilityProb >= threshold else 0)
    
    return actualCreditabilities
    
def showROCurve(fprList, trpList):
    sns.set()
    plt.plot(fprList, trpList, linestyle= "-", color = "k")
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title('ROC curve');    
    plt.show()

def showConfusionMatrix(actualCreditabilities, predictionCreditabilities):
    conmat = confusion_matrix(actualCreditabilities, predictionCreditabilities)
    val = np.mat(conmat)
    classnames = list(set(actualCreditabilities))
    df_cm = pandas.DataFrame(
        val, index=classnames, columns=classnames
    )
    plt.figure()
    heatmap = sns.heatmap(df_cm, annot=True, cmap="Blues")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha="right")
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha="right")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title("Confusion Matrix")
    plt.show()   
# Para el punto 2 quedarse con un 20% de los 300 estos.
noCreditabilityListAllColumns = getColumnsEqualsToValue(df, "Creditability", 0).head(300)
creditabilityListAllColumns = getColumnsEqualsToValue(df, "Creditability", 1).head(300)

# Estos son los valores de los atributos
noCreditabilityList = noCreditabilityListAllColumns.loc[:, atributeColumns]
credibilityList = creditabilityListAllColumns.loc[:, atributeColumns]

# Estos son los valores de la columna creditability (0,1)
noCreditabilityColumnList = noCreditabilityListAllColumns.loc[:, "Creditability"]
creditabilityColumnList = creditabilityListAllColumns.loc[:, "Creditability"]

# Punto 2
noCreditabilityList_Training = getCutList(noCreditabilityList, 0.8, False)
creditabilityList_Training = getCutList(credibilityList, 0.8, False)

noCreditabilityAtributtesList_Test = getCutList(noCreditabilityList, 0.2, True)
noCreditabilityColumn_Test = getCutList(noCreditabilityColumnList, 0.2, True)

creditabilityAtributtesList_Test = getCutList(credibilityList, 0.2, True)
creditabilityColumnValues_Test = getCutList(creditabilityColumnList, 0.2, True)

def main():    
    # Punto 3 y 4. Prueba clasificador de bayes.    
    trueRatePositiveList = []
    falseRatePositiveList = []
       
    concatenateAtributtesValues = np.concatenate((creditabilityAtributtesList_Test, noCreditabilityAtributtesList_Test ))   
    predictionCreditabilities = np.concatenate((creditabilityColumnValues_Test, noCreditabilityColumn_Test))
    
    for threshold in [5,10,20,30,40,50,60,70,80,90,95]:
        actualCreditabilities = calculateActualClasses(concatenateAtributtesValues, threshold / 100)
        TP, FP, TN, FN = getConfusionMatrix(predictionCreditabilities, actualCreditabilities)
        trueRatePositiveList.append(TP / (TP + FN))
        falseRatePositiveList.append(FP / (FP + TN))
    
    # Punto 8
    showROCurve(falseRatePositiveList, trueRatePositiveList)    

if __name__ == '__main__':
    sys.exit(int(main() or 0))