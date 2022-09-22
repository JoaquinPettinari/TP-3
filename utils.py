#positive = "Yes"
#negative = "No"
positive = 1
negative = 0

def obtener_metricas(actualCreditabilities, predictionCreditabilities):
        TP = 0
        FP = 0 
        TN = 0 
        FN = 0
        
        for i, prediction in enumerate(predictionCreditabilities):
            actual = actualCreditabilities[i]
            if actual == positive and prediction == positive:
                TP += 1
            elif actual == negative and prediction == negative:
                TN += 1
            elif actual == positive and prediction == negative:
                FN += 1
            else:
                FP += 1   
        return [TP, FP, TN, FN]
    
def obtener_accuracy(TP, FP, TN, FN):
    return (TP + TN) / (TP + FP + TN + FN)

def obtener_precision(TP, FP):
    return TP / (TP + FP)