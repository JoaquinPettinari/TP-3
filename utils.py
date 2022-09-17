def obtener_metricas(predictionCreditabilities, actualCreditabilities):
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
                FN += 1
            else:
                FP += 1   
        return [TP, FP, TN, FN]
    
def obtener_accuracy(TP, FP, TN, FN):
    return (TP + TN) / (TP + FP + TN + FN)

def obtener_precision(TP, FP):
    return TP / (TP + FP)