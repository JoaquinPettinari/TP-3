#positive = "Yes"
#negative = "No"
positive = 1
negative = 0

def obtener_metricas(predictionCreditabilities, actualCreditabilities):
        TP = 0
        FP = 0 
        TN = 0 
        FN = 0
        
        for i, actual in enumerate(actualCreditabilities):
            prediction = predictionCreditabilities[i]
            if prediction == positive and actual == positive:
                TP += 1
            elif prediction == negative and actual == negative:
                TN += 1
            elif prediction == positive and actual == negative:
                FN += 1
            else:
                FP += 1   
        return [TP, FP, TN, FN]
    
def obtener_accuracy(TP, FP, TN, FN):
    return (TP + TN) / (TP + FP + TN + FN)

def obtener_precision(TP, FP):
    return TP / (TP + FP)