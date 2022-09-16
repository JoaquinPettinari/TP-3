import pandas 
import numpy as np
class Set:
    def __init__(self):
        self.atributeColumns = ["Account Balance", "Payment Status of Previous Credit","Purpose", "Value Savings/Stocks", "Length of current employment","Instalment per cent", "Most valuable available asset", "Occupation", "Duration in Current address", "Sex & Marital Status", "Guarantors"]
        self.df = pandas.read_excel("prestamos_bancarios_alemanes_1994.xls")
        
        self.noCreditabilityListAllColumns = self.getColumnsEqualsToValue(self.df, "Creditability", 0).head(300)
        self.creditabilityListAllColumns = self.getColumnsEqualsToValue(self.df, "Creditability", 1).head(300)
        
        self.noCreditabilityList = self.noCreditabilityListAllColumns.loc[:, self.atributeColumns]
        self.credibilityList = self.creditabilityListAllColumns.loc[:, self.atributeColumns]
        
        self.noCreditabilityColumnList = self.noCreditabilityListAllColumns.loc[:, "Creditability"]
        self.creditabilityColumnList = self.creditabilityListAllColumns.loc[:, "Creditability"]
        
        self.noCreditabilityList_Training = self.getCutList(self.noCreditabilityList, 0.8, False)
        self.creditabilityList_Training = self.getCutList(self.credibilityList, 0.8, False)
        
        self.noCreditabilityAttributesList_Test = self.getCutList(self.noCreditabilityList, 0.2, True)
        self.noCreditabilityColumn_Test = self.getCutList(self.noCreditabilityColumnList, 0.2, True)

        self.creditabilityAttributesList_Test = self.getCutList(self.credibilityList, 0.2, True)
        self.creditabilityColumnValues_Test = self.getCutList(self.creditabilityColumnList, 0.2, True)

    def getColumnsEqualsToValue(self, list, key , value):
        return list[list[key].isin([value])]
    
    def getCutList(self, list, percentage, head):
        trainingLength = int(len(list) * percentage)
        return list.head(trainingLength) if head else list.tail(trainingLength)

    def getTrainingLists(self):
        return [self.creditabilityList_Training, self.noCreditabilityList_Training]
    
    def getTestLists(self):
        return [self.creditabilityAttributesList_Test, self.noCreditabilityAttributesList_Test]
        
    