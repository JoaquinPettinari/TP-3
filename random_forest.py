# File name: RandomForest.py

import numpy as np
import time
from collections import Counter
from arbol_decision import ID3, predecir
import scipy.stats as sps


def random_samples(X, y):
    """
    random_samples method
    :param X: {array-like}
    :param y: {array-like}
    :return: {array-like}, {array-like}
    """
    n_sample = X.shape[0]
    indices = np.random.choice(n_sample, size=n_sample, replace=True)
    return np.array(X)[indices.astype(int)], np.array(y)[indices.astype(int)]


def RandomForest_Train(dataset,number_of_Trees):
    #Create a list in which the single forests are stored
    random_forest_sub_tree = []
    
    #Create a number of n models
    for i in range(number_of_Trees):
        #Create a number of bootstrap sampled datasets from the original dataset 
        bootstrap_sample = dataset.sample(frac=1,replace=True)
        
        #Create a training and a testing datset by calling the train_test_split function
        #bootstrap_training_data = train_test_split(bootstrap_sample)[0]
        #bootstrap_testing_data = train_test_split(bootstrap_sample)[1] 
        
        #Grow a tree model for each of the training data
        #We implement the subspace sampling in the ID3 algorithm itself. Hence take a look at the ID3 algorithm above!
        random_forest_sub_tree.append(ID3(bootstrap_sample,bootstrap_sample,bootstrap_sample.drop(labels=['Creditability'],axis=1).columns))
        
    return random_forest_sub_tree


        
 

#######Predict a new query instance###########
def RandomForest_Predict(query,random_forest,default='p'):
    predictions = []
    for tree in random_forest:
        predictions.append(predecir(query,tree,default))
    return sps.mode(predictions, keepdims=True)[0][0]





#######Test the model on the testing data and return the accuracy###########
def getPrediccionClasses(data,random_forest, primaryClass):
    predictions = []
    for i in range(len(data)):
        query = data.iloc[i,:].drop(primaryClass).to_dict()
        predictions.append(RandomForest_Predict(query,random_forest,default='p'))
    #accuracy = sum(predictions == data['Creditability'])/len(data)*100
    #print('The prediction accuracy is: ',sum(data['predictions'] == data['target'])/len(data)*100,'%')
    return predictions
        