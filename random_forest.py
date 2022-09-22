# File name: RandomForest.py

import numpy as np
import time
from collections import Counter
from arbol_decision import ArbolDecision #, most_common_class


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

def most_common_class(y):
    """
    most_common_class method
    :param y: {array-like}
    :return: {int}
    """
    common_class = Counter(y)
    # Get a list of tuple of most common labels
    most_common_class_list = common_class.most_common(1)
    # Return the first tuple and then the first dimension
    most_common = most_common_class_list[0][0]
    return most_common

class RandomForest:
    """ Random Forests are an improvement over bagged decision trees """
    """
    Bootstrap Aggregation - Bagging 
        Bagging is the application of the Bootstrap procedure to a high-variance machine learning algorithm, 
        typically decision trees.
        
    Bagging of the CART algorithm would work as follows:
        1. Create random sub-samples of the dataset with replacement.
        2. Train a CART model on each sample.
        3. Calculate the average prediction from each model.
        
    Bagging can be used for classification and regression problems.
    """

    def __init__(self, atributos, n_trees=0, min_samples_split=2, max_depth=0):
        """
        Class Constructor
        :param n_trees: {int}
        :param min_samples_split: {int}
        :param max_depth: {int}
        """
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.models = []
        self.atributos = atributos

    def entrenar(self, X, y):
        """
        fit method
        :param X: {array-like}
        :param y: {array-like}
        :return: None
        """
        for _ in range(self.n_trees):
            d_tree = ArbolDecision(self.atributos, self.min_samples_split, self.max_depth)
            # Chose random samples
            x_sample, y_sample = random_samples(X, y)
            # Fit the tree - Train a CART model on each sample.
            d_tree.entrenar(x_sample, y_sample)
            # Store the model in a list of models
            self.models.append(d_tree)

    def predecir(self, X):
        """
        predict method
        :param X: {array-like}
        :return: {array-like}
        """
        """ Calculate the average prediction from each model """
        # Make a predictions - Predict for each model in the list:
        # for example 3 trees with 4 samples will give: [[1111] [0000] [1111]]
        tree_predictions = np.array([model.predecir(X) for model in self.models])
        # Swap the predictions array axes:
        # for example we convert the above example to [[101] [101] [101] [101]]
        swapped_predictions = np.swapaxes(tree_predictions, 0, 1)        
        # Majority Voting: get the most common class of the swapped array        
        return [most_common_class(prediction) for prediction in swapped_predictions]    


def RandomForest_Train(dataset,number_of_Trees):
    #Create a list in which the single forests are stored
    random_forest_sub_tree = []
    
    #Create a number of n models
    for i in range(number_of_Trees):
        #Create a number of bootstrap sampled datasets from the original dataset 
        bootstrap_sample = dataset.sample(frac=1,replace=True)
        
        #Create a training and a testing datset by calling the train_test_split function
        bootstrap_training_data = train_test_split(bootstrap_sample)[0]
        bootstrap_testing_data = train_test_split(bootstrap_sample)[1] 
        
        
        #Grow a tree model for each of the training data
        #We implement the subspace sampling in the ID3 algorithm itself. Hence take a look at the ID3 algorithm above!
        random_forest_sub_tree.append(ID3(bootstrap_training_data,bootstrap_training_data,bootstrap_training_data.drop(labels=['target'],axis=1).columns))
        
    return random_forest_sub_tree


        
random_forest = RandomForest_Train(dataset,50)

 

#######Predict a new query instance###########
def RandomForest_Predict(query,random_forest,default='p'):
    predictions = []
    for tree in random_forest:
        predictions.append(predict(query,tree,default))
    return sps.mode(predictions)[0][0]


query = testing_data.iloc[0,:].drop('target').to_dict()
query_target = testing_data.iloc[0,0]
print('target: ',query_target)
prediction = RandomForest_Predict(query,random_forest)
print('prediction: ',prediction)



#######Test the model on the testing data and return the accuracy###########
def RandomForest_Test(data,random_forest):
    data['predictions'] = None
    for i in range(len(data)):
        query = data.iloc[i,:].drop('target').to_dict()
        data.loc[i,'predictions'] = RandomForest_Predict(query,random_forest,default='p')
    accuracy = sum(data['predictions'] == data['target'])/len(data)*100
    #print('The prediction accuracy is: ',sum(data['predictions'] == data['target'])/len(data)*100,'%')
    return accuracy
        
        
        
RandomForest_Test(testing_data,random_forest)