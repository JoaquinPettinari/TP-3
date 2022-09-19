# File name: RandomForest.py

import numpy as np
import time
from collections import Counter
from decision_tree import DecisionTree #, most_common_class


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
            d_tree = DecisionTree(self.atributos, self.min_samples_split, self.max_depth)
            # Chose random samples
            x_sample, y_sample = random_samples(X, y)
            # Fit the tree - Train a CART model on each sample.
            d_tree.entrenar(x_sample, y_sample)
            # Store the model in a list of models
            self.models.append(d_tree)

    def predict(self, X):
        """
        predict method
        :param X: {array-like}
        :return: {array-like}
        """
        """ Calculate the average prediction from each model """
        # Make a predictions - Predict for each model in the list:
        # for example 3 trees with 4 samples will give: [[1111] [0000] [1111]]
        tree_predictions = np.array([model.predict(X) for model in self.models])
        # Swap the predictions array axes:
        # for example we convert the above example to [[101] [101] [101] [101]]
        swapped_predictions = np.swapaxes(tree_predictions, 0, 1)        
        # Majority Voting: get the most common class of the swapped array        
        return [most_common_class(prediction) for prediction in swapped_predictions]    
