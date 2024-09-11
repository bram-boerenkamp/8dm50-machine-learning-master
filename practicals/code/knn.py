# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 13:59:44 2024

@author: 20192122
"""
import numpy as np

def train_test_split(X,y,test_size=0.5): 
    """
    Splits the data (X) and target (y) into training and test sets.

    :param X: numpy array or matrix, the feature data (input variables).
    :param y: numpy array, the target variable (labels or output).
    :param test_size: float, optional (default=0.5).
                       The proportion of the dataset to include in the training set.
                       Should be a value between 0 and 1. 
    :return: 
             - X_train: numpy array, the feature data for the training set.
             - X_test: numpy array, the feature data for the test set.
             - y_train: numpy array, the target data for the training set.
             - y_test: numpy array, the target data for the test set.
    """
    indices = np.arange(X.shape[0]) 
    np.random.shuffle(indices) 
    split_idx = int(X.shape[0] * (1 - test_size))
    X_train, X_test = X[indices[:split_idx]], X[indices[split_idx:]]
    y_train, y_test = y[indices[:split_idx]], y[indices[split_idx:]]
    return X_train, X_test, y_train, y_test

class KNN_general:
    def __init__(self, k=0):
        """
        Initialize the KNN model.

        :param k: int, optional (default=0).
          The number of nearest neighbors to consider for predictions.
        """
        self.k = k
    
    def fit(self, X_train, y_train):
        """
        Fit the KNN model on the training data.
        
        :param X_train: numpy array of shape (n_samples, n_features).
                        The training data where each row represents a sample and each column represents a feature.
        :param y_train: numpy array of shape (n_samples,).
                        The target labels corresponding to the training data.
        """
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X_test, kind="regression"):
        """
        Predict the target values for the given test data using the fitted KNN model.

        :param X_test: numpy array of shape (n_samples, n_features).
                       The test data to predict the target values for.
        :param kind: str, optional (default="regression").
                     The type of KNN prediction to perform: either "regression" or "classification".
                     If "regression", the mean of the k-nearest neighbors is used for prediction.
                     If "classification", the mode (most frequent label) of the k-nearest neighbors is used.

        :return: numpy array 
                 The predicted target values for the test data.
        """
        predictions = [] 
        for test_point in X_test:
            euclidean_distances = np.linalg.norm(self.X_train - test_point, axis=1) 
            nearest_indices = np.argsort(euclidean_distances)[:self.k] 
            nearest_labels = self.y_train[nearest_indices] 
            if kind == "classification": 
                prediction = np.argmax(np.bincount(nearest_labels)) 
            elif kind == "regression":
                prediction = np.mean(nearest_labels) 
            predictions.append(prediction)
        return np.array(predictions)
    
    def accuracy(self, y_true, y_pred):
        """
        Calculate the accuracy of the model's predictions.
        
        :param y_true: numpy array
                       The true target values.
        :param y_pred: numpy array
                       The predicted target values.
        
        :return: float.
                 The proportion of correctly predicted values out of the total predictions.
        """
        accurate_predicted=np.sum(y_true == y_pred) / len(y_true) 
        return accurate_predicted

