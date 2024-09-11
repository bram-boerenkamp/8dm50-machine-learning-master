import numpy as np

def lsq(X, y):
    """
    Least squares linear regression
    :param X: Input data matrix
    :param y: Target vector
    :return: Estimated coefficient vector for the linear regression
    """

    # add column of ones for the intercept
    ones = np.ones((len(X), 1))
    X = np.concatenate((ones, X), axis=1)

    # calculate the coefficients
    beta = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))

    return beta

def predict(X, beta):
    """
    Predict target values
    :param X: Input data matrix
    :param beta: Coefficient vector
    :return: Predicted target values
    """

    # add column of ones for the intercept
    ones = np.ones((len(X), 1))
    X = np.concatenate((ones, X), axis=1)

    return np.dot(X, beta)

def mse(y_test, y_pred):
    """
    Mean squared error
    :param y_test: True target values
    :param y_pred: Predicted target values
    :return: Mean squared error
    """

    return np.mean((y_test - y_pred) ** 2)