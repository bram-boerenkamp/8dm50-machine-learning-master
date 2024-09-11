# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 14:16:05 2024

@author: 20192122
"""
import numpy as np

def mse(y_test, y_pred):
    """
    Mean squared error
    :param y_test: True target values
    :param y_pred: Predicted target values
    :return: Mean squared error
    """

    return np.mean((y_test - y_pred) ** 2)