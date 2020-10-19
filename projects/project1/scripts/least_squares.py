# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def least_squares(y, tx):
    """calculate the least squares solution."""
    # ***************************************************
    # returns mse, and optimal weights
    optimum = np.dot(np.linalg.inv(np.dot(tx.T,tx)),np.dot(tx.T,y))
    # The error
    res = y- np.dot(tx,optimum);
    #Mean squared error
    MSE = (np.dot(res,res))/(2*np.shape(y)[0])
    # ***************************************************
    return optimum,MSE
