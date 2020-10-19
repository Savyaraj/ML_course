# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    # ***************************************************
    w =  np.dot(np.linalg.inv(np.dot(tx.T,tx)+ lambda_*np.identity(np.shape(tx)[1])),np.dot(tx.T,y))
    # loss
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e)) + lambda_*(np.inner(w,w))
    # ***************************************************
    
    return w,mse 