# -*- coding: utf-8 -*-


#Ridge Regression


import numpy as np


def ridge_regression(y, tx, lambda_):
    # The optimum w
    w =  np.dot(np.linalg.inv(np.dot(tx.T,tx)+ (lambda_*2*np.shape(y)[0])*np.identity(np.shape(tx)[1])),np.dot(tx.T,y))
    # loss
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))

    
    return w,mse 