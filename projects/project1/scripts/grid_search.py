import numpy as np
from reg_logistic_regression import *

def generate_lambda(num_intervals):
    """Generate a grid of values for lambda."""
    lambdas = np.linspace(0, 2, num_intervals)
    return lambdas

def grid_search(y, y_test, tx, tx_testing, w, lambdas, gamma, n_iters):
    """Algorithm for grid search, returns optimum gamma and corresponding loss"""
    losses = np.zeros((len(lambdas)))
    for i in range(len(lambdas)):
        for j in range(n_iters):          
            w, trainingloss = learning_by_penalized_gradient(y, tx, w, gamma, lambdas[i])
            loss = calculate_loss(y_test,tx_testing,w)
        losses[i] = loss
        
    min_row = np.argmin(losses)
    optimum = lambdas[min_row]
    return losses[min_row], optimum