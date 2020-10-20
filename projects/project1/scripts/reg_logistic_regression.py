# -*- coding: utf-8 -*-


import numpy as np

def sigmoid(t):
    """apply the sigmoid function on t."""
    return 1.0 / (1 + np.exp(-t))

def calculate_loss(y, tx, w):
    """compute the loss: negative log likelihood."""
    return np.squeeze(-(np.dot(y.transpose(), np.log(sigmoid(np.dot(tx, w)))) + np.dot((1 - y).transpose(), np.log(1 - sigmoid(np.dot(tx, w))))))

def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    return np.dot(tx.transpose(), sigmoid(np.dot(tx, w)) - y)

def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient."""
    num_samples = y.shape[0]
    loss = calculate_loss(y, tx, w) + lambda_ * np.squeeze(np.dot(w.transpose(), w))
    gradient = calculate_gradient(y, tx, w) + 2 * lambda_ * w
    return loss, gradient

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
    w = w - gamma * gradient
    return loss, w