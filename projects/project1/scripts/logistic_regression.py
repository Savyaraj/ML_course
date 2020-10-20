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

def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    loss = calculate_loss(y, tx, w)
    grad = calculate_gradient(y, tx, w)
    w = w - gamma * grad
    return loss, w

