# -*- coding: utf-8 -*-

import numpy as np

def compute_mse(y, tx, w):
    """Calculate the loss using mse."""
    e = y - np.dot(tx, w)
    loss = e.dot(e) / (2 * len(e))
    return loss


def compute_gradient(y, tx, w):
    """Compute the gradient and the loss."""
    # compute loss
    loss = compute_mse(y, tx, w)

    e = y - np.dot(tx, w)
    # compute gradient
    grad = -1/(np.shape(y)[0])*np.dot((tx.T),e)
    
    return grad, loss


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    w = initial_w
    for n_iter in range(max_iters):
        # compute gradient and loss
        grad, loss = compute_gradient(y, tx, w)
        
        # update w by gradient
        w = w - gamma * grad;
        
    return w, loss


#******************************************************************************************



def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    e = y - np.dot(tx,w)
    stoch_grad = -1/(np.shape(y)[0])*np.dot((tx.T),e)
    return stoch_grad


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    w = initial_w
    for n_iter in range(max_iters):
        # computing stochastic gradient
        for minibatch_y, minibatch_tx in batch_iter(y, tx, 1):
            grad = compute_stoch_gradient(minibatch_y,minibatch_tx,w)
        
        #computing the loss
        loss = compute_mse(y, tx, w)
        #updating the w
        w = w - gamma * grad
    return w, loss
                  
                  
# ****************************************************************************************                


def least_squares(y, tx):
    """calculate the least squares solution."""
    # returns mse and optimal weights
    w = np.dot(np.linalg.inv(np.dot(tx.T,tx)),np.dot(tx.T,y))

    #Mean squared error
    loss = compute_mse(y, tx, w)
    # ***************************************************
    return w, loss


# ****************************************************************************************


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    # The optimum w
    w =  np.dot(np.linalg.inv(np.dot(tx.T,tx) + (lambda_*2*np.shape(y)[0])*np.identity(np.shape(tx)[1])),np.dot(tx.T,y))
    # loss
    loss = compute_mse(y, tx, w) + lambda_*(np.inner(w,w))
    return w, loss


# ****************************************************************************************

def sigmoid(t):
    """apply the sigmoid function on t."""
    return 1.0 / (1 + np.exp(-t))

def calculate_loss(y, tx, w):
    """compute the loss: negative log likelihood."""
    sigm = sigmoid(np.dot(tx, w))
    return np.squeeze(-(np.dot(y.transpose(), np.log(sigm)) + np.dot((1 - y).transpose(), np.log(1 - sigm))))

def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    sigm = sigmoid(np.dot(tx, w))
    return np.dot(tx.transpose(), sigm - y)

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    
    for n_iter in range(max_iters):
        loss = calculate_loss(y, tx, w)
        grad = calculate_gradient(y, tx, w)
        w = w - gamma * grad
        
    return w, loss


# ****************************************************************************************


def calculate_gradient_reg_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient."""
    loss = calculate_loss(y, tx, w) + lambda_ * np.squeeze(np.dot(w.transpose(), w))
    grad = calculate_gradient(y, tx, w) + 2 * lambda_ * w
    return loss, grad

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    w = initial_w
    for n_iter in range(max_iters):
        loss, grad = calculate_gradient_reg_logistic_regression(y, tx, w, lambda_)
        w = w - gamma * grad
        
    return w, loss