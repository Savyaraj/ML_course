import numpy as np
"""Implementations of the methods"""



def compute_gradient(y, tx, w):
    """ computing the gradient and the loss for least square"""
    res = y- np.dot(tx,w);
    loss =  (np.dot(res,res))/(2*np.shape(y)[0])
    
    # compute gradient 
    grad = -1/(np.shape(y)[0])*np.dot((tx.T),(y- np.dot(tx,w)));
    
    return grad,loss


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent algorithm."""
    w = initial_w
    for n_iter in range(max_iters):
        
        # compute gradient 
        grad,_ = compute_gradient(y,tx,w)
        
        # update w by gradient
        w = w-gamma*grad;
        
    _, loss = compute_gradient(y,tx,w)
    return w,loss

def least_squares(y, tx):
    """calculate the least squares solution."""
    # returns mse, and optimal weights
    optimum = np.dot(np.linalg.inv(np.dot(tx.T,tx)),np.dot(tx.T,y))
    # The error
    res = y- np.dot(tx,optimum);
    #Mean squared error
    MSE = (np.dot(res,res))/(2*np.shape(y)[0])
    return optimum,MSE

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
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
    stochgrad = -1/(np.shape(y)[0])*np.dot((tx.T),(y- np.dot(tx,w)));
    return stochgrad

def compute_loss(y, tx, w):
    """Calculate the loss using mse.
    """
    e = y - np.dot(tx, w)
    loss_func = 1 / 2 * np.mean(e ** 2)
    return loss_func

def least_squares_SGD(
        y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm for least square regression ."""
    w = initial_w
    for n_iter in range(max_iters):
        # computing stochastic gradient
        for minibatch_y, minibatch_tx in batch_iter(y, tx, 1):
            grad = compute_stoch_gradient(minibatch_y,minibatch_tx,w)
       #updating the w
        w = w-gamma*grad;  
        
    loss = compute_loss(y,tx,w)    
    return w, loss

def ridge_regression(y, tx, lambda_):
    # The optimum w
    w =  np.dot(np.linalg.inv(np.dot(tx.T,tx)+ (lambda_*2*np.shape(y)[0])*np.identity(np.shape(tx)[1])),np.dot(tx.T,y))
    # loss
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e)) + lambda_*(np.inner(w,w))
    
    return w,mse 

"""Implementations of the helper functions for logistic regression"""

def sigmoid(t):
    """apply the sigmoid function on t."""
    return 1.0 / (1 + np.exp(-t))

def calculate_loss(y, tx, w):
    """compute the loss: negative log likelihood."""
    return np.squeeze(-(np.dot(y.transpose(), np.log(sigmoid(np.dot(tx, w)))) + np.dot((1 - y).transpose(), np.log(1 - sigmoid(np.dot(tx, w))))))

def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    return np.dot(tx.transpose(), sigmoid(np.dot(tx, w)) - y)

def logistic_regression(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    loss = calculate_loss(y, tx, w)
    grad = calculate_gradient(y, tx, w)
    w = w - gamma * grad
    return w, loss
"""Implementations of the regularised logistic regression"""

def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient."""
    # num_samples = y.shape[0]
    loss = calculate_loss(y, tx, w) + lambda_ * np.squeeze(np.dot(w.transpose(), w))
    gradient = calculate_gradient(y, tx, w) + 2 * lambda_ * w
    return loss, gradient

def reg_logistic_regression(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
    w = w - gamma * gradient
    return w, loss

