# -*- coding: utf-8 -*-
"""Gradient Descent"""

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    # compute loss
    res = y- np.dot(tx,w);
    loss =  (np.dot(res,res))/(2*np.shape(y)[0])
    
    # compute gradient 
    grad = -1/(np.shape(y)[0])*np.dot((tx.T),(y- np.dot(tx,w)));
    
    return grad,loss


def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    w = initial_w
    for n_iter in range(max_iters):
        
        # compute gradient and loss
        grad,loss = compute_gradient(y,tx,w)
        
        # update w by gradient
        w = w-gamma*grad;
    return w,loss