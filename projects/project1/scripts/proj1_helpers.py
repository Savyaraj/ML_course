# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np


def load_csv_data(data_path, sub_sample=False, ratio=0.02, seed=7):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (0,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = 0
    
    # sub-sample
    if sub_sample:
        np.random.seed(seed)
        num_row = len(yb)
        indices = np.random.choice(num_row, int(np.floor(ratio * num_row)))
        yb = yb[indices]
        input_data = input_data[indices]
        ids = ids[indices]
        
    return yb, input_data, ids


def predict_labels(weights, data, is_LR):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    # The threshold should be different for least squares and logistic regression when label is {0,1}.
    # least square: decision boundary t >< 0.5
    # logistic regression:  decision boundary sigmoid(t) >< 0.5  <==> t >< 0
    if is_LR:
        y_pred[np.where(y_pred > 0.0)] = 1
        y_pred[np.where(y_pred <= 0.0)] = 0
    else:
        y_pred[np.where(y_pred > 0.5)] = 1
        y_pred[np.where(y_pred <= 0.5)] = 0
    
    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})

            
def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x, axis = 0)
    x = x - mean_x
    std_x = np.std(x, axis = 0)
    x = x / std_x
    return x, mean_x, std_x


def correction_missing_values(tX):
    """Correction missing values by mean value of feature"""
    tX[tX==-999] = float('nan')
    col_mean_tX = np.nanmean(tX, axis=0)
    indices = np.where(np.isnan(tX))
    tX[indices] = np.take(col_mean_tX, indices[1])
    return tX
    
    
def normalize(tX):
    """Normalization data so that all values are between 0 and 1"""
    max = np.max(tX)
    min = np.min(tX)
    tX = (tX - min) / (max - min)
    return tX


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((x.shape[0], (degree + 1) * x.shape[1]))
    ind = 0
    for feature in range(0, x.shape[1]):
        for deg in range(0, degree+1):
            poly[:, ind] = x[:, feature] ** deg
            ind = ind+1
    return np.array(poly)
                   
                   
def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    num_row = len(y)
    indices = np.random.permutation(num_row)
    index_split = int(np.floor(ratio * num_row))
    index_training = indices[: index_split]
    index_testing = indices[index_split:]
    # create split
    x_training = x[index_training]
    x_testing = x[index_testing]
    y_training = y[index_training]
    y_testing = y[index_testing]
    return x_training, x_testing, y_training, y_testing


def compute_loss(y, tx, w):
    """Calculate the loss using mse.
    """
    e = y - np.dot(tx, w)
    loss_func = 1 / 2 * np.mean(e ** 2)
    return loss_func


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)