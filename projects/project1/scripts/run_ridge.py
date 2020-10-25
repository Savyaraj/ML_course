# -*- coding: utf-8 -*-

import csv
import numpy as np
import matplotlib.pyplot as plt

from implementations import *


#helper functions
def load_csv_data(data_path, sub_sample=False, ratio=0.02, seed=7):
    """Loads data """
    """Function arguments: data_path - file path, boolean flag sub_sample - subsample, ratio """
    """Returns yb (class labels), input_data (features) and ids (event ids)"""
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


def predict_labels(weights, data, is_LR=False):
    """Generates class predictions given weights and a test data matrix"""
    y_pred = np.dot(data, weights)
    # The threshold should be different for least squares and logistic regression when label is {0,1}.
    # least square: decision boundary t >< 0.5
    # logistic regression:  decision boundary sigmoid(t) >< 0.5  <==> t >< 0
    # flag is_LR 
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
    """Correction missing values by median of feature"""
    tX[tX==-999] = float('nan')
    col_median_tX = np.nanmedian(tX, axis=0)
    indices = np.where(np.isnan(tX))
    tX[indices] = np.take(col_median_tX, indices[1])
    return tX, col_median_tX

def correction_missing_values_test(tX_test, col_median_tX):
    """Correction missing values in testing using measurements of training set"""
    tX_test[tX_test==-999] = float('nan')
    indices = np.where(np.isnan(tX_test))
    tX_test[indices] = np.take(col_median_tX, indices[1])
    return tX_test


def remove_outliers(tX):
    """Removing outliers using interquartile range"""
    perc_25 = np.percentile(tX, 25)
    perc_75 = np.percentile(tX, 75)
    distance = perc_75 - perc_25
    tX[tX < perc_25 - 1.5 * distance] = float('nan')
    tX[tX > perc_75 + 1.5 * distance] = float('nan')
    col_median_tX = np.nanmedian(tX, axis=0)
    indices = np.where(np.isnan(tX))
    tX[indices] = np.take(col_median_tX, indices[1])
    return tX, perc_25, perc_75, col_median_tX


def remove_outliers_test(tX_test, perc_25, perc_75, col_median_tX):
    """Removing outliers in testing using measurements of training set"""
    distance = perc_75 - perc_25
    tX_test[tX_test < perc_25 - 1.5 * distance] = float('nan')
    tX_test[tX_test > perc_75 + 1.5 * distance] = float('nan')
    indices = np.where(np.isnan(tX_test))
    tX_test[indices] = np.take(col_median_tX, indices[1])
    return tX_test
    
    
def normalize(tX):
    """Normalization data so that all values are between 0 and 1"""
    max_ = np.max(tX)
    min_ = np.min(tX)
    tX = (tX - min_) / (max_ - min_)
    return tX, max_, min_


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((x.shape[0], degree  * x.shape[1]))
    ind = 0
    for feature in range(0, x.shape[1]):
        for deg in range(1, degree+1):
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


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation_ridge_regression(y, x, k_indices, k, lambda_, degree):
    # get k'th subgroup in test, others in train: TODO
    te_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]

    tr_indice = tr_indice.reshape(-1)

    y_testing = y[te_indice]
    y_training = y[tr_indice]
    tx_testing = tX[te_indice]
    tx_training = tX[tr_indice]

    tx_training = np.delete(tx_training, (4,5,6,12,26,27,28), axis=1)
    tx_training, col_median_tX = correction_missing_values(tx_training)
    tx_training, perc_25, perc_75, col_median_tX = remove_outliers(tx_training)
    tx_training, max_, min_ = normalize(tx_training)
    tx_training, mean, std = standardize(tx_training)
    
    
    tx_testing = np.delete(tx_testing, (4,5,6,12,26,27,28), axis=1)
    tx_testing = correction_missing_values_test(tx_testing, col_median_tX)
    tx_testing = remove_outliers_test(tx_testing, perc_25, perc_75, col_median_tX)
    tx_testing = (tx_testing - min_) / (max_ - min_)
    tx_testing = (tx_testing - mean) / std
    
    tx_training_ = build_poly(tx_training, degree)
    tx_training_ = np.c_[np.ones((y_training.shape[0], 1)), tx_training_]
    tx_testing_ = build_poly(tx_testing, degree)
    tx_testing_ = np.c_[np.ones((y_testing.shape[0], 1)), tx_testing_]

    w_ridge, _ = ridge_regression(y_training, tx_training_, lambda_)

    loss_tr = np.sqrt(2 * compute_mse(y_training, tx_training_, w_ridge))
    loss_te = np.sqrt(2 * compute_mse(y_testing, tx_testing_, w_ridge))

    return loss_tr, loss_te, w_ridge


def cross_validation_visualization(parameter, x_label, mse_tr, mse_te):
    """visualization the curves of mse_tr and mse_te."""
    plt.semilogx(parameter, mse_tr, marker=".", color='b', label='train error')
    plt.semilogx(parameter, mse_te, marker=".", color='r', label='test error')
    plt.xlabel(x_label)
    plt.ylabel("rmse")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig(x_label)
    
    
def best_degree_selection_ridge_regression(degrees, lambdas, k_fold, seed = 1):
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    
    #for each degree, we compute the best lambdas and the associated rmse
    best_lambdas = []
    best_rmses = []
    #vary degree
    for degree in degrees:
        # cross validation
        rmse_te = []
        for lambda_ in lambdas:
            rmse_te_tmp = []
            for k in range(k_fold):
                _, loss_te, _ = cross_validation_ridge_regression(y, tX, k_indices, k, lambda_, degree)
                rmse_te_tmp.append(loss_te)
            rmse_te.append(np.mean(rmse_te_tmp))
        
        ind_lambda_opt = np.argmin(rmse_te)
        best_lambdas.append(lambdas[ind_lambda_opt])
        best_rmses.append(rmse_te[ind_lambda_opt])
        
    ind_best_degree =  np.argmin(best_rmses)
        
    return degrees[ind_best_degree], best_rmses[ind_best_degree], best_lambdas[ind_best_degree]
    
    
def best_lambda_selection_ridge_regression(degree, lambdas, k_fold, seed = 1):
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    # cross validation: TODO
    for lambda_ in lambdas:
        rmse_tr_tmp = []
        rmse_te_tmp = []
        for k in range(k_fold):
            loss_tr, loss_te, _ = cross_validation_ridge_regression(y, tX, k_indices, k, lambda_, degree)
            rmse_tr_tmp.append(loss_tr)
            rmse_te_tmp.append(loss_te)
        rmse_tr.append(np.mean(rmse_tr_tmp))
        rmse_te.append(np.mean(rmse_te_tmp))

    x_label = 'lambda'
    cross_validation_visualization(lambdas, x_label, rmse_tr, rmse_te)
    ind_best_lambda = np.argmin(rmse_te)
    return lambdas[ind_best_lambda], rmse_te[ind_best_lambda]


#******************************************************************************************


#download train data
DATA_TRAIN_PATH = '../data/train.csv' 
y, tX, ids = load_csv_data(DATA_TRAIN_PATH, sub_sample=True)

#split the dataset on training (80% of dataset) and testing (20% of dataset) sets 
ratio = 0.8
seed = 6
tx_training, tx_testing, y_training, y_testing = split_data(tX, y, ratio, seed)


#preprocessing training set
#removing features with a large (more than 70 percent) amount of missing values
tx_training = np.delete(tx_training, (4,5,6,12,26,27,28), axis=1)
#replacing the missing values of other features with the median of the corresponding feature
tx_training, col_median_tX = correction_missing_values(tx_training)
#replacing feature outliers with the median of the corresponding feature
tx_training, perc_25, perc_75, col_median_tX = remove_outliers(tx_training)
#normalization
tx_training, max_, min_ = normalize(tx_training)
#standardization
tx_training, mean, std = standardize(tx_training)

#preprocessing testing set (the same actions as for the training set but using measurements obtained from the training set)
tx_testing = np.delete(tx_testing, (4,5,6,12,26,27,28), axis=1)
tx_testing = correction_missing_values_test(tx_testing, col_median_tX)
tx_testing = remove_outliers_test(tx_testing, perc_25, perc_75, col_median_tX)
tx_testing = (tx_testing - min_) / (max_ - min_)
tx_testing = (tx_testing - mean) / std


#cross-validation for selection the best degree of a polynomial in ridge regression
#degree,_,_ = best_degree_selection_ridge_regression(np.arange(1,11), np.logspace(-10, 0, 30), 4)
#print("Best polynomial degree for ridge regression = {deg}".format(deg=degree))

#lambda_, _ = best_lambda_selection_ridge_regression(degree, np.logspace(-10, 0, 50), 4)
#print("Best lambda for ridge regression = {lam}".format(lam=lambda_))


degree = 4
lambda_ = 1.5998587196060573e-10
#polynomial expansion of the training set
tx_training = build_poly(tx_training, degree)
#adding an offset to the training set
tx_training = np.c_[np.ones((y_training.shape[0], 1)), tx_training]

#polynomial expansion of the testing set
tx_testing = build_poly(tx_testing, degree)
#adding an offset to the testing set
tx_testing = np.c_[np.ones((y_testing.shape[0], 1)), tx_testing]

#Ridge regression
w_ridge, loss = ridge_regression(y_training, tx_training, lambda_)
print("weights={w}, loss={l}".format(w=w_ridge, l=loss))
print("training loss={l_tr}".format(l_tr=compute_mse(y_training, tx_training, w_ridge)))
print("testing loss={l_te}".format(l_te=compute_mse(y_testing, tx_testing, w_ridge)))
print("After ridge regression: "+str(round(100*np.sum(predict_labels(w_ridge, tx_testing, False)==y_testing)/len(y_testing),5))+" %\n")


# download test data
DATA_TEST_PATH = '../data/test.csv'
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

tX_test = np.delete(tX_test, (4,5,6,12,26,27,28), axis=1)
tX_test = correction_missing_values_test(tX_test, col_median_tX)
tX_test = remove_outliers_test(tX_test, perc_25, perc_75, col_median_tX)
tX_test = (tX_test - min_) / (max_ - min_)
tX_test = (tX_test - mean) / std
tX_test = build_poly(tX_test, degree)
tX_test = np.c_[np.ones((tX_test.shape[0], 1)), tX_test]

OUTPUT_PATH = 'test_ridge_regr_deg4.csv'
y_pred = predict_labels(w_ridge, tX_test, False)
y_pred[np.where(y_pred == 0)] = -1
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
