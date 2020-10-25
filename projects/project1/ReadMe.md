The file contains 2 python scripts and a Report describing the pipeline that we followed for prediction on the test data-set. 

The file implementations.py contains the implementation of the 6 functions and the corresponding helper functions for them. The 6 functions are :
1) least_squares_GD(y,tx,initial_w,max_iters,gamma)
2) least_squares_GSD(y,tx,initial_w,max_iters,gamma)
3) least_squares(y,tx)
4) ridge_regression(y,tx,lambda_)
5) logistic_regression(y,tx,initial_w,max_iters,gamma)
6) reg_logistic_regression(y,tx,lambda_,initial_w,max_iters,gamma)

where:

y = the column vector of labels,<br />
tx = matrix of features,<br />
max_iters = maximum number of iterations,<br />
gamma = step size, <br />
lambda_ = regularisation parameter,<br />

Each of the above functions outputs the loss and the best weight vector it found by the iteration.

The main file run.py when run through the command line outputs the best set of weight vectors evaluated and the corresponding accuracy on the training set. The script also returns the "csv" file which contains the predictions on the test data-set. It contains implementations for the cross-validations, splitting the dataset and also training any of the 6 models that were implemented in implementations.py.
