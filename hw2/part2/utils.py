import math
import numpy as np
from sklearn import cross_validation
from sklearn import linear_model
import one_vs_all

def calc_accuracy(y, predy):
    """
    Given binary vectors y (actual values) and predy (predicted values),
    Returns the accuracy of the prediction
    """
    return np.mean(y==predy)

######################################################################################
#   The sigmoid function                                                             #
#     Input: z: can be a scalar, vector or a matrix                                  #
#     Output: sigz: sigmoid of scalar, vector or a matrix                            #
#     TODO: 1 line of code expected                                                  #
######################################################################################

def sigmoid (z):
    sig = np.zeros(z.shape)
    # Your code here
    # 1 line expected
    sig = 1 / (1 + math.e ** -z)
    # End your ode

    return sig


def select_lambda_crossval_log_scale(X,y,lambda_low,lambda_high,lambda_scale,penalty):

    best_lambda = lambda_low
    n_folds = 10
    best_accuracy = 0.0
    lambda_test = lambda_low

    # Test each lambda in the inputted range
    while lambda_test <= lambda_high:
        kf = cross_validation.KFold(X.shape[0], n_folds=n_folds)
        average_accuracy = 0.0
        for train_index, test_index in kf:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Train appropriate model (L1 or L2) on these folds and calculate error
            ova_logreg = one_vs_all.one_vs_allLogisticRegressor(np.arange(10))
            ova_logreg.train(X_train,y_train,lambda_test,penalty)
            predy = ova_logreg.predict(X_test)
            accuracy = calc_accuracy(y_test, predy)
            average_accuracy += accuracy / (1.0 * n_folds)

        print "AVERAGE ACCURACY: ", average_accuracy, "LAMBDA: ", lambda_test

        # Update best lambda/accuracy if necessary
        if average_accuracy >= best_accuracy:
            best_accuracy = average_accuracy
            best_lambda = lambda_test

        # Move to next lambda
        lambda_test *= lambda_scale

    return best_lambda
