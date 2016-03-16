from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from reg_linear_regressor_multi import RegularizedLinearReg_SquaredLoss
import plot_utils


#############################################################################
#  Normalize features of data matrix X so that every column has zero        #
#  mean and unit variance                                                   #
#     Input:                                                                #
#     X: N x D where N is the number of rows and D is the number of         #
#        features                                                           #
#     Output: mu: D x 1 (mean of X)                                         #
#          sigma: D x 1 (std dev of X)                                      #
#         X_norm: N x D (normalized X)                                      #
#############################################################################

def feature_normalize(X):

    ########################################################################
    # TODO: modify the three lines below to return the correct values
    mu = np.zeros((X.shape[1],))
    sigma = np.ones((X.shape[1],))
    X_norm = np.zeros(X.shape)

    mu = np.mean(X, axis = 0)
    sigma = np.std(X, axis = 0)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X_norm[i][j] = (X[i][j] - mu[j]) / sigma[j]
  
    ########################################################################
    return X_norm, mu, sigma


#############################################################################
#  Plot the learning curve for training data (X,y) and validation set       #
# (Xval,yval) and regularization lambda reg.                                #
#     Input:                                                                #
#     X: N x D where N is the number of rows and D is the number of         #
#        features                                                           #
#     y: vector of length N which are values corresponding to X             #
#     Xval: M x D where N is the number of rows and D is the number of      #
#           features                                                        #
#     yval: vector of length N which are values corresponding to Xval       #
#     reg: regularization strength (float)                                  #
#     Output: error_train: vector of length N-1 corresponding to training   #
#                          error on training set                            #
#             error_val: vector of length N-1 corresponding to error on     #
#                        validation set                                     #
#############################################################################

def learning_curve(X,y,Xval,yval,reg):
    num_examples,dim = X.shape
    error_train = np.zeros((num_examples,))
    error_val = np.zeros((num_examples,))
    
    ###########################################################################
    # TODO: compute error_train and error_val                                 #
    # 7 lines of code expected                                                #
    ###########################################################################

    for i in range(num_examples):
        reglinear_reg = RegularizedLinearReg_SquaredLoss()
        theta_opt = reglinear_reg.train(X[:i+1],y[:i+1],reg,num_iters=1000)
        single_error_train = np.dot(X[:i+1], theta_opt) - y[:i+1]
        single_error_val = np.dot(Xval, theta_opt) - yval
        error_train[i] = (np.dot(single_error_train.T, single_error_train)) / (2 * (i + 1))
        error_val[i] = (np.dot(single_error_val.T, single_error_val)) / (2 * Xval.shape[0])

    # print "ERROR TRAIN: ", error_train
    # print "ERROR VAL: ", error_val

    ###########################################################################

    return error_train, error_val

#############################################################################
#  Plot the validation curve for training data (X,y) and validation set     #
# (Xval,yval)                                                               #
#     Input:                                                                #
#     X: N x D where N is the number of rows and D is the number of         #
#        features                                                           #
#     y: vector of length N which are values corresponding to X             #
#     Xval: M x D where N is the number of rows and D is the number of      #
#           features                                                        #
#     yval: vector of length N which are values corresponding to Xval       #
#                                                                           #
#     Output: error_train: vector of length N-1 corresponding to training   #
#                          error on training set                            #
#             error_val: vector of length N-1 corresponding to error on     #
#                        validation set                                     #
#############################################################################

def validation_curve(X,y,Xval,yval):

    # reg_vec = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    reg_vec = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]
    error_train = np.zeros((len(reg_vec),))
    error_val = np.zeros((len(reg_vec),))

    ###########################################################################
    # TODO: compute error_train and error_val                                 #
    # 5 lines of code expected                                                #
    ###########################################################################

    # error_val_min = float('inf')
    # lambda_opt = 0
    # theta_opt_final = np.zeros(X.shape[0])
    for i in range(len(reg_vec)):
        reglinear_reg2 = RegularizedLinearReg_SquaredLoss()
        theta_opt = reglinear_reg2.train(X,y,reg=reg_vec[i], num_iters=1000)
        print "Theta at reg = " + str(reg_vec[i]) + " is: ", theta_opt
        error_train[i] = reglinear_reg2.loss(theta_opt, X, y, 0)
        error_val[i] = reglinear_reg2.loss(theta_opt, Xval, yval, 0)
        # if error_val[i] < error_val_min:
        #     lambda_opt = reg_vec[i]
        #     theta_opt_final = theta_opt

    return reg_vec, error_train, error_val


def validation_curve_normal_equation(X,y,Xval,yval):

    # reg_vec = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    reg_vec = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]
    error_train = np.zeros((len(reg_vec),))
    error_val = np.zeros((len(reg_vec),))

    ###########################################################################
    # TODO: compute error_train and error_val                                 #
    # 5 lines of code expected                                                #
    ###########################################################################

    for i in range(len(reg_vec)):
        reglinear_reg2 = RegularizedLinearReg_SquaredLoss()
        theta_opt = reglinear_reg2.normal_equation(X,y,reg=reg_vec[i])
        print "Theta at reg = " + str(reg_vec[i]) + " is: ", theta_opt
        error_train[i] = np.square(np.dot(X, theta_opt) - y).sum() / (2.0 * y.size)
        error_val[i] = np.square(np.dot(Xval, theta_opt) - yval).sum() / (2.0 * yval.size)
    return reg_vec, error_train, error_val

import random

#############################################################################
#  Plot the averaged learning curve for training data (X,y) and             #
#  validation set  (Xval,yval) and regularization lambda reg.               #
#     Input:                                                                #
#     X: N x D where N is the number of rows and D is the number of         #
#        features                                                           #
#     y: vector of length N which are values corresponding to X             #
#     Xval: M x D where N is the number of rows and D is the number of      #
#           features                                                        #
#     yval: vector of length N which are values corresponding to Xval       #
#     reg: regularization strength (float)                                  #
#     Output: error_train: vector of length N-1 corresponding to training   #
#                          error on training set                            #
#             error_val: vector of length N-1 corresponding to error on     #
#                        validation set                                     #
#############################################################################

def averaged_learning_curve(X,y,Xval,yval,reg):
    num_examples,dim = X.shape
    error_train = np.zeros((num_examples,))
    error_val = np.zeros((num_examples,))

    ###########################################################################
    # TODO: compute error_train and error_val                                 #
    # 10-12 lines of code expected                                            #
    ###########################################################################

    num_iters = 50
    for sample_size in xrange(1, num_examples+1):
        for num_iter in xrange(num_iters):
            # random_nums = [random.randint(0, num_examples-1) for j in range(sample_size)]
            random_nums = random.sample(range(num_examples), sample_size)
            Xrand = X[random_nums,:]
            yrand = y[random_nums]

            reglinear_reg = RegularizedLinearReg_SquaredLoss()
            theta_opt = reglinear_reg.train(Xrand,yrand,reg,num_iters=1000)

            # single_error_train = np.dot(Xrand, theta_opt) - yrand
            # single_error_val = np.dot(Xval, theta_opt) - yval
            # error_train[sample_size - 1] += ((np.dot(single_error_train.T, single_error_train)) / (2. * sample_size))
            # error_val[sample_size - 1] += ((np.dot(single_error_val.T, single_error_val)) / (2. * Xval.shape[0]))

            new_error_train = np.square(np.dot(Xrand, theta_opt) - yrand).sum() / (2.0 * yrand.size)
            new_error_val = np.square(np.dot(Xval, theta_opt) - yval).sum() / (2.0 * yval.size)
            error_train[sample_size - 1] += new_error_train
            error_val[sample_size - 1] += new_error_val

        error_train[sample_size - 1] /= num_iters
        error_val[sample_size - 1] /= num_iters
        # print "SAMPLE SIZE: ", sample_size
        # print "ERROR TRAIN: ", error_train[sample_size - 1]
        # print "ERROR VAL:   ", error_val[sample_size - 1]


    ###########################################################################
    return error_train, error_val


#############################################################################
# Utility functions
#############################################################################
    
def load_mat(fname):
  d = scipy.io.loadmat(fname)
  X = d['X']
  y = d['y']
  Xval = d['Xval']
  yval = d['yval']
  Xtest = d['Xtest']
  ytest = d['ytest']

  # need reshaping!

  X = np.reshape(X,(len(X),))
  y = np.reshape(y,(len(y),))
  Xtest = np.reshape(Xtest,(len(Xtest),))
  ytest = np.reshape(ytest,(len(ytest),))
  Xval = np.reshape(Xval,(len(Xval),))
  yval = np.reshape(yval,(len(yval),))

  return X, y, Xtest, ytest, Xval, yval









