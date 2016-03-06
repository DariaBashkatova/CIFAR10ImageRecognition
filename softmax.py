import math
import numpy as np
import random
from random import shuffle
import scipy.sparse
import scipy.optimize

import utils


class SoftmaxClassifier:

  def __init__(self):
    self.theta = None
    self.learning_rate = None
    self.reg = None
    self.num_iters = None
    self.batch_size = None

  def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train the classifier using mini-batch stochastic gradient descent.

    Inputs:
    - X: m x d array of training data. Each training point is a d-dimensional
         row.
    - y: 1-dimensional array of length m with labels 0...K-1, for K classes.
    - learning_rate: (float) learning rate for optimization.
    - reg: (float) regularization strength.
    - num_iters: (integer) number of steps to take when optimizing
    - batch_size: (integer) number of training examples to use at each step.
    - verbose: (boolean) If true, print progress during optimization.

    Outputs:
    A list containing the value of the loss function at each training iteration.
    """
    num_train,dim = X.shape
    num_classes = np.max(y) + 1 # assume y takes values 0...K-1 where K is number of classes
    self.theta = np.random.randn(dim,num_classes) * 0.001
    self.learning_rate = learning_rate
    self.reg = reg
    self.num_iters = num_iters
    self.batch_size = batch_size

    # Use fmin_bfgs if desired
    if learning_rate == "fmin_bfgs":
      print "Using fmin_bfgs!"
      print self.theta.shape
      self.theta = scipy.optimize.fmin_bfgs(
          self.J_cost, np.asarray(self.theta).flatten(), fprime=self.grad_loss, args=(X, y, reg, num_classes), maxiter=num_iters).reshape((dim, num_classes))
      return []

    # Run stochastic gradient descent to optimize theta
    loss_history = []
    for it in xrange(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO:                                                                 #
      # Sample batch_size elements from the training data and their           #
      # corresponding labels to use in this round of gradient descent.        #
      # Store the data in X_batch and their corresponding labels in           #
      # y_batch; after sampling X_batch should have shape (batch_size, dim)   #
      # and y_batch should have shape (batch_size,)                           #
      #                                                                       #
      # Hint: Use np.random.choice to generate indices. Sampling with         #
      # replacement is faster than sampling without replacement.              #
      #########################################################################
      # Hint: 3 lines of code expected

      random_indices = random.sample(range(num_train), batch_size)
      X_batch = X[random_indices,:]
      y_batch = y[random_indices]

      #########################################################################
      #                       END OF YOUR CODE                                #
      #########################################################################

      # evaluate loss and gradient
      loss, grad = self.loss(X_batch, y_batch, reg)
      loss_history.append(loss)

      # perform parameter update
      #########################################################################
      # TODO:                                                                 #
      # Update the weights using the gradient and the learning rate.          #
      #########################################################################
      # Hint: 1 line of code expected

      self.theta -= learning_rate * grad
      # print np.average(np.abs(grad))

      #########################################################################
      #                       END OF YOUR CODE                                #
      #########################################################################

      if verbose and it % 100 == 0:
        print 'iteration %d / %d: loss %f' % (it, num_iters, loss)

    return loss_history

  def predict(self, X):
    """
    Use the trained weights of this linear classifier to predict labels for
    data points.

    Inputs:
    - X: m x d array of training data. Each row is a d-dimensional point.

    Returns:
    - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
      array of length m, and each element is an integer giving the predicted
      class.
    """
    y_pred = np.zeros(X.shape[1])
    y_pred = np.argmax(np.dot(X, self.theta), axis=1)
    return y_pred

  def loss(self, X_batch, y_batch, reg):
    """
    Compute the loss function and its derivative.
    Subclasses will override this.

    Inputs:
    - X_batch: m x d array of data; each row is a data point.
    - y_batch: 1-dimensional array of length m with labels 0...K-1, for K classes.
    - reg: (float) regularization strength.

    Returns: A tuple containing:
    - loss as a single float
    - gradient with respect to self.theta; an array of the same shape as theta
    """

    return softmax_loss_vectorized(self.theta, X_batch, y_batch, reg)

  def J_cost(self, *args):
    """
    Helper function for fmin_bfgs.
    Calculates J cost (i.e. loss)
    """
    theta, X, y, reg, k = args
    m, dim = X.shape

    theta = theta.reshape((dim, k))
    matrix_y = convert_y_to_matrix_corrected(y, k)
    X_theta = np.dot(X, theta)
    X_theta_scaled = (X_theta.T - np.max(X_theta, axis=1)).T
    e_X_theta = math.e ** X_theta_scaled
    sums = np.sum(e_X_theta, axis=1)
    prob_matrix = (e_X_theta.T / sums).T
    unreg_J = -np.sum(matrix_y * np.log(prob_matrix)) / (1.0 * m)
    J_regularization = np.sum(theta * theta) * reg / (2.0 * m)
    return unreg_J + J_regularization

  def grad_loss(self, *args):
    """
    Helper function for fmin_bfgs.
    Calculates grad_loss.
    """
    theta, X, y, reg, k = args
    m, dim = X.shape

    theta = theta.reshape((dim, k))
    matrix_y = convert_y_to_matrix_corrected(y, k)
    X_theta = np.dot(X, theta)
    X_theta_scaled = (X_theta.T - np.max(X_theta, axis=1)).T
    e_X_theta = math.e ** X_theta_scaled
    sums = np.sum(e_X_theta, axis=1)
    prob_matrix = (e_X_theta.T / sums).T
    unreg_grad = np.dot(X.T, matrix_y - prob_matrix) / (-1.0 * m)
    grad_regularization = theta * reg / (1.0 * m)
    return np.asarray(unreg_grad + grad_regularization).flatten()


# def softmax_loss_naive(theta, X, y, reg):
#   """
#   Softmax loss function, naive implementation (with loops)
#   Inputs:
#   - theta: d x K parameter matrix. Each column is a coefficient vector for class k
#   - X: m x d array of data. Data are d-dimensional rows.
#   - y: 1-dimensional array of length m with labels 0...K-1, for K classes
#   - reg: (float) regularization strength
#   Returns:
#   a tuple of:
#   - loss as single float
#   - gradient with respect to parameter matrix theta, an array of same size as theta
#   """
#   # Initialize the loss and gradient to zero.
#   J = 0.0
#   grad = np.zeros_like(theta)
#   m, dim = X.shape
#
#   # Calculate non-regularized part of J
#   K = theta.shape[1]
#   J_nonreg = 0.0
#   for i in range(m):
#     J_nonreg += math.log(utils.softmax_prob(theta, X[i], y[i]))
#   J_nonreg *= -1.0 / m
#
#   # Calculate regularization
#   regularization = 0.0
#   for j in range(K):
#     for k in range(K):
#       regularization += theta[j,k] ** 2
#
#   regularization *= (reg / (2.0 * m))
#
#   J = J_nonreg + regularization
#
#
#   # Calculate gradient
#   for k in range(K):
#     for i in range(m):
#       # print grad[:,k].shape
#       # print X[i].shape
#       grad[:,k] += X[i] * (utils.I(y[i] == k) - utils.softmax_prob(theta, X[i], k))
#   grad *= -1.0 / m
#
#   return J, grad
#
#
# def convert_y_to_matrix(y):
#   """
#   convert an array of m elements with values in {0,...,K-1} to a boolean matrix
#   of size m x K where there is a 1 for the value of y in that row.
#   """
#   y = np.array(y)
#   data = np.ones(len(y))
#   indptr = np.arange(len(y)+1)
#   mat = scipy.sparse.csr_matrix((data,y,indptr))
#   return mat.todense()


def convert_y_to_matrix_corrected(y, k):
  """
  Self made convert_y_to_matrix function
  """
  y_matrix = np.zeros((len(y), k))
  for i in range(len(y)):
    y_matrix[i, y[i]] = 1.0
  return y_matrix



def softmax_loss_vectorized(theta, X, y, reg):
  """
  Softmax loss function, vectorized version.
  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  J = 0.0
  grad = np.zeros_like(theta)
  m, dim = X.shape

  # Calculate J cost
  k = theta.shape[1]
  matrix_y = convert_y_to_matrix_corrected(y, k)
  X_theta = np.dot(X, theta)
  X_theta_scaled = (X_theta.T - np.max(X_theta, axis=1)).T
  e_X_theta = math.e ** X_theta_scaled
  sums = np.sum(e_X_theta, axis=1)
  prob_matrix = (e_X_theta.T / sums).T
  unreg_J = -np.sum(matrix_y * np.log(prob_matrix)) / (1.0 * m)
  J_regularization = np.sum(theta * theta) * reg / (2.0 * m)
  J = unreg_J + J_regularization

  # Calculate grad
  unreg_grad = np.dot(X.T, matrix_y - prob_matrix) / (-1.0 * m)
  grad_regularization = theta * reg / (1.0 * m)
  grad = unreg_grad + grad_regularization

  return J, grad
