import numpy as np

from layers import *
from layer_utils import *


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of d, a hidden dimension of h, and perform classification over C classes.
  
  The architecture should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """
  
  def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg
    
    ############################################################################
    # TODO: Initialize the weights and biases of the two-layer net. Weights    #
    # should be initialized from a Gaussian with standard deviation equal to   #
    # weight_scale, and biases should be initialized to zero. All weights and  #
    # biases should be stored in the dictionary self.params, with first layer  #
    # weights and biases using the keys 'theta1' and 'theta1_0' and second     #
    # layer weights and biases using the keys 'theta2' and 'theta2_0'.         #
    ############################################################################
    # 4 lines of code expected
    self.params['theta1_0'] = np.zeros(hidden_dim)
    self.params['theta2_0'] = np.zeros(num_classes)
    self.params['theta1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
    self.params['theta2'] = weight_scale * np.random.randn(hidden_dim, num_classes)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (m, d_1, ..., d_k)
    - y: Array of labels, of shape (m,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - output: Array of shape (m, C) giving classification scores, where
      output[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """  
    output = None
    ############################################################################
    # TODO: Implement the forward pass for the two-layer net, computing the    #
    # output for X and storing them in the output variable.                    #
    ############################################################################
    # Hint: unpack the weight parameters from self.params
    # 3 lines of code expected
    out1, cache1 = affine_relu_forward(X, self.params['theta1'], self.params['theta1_0'])
    fc_cache1, relu_cache1 = cache1
    output, fc_cache2 = affine_forward(out1, self.params['theta2'], self.params['theta2_0'])

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If y is None then we are in test mode so just return output
    if y is None:
      return output
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the two-layer net. Store the loss  #
    # in the loss variable and gradients in the grads dictionary. Compute data #
    # loss using softmax, and make sure that grads[k] holds the gradients for  #
    # self.params[k]. Don't forget to add L2 regularization!                   #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient. affine_backward(dout, cache) #
    ############################################################################
    # 8 lines of code expected
    loss, dout = softmax_loss(output, y)
    dx2, grads['theta2'], grads['theta2_0'] = affine_backward(dout, fc_cache2)
    dx1, grads['theta1'], grads['theta1_0'] = affine_relu_backward(dx2, cache1)

    # Add Regularization
    grads['theta2'] += self.reg * self.params['theta2']
    grads['theta1'] += self.reg * self.params['theta1']
    loss += (self.reg / 2.0) * np.sum(self.params['theta2'] ** 2)
    loss += (self.reg / 2.0) * np.sum(self.params['theta1'] ** 2)


    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. 
  For a network with L layers,  the architecture will be
  
  {affine - [batch norm] - relu } x (L - 1) - affine - softmax
  
  and the {...} block is  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               reg=0.0,
               weight_scale=1e-2, dtype=np.float32):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.

    """

    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

    ############################################################################
    # TODO: Initialize the parameters of the network, storing all values in    #
    # the self.params dictionary. Store weights and biases for the first layer #
    # in theta1 and theta1_0; for the second layer use theta2 and theta2_0,    #
    # etc. Weights should be initialized from a normal distribution with       #
    # standard deviation equal to weight_scale and biases should be            #
    # initialized to zero.                                                     #
    ############################################################################
    # 7 lines of code expected

    self.params['theta1'] = weight_scale * np.random.randn(input_dim, hidden_dims[0])
    self.params['theta1_0'] = np.zeros(hidden_dims[0])

    for i in range(2, len(hidden_dims) + 1):
      self.params['theta' + str(i)] = weight_scale * np.random.randn(hidden_dims[i-2], hidden_dims[i-1])
      self.params['theta' + str(i) + '_0'] = np.zeros(hidden_dims[i-1])

    self.params['theta' + str(len(hidden_dims) + 1)] = weight_scale * np.random.randn(hidden_dims[-1], num_classes)
    self.params['theta' + str(len(hidden_dims) + 1) + '_0'] = np.zeros(num_classes)

    i=1
    if self.params['theta' + str(i)] is None:
      print "THETA1 is NONE in INIT"
    if self.params['theta' + str(i) + "_0"] is None:
      print "THETA1_0 is NONE in INIT"
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    output = None
    ############################################################################
    # TODO: Implement the forward pass for the fully-connected net, computing  #
    # the output for X and storing them in the output variable.                #
    #                                                                          #
    ############################################################################
    # 6 lines of code expected.

    # out1, cache1 = affine_relu_forward(X, self.params['theta1'], self.params['theta1_0'])
    # fc_cache1, relu_cache1 = cache1
    # output, fc_cache2 = affine_forward(out1, self.params['theta2'], self.params['theta2_0'])

    prev_layer = X.copy()
    caches = {}
    i=1
    if self.params['theta' + str(i)] is None:
      print "THETA1 is NONE in LOSS"
    if self.params['theta' + str(i) + "_0"] is None:
      print "THETA1_0 is NONE in LOSS"

    # print self.params['theta' + str(i) + '_0']
    for i in range(1, self.num_layers):
      prev_layer, cache = affine_relu_forward(prev_layer, self.params['theta' + str(i)], self.params['theta' + str(i) + '_0'])
      caches[i] = cache
    output, caches[self.num_layers] = affine_forward(prev_layer, self.params['theta' + str(self.num_layers)], self.params['theta' + str(self.num_layers) + '_0'])

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If test mode return early
    if mode == 'test':
      return output

    loss, grads = 0.0, {}
    ############################################################################
    # TODO: Implement the backward pass for the fully-connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    # 10-12 lines of code expected]

    # loss, dout = softmax_loss(output, y)
    # dx2, grads['theta2'], grads['theta2_0'] = affine_backward(dout, fc_cache2)
    # dx1, grads['theta1'], grads['theta1_0'] = affine_relu_backward(dx2, cache1)
    #
    # # Add Regularization
    # grads['theta2'] += self.reg * self.params['theta2']
    # grads['theta1'] += self.reg * self.params['theta1']
    # loss += (self.reg / 2.0) * np.sum(self.params['theta2'] ** 2)
    # loss += (self.reg / 2.0) * np.sum(self.params['theta1'] ** 2)


    loss, dout = softmax_loss(output, y)
    dx = {}
    dx[self.num_layers], grads['theta' + str(self.num_layers)], grads['theta' + str(self.num_layers) + '_0'] = affine_backward(dout, caches[self.num_layers])
    for i in range(self.num_layers-1, 0, -1):
      dx[i], grads['theta' + str(i)], grads['theta' + str(i) + '_0'] = affine_relu_backward(dx[i+1], caches[i])

    for i in range(1, self.num_layers + 1):
      grads['theta' + str(i)] += self.reg * self.params['theta' + str(i)]
      loss += (self.reg / 2.0) * np.sum(self.params['theta' + str(i)] ** 2)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads
