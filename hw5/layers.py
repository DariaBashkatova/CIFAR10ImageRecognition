import numpy as np
import fast_layers

def affine_forward(x, theta, theta_0):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (m, d_1, ..., d_k) and contains a minibatch of m
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension d = d_1 * ... * d_k, and
  then transform it to an output vector of dimension h.

  Inputs:
  - x: A numpy array containing input data, of shape (m, d_1, ..., d_k)
  - theta: A numpy array of weights, of shape (d, h)
  - theta_0: A numpy array of biases, of shape (h,)
  
  Returns a tuple of:
  - out: output, of shape (m, h)
  - cache: (x, theta, theta_0)
  """
  out = None
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  # 2 lines of code expected

  x_orig_shape = x.shape
  x = x.reshape(x.shape[0], theta.shape[0])
  out = x.dot(theta) + theta_0
  x = x.reshape(x_orig_shape)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, theta, theta_0)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (m, h)
  - cache: Tuple of:
    - x: Input data, of shape (m, d_1, ... d_k)
    - theta: Weights, of shape (d, h)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (m, d_1, ..., d_k)
  - dtheta: Gradient with respect to theta, of shape (d, h)
  - dtheta_0: Gradient with respect to b, of shape (h,)
  """
  x, theta, theta_0 = cache
  dx, dtheta, dtheta_0 = None, None, None
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  # Hint: do not forget to reshape x into (m,d) form
  # 4-5 lines of code expected

  dx = dout.dot(theta.T).reshape(x.shape)
  x = x.reshape(x.shape[0], theta.shape[0])
  dtheta = x.T.dot(dout)
  dtheta_0 = dout.T.dot(np.ones(x.shape[0]))

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dtheta, dtheta_0


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  # 1-2 lines of code expected.

  out = np.maximum(np.zeros(x.shape), x)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  # 1-2 lines of code expected. Hint: use np.where

  dx = np.where(x > 0, dout, np.zeros(dout.shape))

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx




def conv_forward_naive(x, theta, theta0, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of m data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width WW.

  Input:
  - x: Input data of shape (m, C, H, W)
  - theta: Filter weights of shape (K, C, HH, WW)
  - theta0: Biases, of shape (K,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (m, K, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, theta, theta0, conv_param)
  """

  out = None
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  # 14 lines of code expected
  m = x.shape[0]
  K = theta.shape[0]
  C = x.shape[1]
  P = conv_param['pad']
  S = conv_param['stride']
  HH = theta.shape[2]
  WW = theta.shape[3]
  H_prime = 1 + (x.shape[2] + 2 * P - HH) / S
  W_prime = 1 + (x.shape[3] + 2 * P - WW) / S
  out = np.zeros([m, K, H_prime, W_prime])
  for example_index in xrange(m):
    example = x[example_index]
    for filter_index in xrange(K):
      # print "filter", filter_index
      filter = theta[filter_index]
      out[example_index][filter_index] += theta0[filter_index]
      for channel in xrange(C):
        # print "channel", channel
        example_channel = np.lib.pad(example[channel], ((P, P), (P, P)), 'constant')
        filter_channel = filter[channel].reshape(WW * HH)
        for i in xrange(H_prime):
          for j in xrange(W_prime):
            # print "i, j", i, j
            window = example_channel[i * S: i * S + HH, j * S : j * S + WW]
            window_vec = window.reshape(HH * WW)
            out[example_index][filter_index][i][j] += filter_channel.dot(window_vec)
  pass

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, theta, theta0, conv_param)
  # print out
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives. (m, K, H', W')
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x (m, C, H, W)
  - dtheta: Gradient with respect to theta (K, C, HH, WW)
  - dtheta0: Gradient with respect to theta0 (K)
  """
  dx, dtheta, dtheta0 = None, None, None
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  # 20-22 lines of code expected
  x = cache[0]
  theta = cache[1]
  m = dout.shape[0]
  K = dout.shape[1]
  H_prime = dout.shape[2]
  W_prime = dout.shape[3]
  C = x.shape[1]
  H = x.shape[2]
  W = x.shape[3]
  HH = theta.shape[2]
  WW = theta.shape[3]
  S = cache[3]['stride']
  P = cache[3]['pad']
  dxtemp = np.zeros([m, C, H+2*P, W+2*P])
  dtheta0 = np.zeros(K)
  dtheta = np.zeros([K, C, HH, WW])
  for example_index in xrange(m):
    example = x[example_index]
    for filter_index in xrange(K):
      for channel in xrange(C):
        example_channel = np.lib.pad(example[channel], ((P, P), (P, P)), 'constant')
        for i in xrange(H_prime):
          for j in xrange(W_prime):
            if channel == 0:
              dtheta0[filter_index] += dout[example_index][filter_index][i][j]
            for a in xrange(HH):
              for b in xrange(WW):
                dtheta[filter_index][channel][a][b] += example_channel[i * S + a][j * S + b] * dout[example_index][filter_index][i][j]
                dxtemp[example_index][channel][i * S + a][j * S + b] += theta[filter_index][channel][a][b] * dout[example_index][filter_index][i][j]
  dx = dxtemp[:, :, P:H+P, P:W+P]




  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dtheta, dtheta0


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (m, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  # 12-13 lines of code expected
  m = x.shape[0]
  C = x.shape[1]
  H = x.shape[2]
  W = x.shape[3]
  Fh = pool_param['pool_height']
  Fw = pool_param['pool_width']
  S = pool_param['stride']
  H2 = 1 + (H - Fh)/S
  W2 = 1 + (W - Fw)/S
  out = np.zeros([m, C, H2, W2])
  for example_index in xrange(m):
    for channel_index in xrange(C):
      for i in xrange(H2):
        for j in xrange(W2):
          out[example_index][channel_index][i][j] = np.max(x[example_index][channel_index][i*S:i*S + Fh, j*S:j*S + Fw])
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  # 15 lines of code expected
  x = cache[0]
  pool_param = cache[1]
  m = x.shape[0]
  C = x.shape[1]
  H = x.shape[2]
  W = x.shape[3]
  Fh = pool_param['pool_height']
  Fw = pool_param['pool_width']
  S = pool_param['stride']
  H2 = 1 + (H - Fh)/S
  W2 = 1 + (W - Fw)/S
  dx = np.zeros([m, C, H, W])
  for example_index in xrange(m):
    for channel_index in xrange(C):
      for i in xrange(H2):
        for j in xrange(W2):
          (maxH2, maxW2) = np.unravel_index(x[example_index][channel_index][i*S:i*S + Fh, j*S:j*S + Fw].argmax(), x[example_index][channel_index][i*S:i*S + Fh, j*S:j*S + Fw].shape)
          maxH = maxH2 + i*S
          maxW = maxW2 + j*S
          dx[example_index][channel_index][maxH][maxW] += dout[example_index][channel_index][i][j]

  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx



def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (m, C) where x[i, j] is the output for the jth class
    for the ith input.
  - y: Vector of labels, of shape (m,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  m = x.shape[0]
  correct_class_output = x[np.arange(m), y]
  margins = np.maximum(0, x - correct_class_output[:, np.newaxis] + 1.0)
  margins[np.arange(m), y] = 0
  loss = np.sum(margins) / m
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(m), y] -= num_pos
  dx /= m
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (m, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (m,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  m = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(m), y])) / m
  dx = probs.copy()
  dx[np.arange(m), y] -= 1
  dx /= m
  return loss, dx



