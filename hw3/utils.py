import numpy as np
import math
import matplotlib.pyplot as plt
import cPickle as pickle
import numpy as np
import os
import softmax
from sklearn import cross_validation
import one_vs_all


def I(bool):
  """
  The indicator function:
  Returns 1 if true, 0 if false
  """
  if bool:
    return 1
  else:
    return 0

def softmax_prob(theta, x, k):
  """
  Non-vectorized function that returns the softmax probability, given a (d+1) x K
  matrix theta, a d+1 dimension vector x, and an integer class k (0 <= k < K)
  """
  # Calculate Theta(j))T * x for all j's
  K = theta.shape[1]
  thetaj_t_x = []
  for j in range(K):
    thetaj_t_x.append(np.dot(theta[:,j],x))

  # Find the max and subtract that from all Theta(j))T * x, then exponentiate
  max_thetaj_t_x = max(thetaj_t_x)
  exp_thetaj_t_x = []
  for j in range(len(thetaj_t_x)):
    thetaj_t_x[j] -= max_thetaj_t_x
    exp_thetaj_t_x.append(math.e ** (thetaj_t_x[j]))

  return exp_thetaj_t_x[k] / (1.0 * sum(exp_thetaj_t_x))

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
            accuracy = np.mean(y_test == predy)
            average_accuracy += accuracy / (1.0 * n_folds)

        print "AVERAGE ACCURACY: ", average_accuracy, "LAMBDA: ", lambda_test

        # Update best lambda/accuracy if necessary
        if average_accuracy >= best_accuracy:
            best_accuracy = average_accuracy
            best_lambda = lambda_test

        # Move to next lambda
        lambda_test *= lambda_scale

    return best_lambda

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=10000):
  """
  Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
  it for the softmax classifier. 
  """
  # Load the raw CIFAR-10 data
  print "Loading raw CIFAR-10 data..."
  cifar10_dir = 'datasets/cifar-10-batches-py'
  X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
  
  # subsample the data

  X_train, y_train, X_val, y_val, X_test, y_test = subsample(num_training,num_validation,num_test,X_train,y_train,X_test,y_test)

  # visualize a subset of the training data
  
  visualize_cifar10(X_train,y_train)

  # preprocess data
  
  X_train, X_val, X_test = preprocess(X_train, X_val, X_test)
  print "Done loading!"
  return X_train, y_train, X_val, y_val, X_test, y_test


def load_CIFAR_batch(filename):
  """ load single batch of cifar """
  with open(filename, 'rb') as f:
    datadict = pickle.load(f)
    X = datadict['data']
    Y = datadict['labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
    Y = np.array(Y)
    return X, Y

def load_CIFAR10(cifar10_root):
  """ load all of cifar """
  xs = []
  ys = []
  for b in range(1,6):
    f = os.path.join(cifar10_root, 'data_batch_%d' % (b, ))
    X, Y = load_CIFAR_batch(f)
    xs.append(X)
    ys.append(Y)    
  Xtr = np.concatenate(xs)
  Ytr = np.concatenate(ys)
  del X, Y
  Xte, Yte = load_CIFAR_batch(os.path.join(cifar10_root, 'test_batch'))
  return Xtr, Ytr, Xte, Yte


# Visualize some examples from the dataset.
# We show a few examples of training images from each class.

def visualize_cifar10(X_train,y_train):
  classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
  num_classes = len(classes)
  samples_per_class = 7
  for y, cls in enumerate(classes):
    idxs = np.flatnonzero(y_train == y)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(X_train[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
  plt.savefig('cifar10_samples.pdf')
  plt.close()

# subsampling  the data

def subsample(num_training,num_validation,num_test,X_train,y_train,X_test,y_test):

  # Our validation set will be num_validation points from the original
  # training set.

  mask = range(num_training, num_training + num_validation)
  X_val = X_train[mask]
  y_val = y_train[mask]

  # Our training set will be the first num_train points from the original
  # training set.
  mask = range(num_training)
  X_train = X_train[mask]
  y_train = y_train[mask]

  # We use the first num_test points of the original test set as our
  # test set.

  mask = range(num_test)
  X_test = X_test[mask]
  y_test = y_test[mask]

  return X_train, y_train, X_val, y_val, X_test, y_test

def preprocess(X_train,X_val,X_test):

  # Preprocessing: reshape the image data into rows

  X_train = np.reshape(X_train, (X_train.shape[0], -1))
  X_val = np.reshape(X_val, (X_val.shape[0], -1))
  X_test = np.reshape(X_test, (X_test.shape[0], -1))

  # As a sanity check, print out the shapes of the data
  print 'Training data shape: ', X_train.shape
  print 'Validation data shape: ', X_val.shape
  print 'Test data shape: ', X_test.shape

  # Preprocessing: subtract the mean image
  # first: compute the image mean based on the training data

  mean_image = np.mean(X_train, axis=0)
#  plt.figure(figsize=(4,4))
#  plt.imshow(mean_image.reshape((32,32,3)).astype('uint8')) # visualize the mean image

  # second: subtract the mean image from train and test data

  X_train -= mean_image
  X_val -= mean_image
  X_test -= mean_image

  # third: append the bias dimension of ones (i.e. bias trick) so that our softmax regressor
  # only has to worry about optimizing a single weight matrix theta.
  # Also, lets transform data matrices so that each image is a row.

  X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
  X_val = np.hstack([np.ones((X_val.shape[0], 1)), X_val])
  X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

  print 'Training data shape with bias term: ', X_train.shape
  print 'Validation data shape with bias term: ', X_val.shape
  print 'Test data shape with bias term: ', X_test.shape

  return X_train, X_val, X_test
  
