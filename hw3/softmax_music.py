import random
import numpy as np
import matplotlib.pyplot as plt

import music_utils
import utils
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix, classification_report
from softmax import softmax_loss_naive, softmax_loss_vectorized
from softmax import SoftmaxClassifier
import time
from one_vs_all import one_vs_allLogisticRegressor


# some global constants
MUSIC_DIR = "music/"
genres = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]

# TODO: Get the music dataset (CEFS representation) [use code from Hw2]
print "Reading data..."
# select the CEPS or FFT representation
# X,y = music_utils.read_ceps(genres,MUSIC_DIR)
X,y = music_utils.read_fft(genres,MUSIC_DIR)

# TODO: Split into train, validation and test sets
print "Splitting data..."
X_train_val, X_test, y_train_val, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = cross_validation.train_test_split(X, y, test_size=0.1)



# TODO: Use the validation set to tune hyperparameters for softmax classifier
# choose learning rate and regularization strength (use the code from softmax_hw.py)
best_accuracy = 0.0
# learning_rates = [.000001, .00001, .0001, .001, .01]  # [.00001, .0001, .001] is best
learning_rates = [.00001, .0001, .001]
# regularization_strengths = [.01, .1, 1.0, 10.0, 100.0]  # [.1, 1.0, 10.0] is best
regularization_strengths = [.1, 1.0, 10.0]
# batch_sizes = [3, 10, 30]  # [3, 10] is best
batch_sizes = [3, 10]
# num_iters_list = [30000, 100000, 300000]  # [100000] is best
num_iters_list = [100000]

for num_iters in num_iters_list:
  for alpha in learning_rates:
    for batch_size in batch_sizes:
      for reg in regularization_strengths:
        softmax_model = SoftmaxClassifier()
        softmax_model.train(X_train, y_train, learning_rate=alpha, reg=reg, num_iters=num_iters, batch_size=batch_size)
        y_pred = softmax_model.predict(X_val)
        accuracy = np.mean(y_pred == y_val)
        print "NumIters: ", num_iters, " LR: ", alpha, " BatchSize ", batch_size, " Reg ", reg, " Accuracy: ", accuracy
        if accuracy > best_accuracy:
          best_accuracy = accuracy
          best_softmax = softmax_model

print "BEST ACCURACY: ", best_accuracy
print "BEST REG: ", best_softmax.reg
print "BEST ALPHA: ", best_softmax.learning_rate

best_softmax_train_plus_val = SoftmaxClassifier()
best_softmax_train_plus_val.train(
    X_train_val, y_train_val, learning_rate=best_softmax.learning_rate, reg=best_softmax.reg)

# TODO: Evaluate best softmax classifier on set aside test set (use the code from softmax_hw.py)
y_pred = best_softmax_train_plus_val.predict(X_test)
c_matrix = confusion_matrix(y_test,y_pred)
print c_matrix
print "ACCURACY = ", np.trace(c_matrix) / (1.0 * np.sum(c_matrix))
num_songs_actual = np.sum(c_matrix, axis=1)
accuracy_per_class = np.zeros(10)
for i in range(10):
    accuracy_per_class[i] = c_matrix[i,i] / (1.0 * num_songs_actual[i])
print num_songs_actual
print accuracy_per_class

# TODO: Compare performance against OVA classifier of Homework 2 with the same
# train, validation and test sets (use sklearn's classifier evaluation metrics)

# select a regularization parameter
penalty = 'l1'  # 'l1' or 'l2'
reg = 1.0  # anything in the range .1 - 100.0
reg = utils.select_lambda_crossval_log_scale(X_train, y_train, 0.1, 100.0, 10.0, penalty)

print "BEST LAMBDA = ", reg

# create a 1-vs-all classifier
ova_logreg = one_vs_allLogisticRegressor(np.arange(10))

# train the K classifiers in 1-vs-all mode
ova_logreg.train(X_train_val,y_train_val,reg,penalty)

# predict on the set aside test set
ypred = ova_logreg.predict(X_test)
c_matrix = confusion_matrix(y_test,ypred)
print c_matrix
print "ACCURACY = ", np.trace(c_matrix) / (1.0 * np.sum(c_matrix))
num_songs_actual = np.sum(c_matrix, axis=1)
accuracy_per_class = np.zeros(10)
for i in range(10):
    accuracy_per_class[i] = c_matrix[i,i] / (1.0 * num_songs_actual[i])
print num_songs_actual
print accuracy_per_class