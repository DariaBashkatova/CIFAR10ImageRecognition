import numpy as np
import music_utils
from one_vs_all import one_vs_allLogisticRegressor
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix, classification_report
import utils

# some global constants
MUSIC_DIR = "music/"
genres = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]
penalty = 'l2'

# select the CEPS or FFT representation
X,y = music_utils.read_ceps(genres,MUSIC_DIR)
# X,y = music_utils.read_fft(genres,MUSIC_DIR)


#  divide X into train and test sets
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
print y_train.shape
# select a regularization parameter
reg = utils.select_lambda_crossval_log_scale(X_train, y_train, 0.0001, 10000.0, 10.0, penalty)

print "BEST LAMBDA = ", reg

# create a 1-vs-all classifier
ova_logreg = one_vs_allLogisticRegressor(np.arange(10))

# train the K classifiers in 1-vs-all mode
ova_logreg.train(X_train,y_train,reg,penalty)

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
