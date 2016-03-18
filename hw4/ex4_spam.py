from sklearn import cross_validation, preprocessing, metrics
import utils
import scipy.io
import numpy as np
from linear_classifier import LinearSVM_twoclass


#############################################################################
# load the SPAM email training and test dataset                             #
#############################################################################

X,y = utils.load_mat('data/spamTrain.mat')
yy = np.ones(y.shape)
yy[y==0] = -1


test_data = scipy.io.loadmat('data/spamTest.mat')
X_test = test_data['Xtest']
y_test = test_data['ytest'].flatten()

#############################################################################
# your code for setting up the best SVM classifier for this dataset         #
# Design the training parameters for the SVM.                               #
# What should the learning_rate be? What should C be?                       #
# What should num_iters be? Should X be scaled? Should X be kernelized?     #
#############################################################################
# your experiments below

# C = 1
svm = LinearSVM_twoclass()
# svm.theta = np.zeros((X.shape[1],))
# svm.train(X,yy,learning_rate=1e-2,C=C,num_iters=2000,verbose=True)

print "Selecting Hyperparameters..."
X_train, X_val, y_train, y_val = cross_validation.train_test_split(X, yy, test_size=0.2)
poly = preprocessing.PolynomialFeatures(1)

# Cvals = [0.01,0.03,0.1,0.3,1,3,10,30]
# sigma_vals = [0.01,0.03,0.1,0.3,1,3,10,30]
Cvals = [0.1,1,10]
sigma_vals = [0.1,1,10]
learning_rates = [1e-4]
num_iters_list = [10000]

best_accuracy = -1.0
best_C = 1
best_sigma = .1
best_learning_rate = 1e-4
best_num_iters = 1000
kernel = None
# kernel = utils.gaussian_kernel
# kernel = utils.polynomial_kernel
print "KERNEL: ", kernel

for C in Cvals:
	for sigma in sigma_vals:
		for learning_rate in learning_rates:
			for num_iters in num_iters_list:
				if kernel != None:
					# Preprocess train data (Kernelize, scale, and add intercept)
					print "."
					K = np.array([kernel(x1,x2,sigma) for x1 in X_train for x2 in X_train]).reshape(X_train.shape[0],X_train.shape[0])
					scaler = preprocessing.StandardScaler().fit(K)
					scale_K = scaler.transform(K)
					KK = poly.fit_transform(scale_K)

					# Preprocess val data (Kernelize, scale, and add intercept)
					print "."
					K_val = np.array([kernel(x1,x2,sigma) for x1 in X_val for x2 in X_train]).reshape(X_val.shape[0],X_train.shape[0])
					scale_K_val = scaler.transform(K_val)
					KK_val = poly.fit_transform(scale_K_val)

					# Train model and get val accuracy
					print "."
					svm = LinearSVM_twoclass()
					svm.theta = np.zeros((KK.shape[1],))
					svm.train(KK,y_train,learning_rate=learning_rate,C=C,num_iters=num_iters)
					y_val_pred = svm.predict(KK_val)
					accuracy = np.mean(y_val == y_val_pred)

				else:
					XX_train = poly.fit_transform(X_train)
					XX_val = poly.fit_transform(X_val)
					svm.train(XX_train,y_train,learning_rate=learning_rate,C=C,num_iters=num_iters)
					y_val_pred = svm.predict(XX_val)
					accuracy = np.mean(y_val == y_val_pred)

				print "LR:", learning_rate, " NumIters:", num_iters, " C:", C, " Sigma:", sigma, " Accuracy:", accuracy
				if accuracy > best_accuracy:
					best_accuracy = accuracy
					best_C = C
					best_sigma = sigma
					best_learning_rate = learning_rate
					best_num_iters = num_iters

print "Best LR:", learning_rate, " Best NumIters:", num_iters, " Best C:", best_C, " Best Sigma:", best_sigma, " Best Accuracy:", best_accuracy

#############################################################################
#  end of your code                                                         #
#############################################################################

#############################################################################
# what is the accuracy of the best model on the training data itself?       #
#############################################################################
# 2 lines of code expected

y_pred = svm.predict(X)
print "Accuracy of model on training data is: ", metrics.accuracy_score(yy,y_pred)


#############################################################################
# what is the accuracy of the best model on the test data?                  #
#############################################################################
# 2 lines of code expected


yy_test = np.ones(y_test.shape)
yy_test[y_test==0] = -1
test_pred = svm.predict(X_test)
print "Accuracy of model on test data is: ", metrics.accuracy_score(yy_test,test_pred)


#############################################################################
# Interpreting the coefficients of an SVM                                   #
# which words are the top predictors of spam?                               #
#############################################################################
# 4 lines of code expected

words, inv_words = utils.get_vocab_dict()

index = np.argsort(svm.theta)[-15:]
print "Top 15 predictors of spam are: "
for i in range(-1,-16,-1):
    print words[index[i]+1]


