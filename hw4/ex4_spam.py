from sklearn import cross_validation, preprocessing, metrics
import utils
import scipy.io
import numpy as np
from linear_classifier import LinearSVM_twoclass


#############################################################################
# load the SPAM email training and test dataset                             #
#############################################################################

poly = preprocessing.PolynomialFeatures(1)
X,y = utils.load_mat('data/spamTrain.mat')
XX = poly.fit_transform(X)
yy = np.ones(y.shape)
yy[y==0] = -1

test_data = scipy.io.loadmat('data/spamTest.mat')
X_test = test_data['Xtest']
XX_test = poly.fit_transform(X_test)
y_test = test_data['ytest'].flatten()
yy_test = np.ones(y_test.shape)
yy_test[y_test==0] = -1

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
XX_train = poly.fit_transform(X_train)
XX_val = poly.fit_transform(X_val)


# Some somewhat arbitrary initializations
best_accuracy = -1.0
best_C = 1
best_kernel_param = 10
best_lr = 1e0
best_num_iters = 100


kernel = "polynomial"

if kernel == "gaussian":
	kernel = utils.gaussian_kernel
	kernel_param_vals = [1, 3, 10, 30]  # For sigma
	Cvals = [.1, 1, 10]
	learning_rates = [3e-1, 1e0, 3e0]
	num_iters_list = [100]

elif kernel == "polynomial":
	kernel = utils.polynomial_kernel
	kernel_param_vals = [-10, -1, 0, 1, 10]  # For c
	Cvals = [.1, 1, 10]
	learning_rates = [1e1, 1e0, 1e-1, 1e-2]
	num_iters_list = [100]

else:
	kernel = None
	kernel_param_vals = [None]
	Cvals = [1, 3, 10]
	learning_rates = [3e0, 1e0, 3e-1]
	num_iters_list = [30, 100, 300]


print "KERNEL: ", kernel

for kernel_param in kernel_param_vals:
	for C in Cvals:
		for lr in learning_rates:
			for num_iters in num_iters_list:
				if kernel is not None:
					# Preprocess train data (Kernelize, scale, and add intercept)
					print "."
					K = np.array([kernel(x1,x2,kernel_param) for x1 in X_train for x2 in X_train]).reshape(X_train.shape[0],X_train.shape[0])
					scaler = preprocessing.StandardScaler().fit(K)
					scale_K = scaler.transform(K)
					KK = poly.fit_transform(scale_K)

					# Preprocess val data (Kernelize, scale, and add intercept)
					print "."
					K_val = np.array([kernel(x1,x2,kernel_param) for x1 in X_val for x2 in X_train]).reshape(X_val.shape[0],X_train.shape[0])
					scale_K_val = scaler.transform(K_val)
					KK_val = poly.fit_transform(scale_K_val)

					# Train model and get val accuracy
					print "."
					svm = LinearSVM_twoclass()
					svm.theta = np.zeros((KK.shape[1],))
					svm.train(KK,y_train,learning_rate=lr,C=C,num_iters=num_iters)
					y_val_pred = svm.predict(KK_val)
					accuracy = np.mean(y_val == y_val_pred)

				else:
					svm = LinearSVM_twoclass()
					svm.train(XX_train,y_train,learning_rate=lr,C=C,num_iters=num_iters,verbose=True)
					y_val_pred = svm.predict(XX_val)
					accuracy = np.mean(y_val == y_val_pred)

				print "LR:", lr, " NumIters:", num_iters, " C:", C, " KP:", kernel_param, " Accuracy:", accuracy
				if accuracy > best_accuracy:
					best_accuracy = accuracy
					best_C = C
					best_kernel_param = kernel_param
					best_lr = lr
					best_num_iters = num_iters

print "Best LR:", best_lr, " Best NumIters:", best_num_iters, " Best C:", best_C, " BestKP:", kernel_param, " Best Accuracy:", best_accuracy

svm.train(XX,y,learning_rate=best_lr,C=best_C,num_iters=best_num_iters)


#############################################################################
#  end of your code                                                         #
#############################################################################

#############################################################################
# what is the accuracy of the best model on the training data itself?       #
#############################################################################
# 2 lines of code expected

y_pred = svm.predict(XX)
print "Accuracy of model on training data is: ", metrics.accuracy_score(yy,y_pred)


#############################################################################
# what is the accuracy of the best model on the test data?                  #
#############################################################################
# 2 lines of code expected



test_pred = svm.predict(XX_test)
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


