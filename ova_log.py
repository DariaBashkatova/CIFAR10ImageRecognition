from sklearn import linear_model
import numpy as np


class ovaLogisticRegressor:

	def __init__(self,labels):
		self.theta = None
		self.reg = None
		self.penalty = None
		self.labels = labels

	def train(self, X, y, reg, penalty):
		"""
		Uses sklearn LogisticRegression for training K classifiers in one-vs-rest mode

		X = m X d+1 array of training data
		y = 1 dimensional vector of length m (with K labels)
		reg = regularization strength
		penalty = "l1" or "l2"
		Returns coefficents for K classifiers: a matrix with K rows and d columns
			- one theta of length d for each class
		"""
		self.reg = reg
		self.penalty = penalty
		m, dim = X.shape
		theta_opt = np.zeros((len(self.labels), dim))

		for i in range(len(self.labels)):
			one_vs_all_labels = np.vectorize(lambda x: 1 if x == self.labels[i] else 0)(y)
			if penalty == "l2":
				lreg = linear_model.LogisticRegression(
						penalty=penalty, C=1.0/reg, solver='sag', fit_intercept=False)  # Before used lbfgs
			else:
				lreg = linear_model.LogisticRegression(
						penalty=penalty, C=1.0/reg, solver='liblinear', fit_intercept=False, max_iter=10)
			lreg.fit(X,one_vs_all_labels)
			theta_opt[i] = lreg.coef_

		self.theta = theta_opt
		return theta_opt

	def predict(self, X):
		"""
		Use the trained weights of this linear classifier to predict labels for
		data points.

		Inputs:
		- X: m x d+1 array of training data.

		Returns:
		- y_pred: Predicted output for the data in X. y_pred is a 1-dimensional
		  array of length m, and each element is a class label from one of the
		  set of labels -- the one with the highest probability
		"""
		theta_mult_x = np.dot(self.theta, X.T)
		return np.argmax(theta_mult_x, axis = 0).T
