import numpy as np
import matplotlib.pyplot as plt
import utils
from sklearn import model_selection
from sklearn.metrics import confusion_matrix, classification_report
from softmax import SoftmaxClassifier


# Initialize variables
FINAL_RUN = False
X = utils.get_X("data/train", 50000)
y = utils.get_y("data/trainLabels.csv")


# Create training, validation, and test data sets
print "Creating Train, Val, and Test Sets..."
if FINAL_RUN:  # When running on Training Data and untouched Test Data!
	X_train_val = X
	y_train_val = y
	X_test = utils.get_X("data/test", 300000)
	y_test = None
else:  # When running ONLY on Training Data!
	X_train_val, X_test, y_train_val, y_test = model_selection.train_test_split(X, y, test_size=0.2)

X_train, X_val, y_train, y_val = model_selection.train_test_split(X_train_val, y_train_val, test_size=0.2)


# Test Models with Various Hyper Parameters
print "Selecting Hyperparameters..."
detailed_results = {}
best_val_accuracy = -1
train_accuracy_of_best_softmax = -1
best_softmax_final = None
num_iters_list = [100000]  # 30000 is good enough, 100000 or higher is better
batch_sizes = [1000]  # Larger is better but more time-consuming
learning_rates = [0.003, 0.01, 0.03]
regularization_strengths = [1.0, 3.0, 10.0]  # [3e0]
# Works well: NumIters:  30000,  BatchSize:  1000, LR:  0.01,  Reg:  3.0
# Works well: NumIters:  100000, BatchSize:  1000, LR:  0.003, Reg:  1.0

for num_iters in num_iters_list:
	for batch_size in batch_sizes:
		for alpha in learning_rates:
			for reg in regularization_strengths:
				softmax_model = SoftmaxClassifier()
				softmax_model.train(
						X_train, y_train, learning_rate=alpha, reg=reg, num_iters=num_iters, batch_size=batch_size)

				y_train_pred = softmax_model.predict(X_train)
				train_accuracy = np.mean(y_train_pred == y_train)
				y_val_pred = softmax_model.predict(X_val)
				val_accuracy = np.mean(y_val_pred == y_val)
				detailed_results[(num_iters, reg, alpha, batch_size)] = val_accuracy
				print "NumIters: ", num_iters, " BatchSize: ", batch_size, " LR: ", alpha, " Reg: ", reg,\
					" Train Accuracy: ", train_accuracy, " Val Accuracy: ", val_accuracy

				if val_accuracy > best_val_accuracy:
					best_val_accuracy = val_accuracy
					train_accuracy_of_best_softmax = train_accuracy
					best_softmax = softmax_model

print "\nBEST SOFTMAX HYPER PARAMETERS:"
print "NumIters: ", best_softmax.num_iters, " BatchSize: ", best_softmax.batch_size,\
	" LR: ", best_softmax.learning_rate, " Reg: ", best_softmax.reg,\
	" Train Accuracy: ", train_accuracy_of_best_softmax, " Val Accuracy: ", best_val_accuracy, "\n"


# With ascertained optimal hyperparameters, train new model on Training and Validation Sets
best_softmax_final = SoftmaxClassifier()
best_softmax_final.train(X_train_val, y_train_val, learning_rate=best_softmax.learning_rate, reg=best_softmax.reg,
						 num_iters=best_softmax.num_iters, batch_size=best_softmax.batch_size)


# Make and store predictions!
y_test_pred = best_softmax_final.predict(X_test)
utils.y_to_csv(y_test_pred, "data/testLabels.csv")


# Visualize the learned weights for each class
theta = best_softmax_final.theta[1:,:].T # strip out the bias term
theta = theta.reshape(10, 32, 32, 3)

theta_min, theta_max = np.min(theta), np.max(theta)

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for i in xrange(10):
	plt.subplot(2, 5, i + 1)

	# Rescale the weights to be between 0 and 255
	thetaimg = 255.0 * (theta[i].squeeze() - theta_min) / (theta_max - theta_min)
	plt.imshow(thetaimg.astype('uint8'))
	plt.axis('off')
	plt.title(classes[i])


plt.savefig('cifar_theta.pdf')
plt.close()


# Evaluate the best softmax classifier on test set (if results are known)
if y_test is not None:
	test_accuracy = np.mean(y_test == y_test_pred)
	print 'Softmax Test Set Accuracy: %f' % (test_accuracy, )
	utils.print_accuracy_report(y_test, y_test_pred)
