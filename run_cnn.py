import numpy as np
import utils
from lasagne import layers
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from sklearn import model_selection, preprocessing
from scipy import stats
import random


# Initialize variables
FINAL_RUN = False
bins = False
hog_repr = False
training_examples = 50000  # Max = 50000

# X_train = utils.get_X("data/train", training_examples, hog_repr=hog_repr, bins=bins)
X_train = utils.load("X.pickle")
y_train = utils.get_y("data/trainLabels.csv")[range(training_examples)].reshape(training_examples, 1)
print y_train.shape
y_train = utils.convert_y_to_matrix(y_train)
print y_train.shape



# Create training, validation, and test data sets
print "Creating Train and Test Sets..."
if FINAL_RUN:  # When running on Training Data and untouched Test Data!
	X_test = utils.get_X("data/test", 300000, hog_repr=hog_repr, bins=bins)
	y_test = None
else:  # When running ONLY on Training Data!
	X_train, X_test, y_train, y_test = model_selection.train_test_split(X_train, y_train, test_size=0.2, random_state=0)


# Preprocess data
print "Preprocessing Data..."
poly = preprocessing.PolynomialFeatures(1)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = poly.fit_transform(scaler.transform(X_train))
X_test = poly.fit_transform(scaler.transform(X_test))


# Train and Test Model
print "Training Neural Net..."
nn = NeuralNet(
	layers=[  # three layers: one hidden layer
		('input', layers.InputLayer),
		('hidden', layers.DenseLayer),
		('output', layers.DenseLayer),
		],
	# layer parameters:
	input_shape=(None, X_train.shape[1]),  # 96x96 input pixels per batch
	hidden_num_units=1582,  # number of units in hidden layer
	output_nonlinearity=softmax,  # output layer uses identity function
	output_num_units=10,  # 1 target values

	# optimization method:
	update=nesterov_momentum,
	update_learning_rate=0.01,
	update_momentum=0.9,

	regression=False,  # flag to indicate we're dealing with regression problem
	max_epochs=400,  # we want to train this many epochs
	verbose=1,
	)

nn.fit(X_train, y_train)
best_y_test_pred = nn.predict(X_test)

if FINAL_RUN:
	utils.y_to_csv(best_y_test_pred, "data/testLabels.csv")


# Evaluate the best classifier on test set (if results are known)
if y_test is not None:
	best_accuracy = np.mean(best_y_test_pred == y_test)
	print "Neural Network Test Set Accuracy: ", best_accuracy
	utils.print_accuracy_report(y_test, best_y_test_pred)


