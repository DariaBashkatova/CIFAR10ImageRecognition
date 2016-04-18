import nn_utils
import numpy as np
import theano
import utils
from lasagne import layers
from lasagne.init import HeNormal
from lasagne.nonlinearities import softmax
from lasagne.nonlinearities import rectify
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize
from scipy import stats
from sklearn import model_selection, preprocessing
import random
import sys
import time


# Initialize variables
sys.setrecursionlimit(50000)
FINAL_RUN = True
training_examples = 50000  # Max = 50000

X_train = utils.load("X2d.pickle")
y_train = utils.get_y("data/trainLabels.csv")[range(training_examples)]


# Create training, validation, and test data sets
print "Creating Train and Test Sets..."
if FINAL_RUN:  # When running on Training Data and untouched Test Data!
	# X_test = utils.load2d("data/test", 300000)
	X_test = np.concatenate((utils.load("XTest2d1.pickle"), utils.load("XTest2d2.pickle"),
							 utils.load("XTest2d3.pickle"), utils.load("XTest2d4.pickle")))
	y_test = None
else:  # When running ONLY on Training Data!
	X_train, X_test, y_train, y_test = model_selection.train_test_split(X_train, y_train, test_size=0.2, random_state=0)


# Preprocess data
print "Preprocessing Data..."
# TODO: Subtract per pixel mean!
X_train /= 255.0
X_test /= 255.0
X_train = X_train.astype('float32')  # need this cast to use GPU
X_test = X_test.astype('float32')  # need this case to use GPU
y_train = y_train.astype(np.uint8)
if y_test is not None:
	y_test = y_test.astype(np.uint8)

# Train and Test Model
print "Training CNN..."
nn = NeuralNet(
	layers=[
		('input', layers.InputLayer),
		('conv1', layers.Conv2DLayer),
		('bn1', layers.BatchNormLayer),
		('conv2', layers.Conv2DLayer),
		('bn2', layers.BatchNormLayer),
		('conv3', layers.Conv2DLayer),
		('bn3', layers.BatchNormLayer),
		('conv4', layers.Conv2DLayer),
		('bn4', layers.BatchNormLayer),
		('conv5', layers.Conv2DLayer),
		('bn5', layers.BatchNormLayer),
		('conv6', layers.Conv2DLayer),
		('bn6', layers.BatchNormLayer),
		('conv7', layers.Conv2DLayer),
		('bn7', layers.BatchNormLayer),
		('globalpool', layers.GlobalPoolLayer),
		('output', layers.DenseLayer),
		],

	input_shape=(None, 3, 32, 32),
	conv1_num_filters=16, conv1_filter_size=(3, 3), conv1_pad=1, conv1_W=HeNormal(),
	conv2_num_filters=16, conv2_filter_size=(3, 3), conv2_pad=1, conv2_W=HeNormal(),
	conv3_num_filters=16, conv3_filter_size=(3, 3), conv3_pad=1, conv3_W=HeNormal(),

	conv4_stride=2,
	conv4_num_filters=32, conv4_filter_size=(3, 3), conv4_pad=1, conv4_W=HeNormal(),
	conv5_num_filters=32, conv5_filter_size=(3, 3), conv5_pad=1, conv5_W=HeNormal(),

	conv6_stride=2,
	conv6_num_filters=64, conv6_filter_size=(3, 3), conv6_pad=1, conv6_W=HeNormal(),
	conv7_num_filters=64, conv7_filter_size=(3, 3), conv7_pad=1, conv7_W=HeNormal(),

	output_num_units=10, output_nonlinearity=softmax,

	batch_iterator_train=nn_utils.FlipBatchIterator(batch_size=128),
	update_learning_rate=theano.shared(utils.float32(0.1)),
	update_momentum=theano.shared(utils.float32(0.9)),
	objective_l2=0.0001,
	max_epochs=32000,
	verbose=1,
	)

# read_filename = "cnn2_1-?"
write_filename = "cnn2_1-?"

# print "Loading Model!"
# nn = utils.load(read_filename + ".pickle")

nn.fit(X_train, y_train)

print "Pickling Model..."
utils.dump(nn, write_filename + ".pickle")

print "Predicting on Train and Test Sets!"
best_y_test_pred = nn.predict(X_test)
train_pred = nn.predict(X_train)
train_accuracy = np.mean(train_pred == y_train)
print "CNN Train Accuracy: ", train_accuracy


feature_extraction = False
if feature_extraction:
	# Feature Extraction using CNN
	input_var = nn.layers_['input'].input_var
	global_pool_layer = layers.get_output(nn.layers_['globalpool'], deterministic=True)
	f_global_pool = theano.function([input_var], global_pool_layer)

	print "Extracting Features for Training Data!"
	X_train_extracted = np.zeros((X_train.shape[0], 64))
	for i in range(X_train.shape[0]):
		X_train_extracted[i] = f_global_pool(X_train[i][None, :, :, :])

	print "Extracting Features for Testing Data!"
	X_test_extracted = np.zeros((X_test.shape[0], 64))
	for i in range(X_test.shape[0]):
		X_test_extracted[i] = f_global_pool(X_test[i][None, :, :, :])

	print "Pickling Data!"
	utils.dump(X_train_extracted, "X_train_extracted.pickle")
	utils.dump(X_test_extracted, "X_test_extracted.pickle")


print "Printing Final Results to File..."
if FINAL_RUN:
	utils.y_to_csv(best_y_test_pred, "data/" + write_filename + "TestLabels.csv")


# Evaluate the best classifier on test set (if results are known)
if y_test is not None:
	best_accuracy = np.mean(best_y_test_pred == y_test)
	print "CNN Test Accuracy: ", best_accuracy
	utils.print_accuracy_report(y_test, best_y_test_pred)


