import numpy as np
import theano
import utils
from lasagne import layers
from lasagne.init import HeNormal
from lasagne.nonlinearities import softmax
from lasagne.nonlinearities import rectify
from nolearn.lasagne import BatchIterator
from nolearn.lasagne import NeuralNet
from sklearn import model_selection, preprocessing
from scipy import stats
import random
import time


# Initialize variables
FINAL_RUN = False
training_examples = 50000  # Max = 50000

# X_train = utils.get_X("data/train", training_examples, hog_repr=hog_repr, bins=bins)
# X_train = utils.load("X.pickle")
# X_train = utils.load2d("data/train", training_examples)
X_train = utils.load("X2d.pickle")

y_train = utils.get_y("data/trainLabels.csv")[range(training_examples)]



# Create training, validation, and test data sets
print "Creating Train and Test Sets..."
if FINAL_RUN:  # When running on Training Data and untouched Test Data!
	X_test = utils.load2d("data/test", 300000)
	y_test = None
else:  # When running ONLY on Training Data!
	X_train, X_test, y_train, y_test = model_selection.train_test_split(X_train, y_train, test_size=0.2, random_state=0)


# Preprocess data
print "Preprocessing Data..."
poly = preprocessing.PolynomialFeatures(1)
# scaler = preprocessing.StandardScaler().fit(X_train)
# X_train = poly.fit_transform(scaler.transform(X_train))
# X_test = poly.fit_transform(scaler.transform(X_test))
X_train /= 255.0
X_test /= 255.0
y_train = y_train.astype(np.uint8)
y_test = y_test.astype(np.uint8)


# Train and Test Model
print "Training CNN..."


class FlipBatchIterator(BatchIterator):

	def transform(self, Xb, yb):
		Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)

		# Flip half of the images in this batch at random:
		bs = Xb.shape[0]
		indices = np.random.choice(bs, bs / 2, replace=False)
		Xb[indices] = Xb[indices, :, :, ::-1]

		return Xb, yb


class AdjustVariable(object):
	def __init__(self, name, start=0.01, stop=0.001):
		self.name = name
		self.start, self.stop = start, stop
		self.ls = None

	def __call__(self, nn, train_history):
		if self.ls is None:
			self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

		epoch = train_history[-1]['epoch']
		new_value = utils.float32(self.ls[epoch - 1])
		getattr(nn, self.name).set_value(new_value)
		

class EarlyStopping(object):
	def __init__(self, patience=100):
		self.patience = patience
		self.best_valid = np.inf
		self.best_valid_epoch = 0
		self.best_weights = None

	def __call__(self, nn, train_history):
		current_valid = train_history[-1]['valid_loss']
		current_epoch = train_history[-1]['epoch']
		if current_valid < self.best_valid:
			self.best_valid = current_valid
			self.best_valid_epoch = current_epoch
			self.best_weights = nn.get_all_params_values()
		elif self.best_valid_epoch + self.patience < current_epoch:
			print("Early stopping.")
			print("Best valid loss was {:.6f} at epoch {}.".format(
				self.best_valid, self.best_valid_epoch))
			nn.load_params_from(self.best_weights)
			raise StopIteration()


nn = NeuralNet(
		layers=[
			('input', layers.InputLayer),
			('conv1', layers.Conv2DLayer),
			('pool1', layers.MaxPool2DLayer),
			('dropout1', layers.DropoutLayer),
			('conv2', layers.Conv2DLayer),
			('pool2', layers.MaxPool2DLayer),
			('dropout2', layers.DropoutLayer),
			('conv3', layers.Conv2DLayer),
			('pool3', layers.MaxPool2DLayer),
			('dropout3', layers.DropoutLayer),
			('hidden4', layers.DenseLayer),
			('dropout4', layers.DropoutLayer),
			('hidden5', layers.DenseLayer),
			('output', layers.DenseLayer),
			],
		input_shape=(None, 3, 32, 32),
		conv1_num_filters=32, conv1_filter_size=(5, 5), conv1_W=HeNormal(), pool1_pool_size=(2, 2), dropout1_p=0.5,
		conv2_num_filters=32, conv2_filter_size=(5, 5), conv2_W=HeNormal(), pool2_pool_size=(2, 2), dropout2_p=0.5,
		conv3_num_filters=64, conv3_filter_size=(5, 5), conv3_W=HeNormal(), pool3_pool_size=(2, 2), dropout3_p=0.5,
		hidden4_num_units=64, hidden4_W=HeNormal(), dropout4_p=0.5, hidden5_num_units=10, hidden5_W=HeNormal(),
		output_num_units=10, output_nonlinearity=softmax,

		update_learning_rate=theano.shared(utils.float32(0.01)),  # constant lr = 0.01 works great
		update_momentum=theano.shared(utils.float32(0.9)),

		batch_iterator_train=FlipBatchIterator(batch_size=128),
		on_epoch_finished=[
			AdjustVariable('update_learning_rate', start=0.01, stop=0.0001),
			AdjustVariable('update_momentum', start=0.9, stop=0.999),
			EarlyStopping(patience=200),
		],

		max_epochs=100,
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


