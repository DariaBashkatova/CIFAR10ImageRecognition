from lasagne import layers
from lasagne.init import HeNormal
from lasagne.nonlinearities import softmax
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import BatchIterator
import numpy as np
import utils
import theano

class FlipBatchIterator(BatchIterator):
	"""
	Code credit to Daniel Nouri and his Nolearn Lasagne tutorial
	- http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/
	- Usage: on_epoch_finished=[AdjustVariable('update_learning_rate', start=0.1, stop=0.1)]
	"""
	def transform(self, Xb, yb):
		Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)

		# Flip half of the images in this batch at random:
		bs = Xb.shape[0]
		indices = np.random.choice(bs, bs / 2, replace=False)
		Xb[indices] = Xb[indices, :, :, ::-1]

		return Xb, yb


class AdjustVariable(object):
	"""
	Code credit to Daniel Nouri and his Nolearn Lasagne tutorial
	- http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/
	- Usage: on_epoch_finished = [AdjustVariable('update_momentum', start=0.9, stop=0.9)]
	"""
	def __init__(self, name, start=0.01, stop=0.001, weight_decay='exp'):
		self.name = name
		self.start, self.stop = start, stop
		self.ls = None
		self.weight_decay = weight_decay

	def __call__(self, nn, train_history):
		if self.ls is None:
			if self.weight_decay == 'linear':
				self.ls = np.linspace(self.start, self.stop, nn.max_epochs)  # Linear Weight Decay
			else:
				self.ls = np.logspace(self.start, self.stop, nn.max_epochs)  # Exponential Weight Decay

		epoch = train_history[-1]['epoch']
		new_value = utils.float32(self.ls[epoch - 1])
		getattr(nn, self.name).set_value(new_value)


class EarlyStopping(object):
	"""
	Code credit to Daniel Nouri and his Nolearn Lasagne tutorial
	- http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/
	- Usage: on_epoch_finished=[EarlyStopping(patience=30)]
	"""
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


def print_resnet_cnn(n):
	"""
	Prints out the python/lasagne code to generate a resnet-specification CIFAR 10 CNN
	- n is the resnet size parameter: number of layers is 6n+2, and n=18 is recommended,
		but n=1 trained for many iterations (~32K) will give better results for the time spent
	Network Architecture and Hyperparameters Credit to He, Zhang, Ren, and Sun at Microsoft Research
	- (http://arxiv.org/pdf/1512.03385v1.pdf)
	"""
	print "nn = NeuralNet(\n" +\
		  "    layers=[\n" +\
		  "        ('input', layers.InputLayer),\n" +\
		  "        ('conv1', layers.Conv2DLayer),\n" +\
		  "        ('bn1', layers.BatchNormLayer),"

	for i in range(2, 6*n + 2):
		print "        ('conv" + str(i) + "', layers.Conv2DLayer),\n" +\
			  "        ('bn" + str(i) + "', layers.BatchNormLayer),"

	print "        ('globalpool', layers.GlobalPoolLayer),\n" +\
		  "        ('output', layers.DenseLayer),\n" +\
		  "        ],\n\n" +\
		  "    input_shape=(None, 3, 32, 32),\n" +\
		  "    conv1_num_filters=16, conv1_filter_size=(3, 3), conv1_pad=1, conv1_W=HeNormal(),"

	for i in range(1, n + 1):
		print "    conv" + str(2*i) + "_num_filters=16, conv" + str(2*i) +\
			  "_filter_size=(3, 3), conv" + str(2*i) + "_pad=1, conv" + str(2*i) + "_W=HeNormal(),"
		print "    conv" + str(2*i + 1) + "_num_filters=16, conv" + str(2*i + 1) + \
			  "_filter_size=(3, 3), conv" + str(2*i + 1) + "_pad=1, conv" + str(2*i + 1) + "_W=HeNormal(),"

	print "\n    conv" + str(2*n + 2) + "_stride=2,"
	for i in range(n + 1, 2*n + 1):
		print "    conv" + str(2 * i) + "_num_filters=32, conv" + str(2 * i) + \
			  "_filter_size=(3, 3), conv" + str(2 * i) + "_pad=1, conv" + str(2 * i) + "_W=HeNormal(),"
		print "    conv" + str(2 * i + 1) + "_num_filters=32, conv" + str(2 * i + 1) + \
			  "_filter_size=(3, 3), conv" + str(2 * i + 1) + "_pad=1, conv" + str(2 * i + 1) + "_W=HeNormal(),"

	print "\n    conv" + str(4*n + 2) + "_stride=2,"
	for i in range(2*n + 1, 3*n + 1):
		print "    conv" + str(2 * i) + "_num_filters=64, conv" + str(2 * i) + \
			  "_filter_size=(3, 3), conv" + str(2 * i) + "_pad=1, conv" + str(2 * i) + "_W=HeNormal(),"
		print "    conv" + str(2 * i + 1) + "_num_filters=64, conv" + str(2 * i + 1) + \
			  "_filter_size=(3, 3), conv" + str(2 * i + 1) + "_pad=1, conv" + str(2 * i + 1) + "_W=HeNormal(),"

	print "\n    output_num_units=10, output_nonlinearity=softmax,\n\n" + \
		  "    batch_iterator_train=FlipBatchIterator(batch_size=128),\n" +\
		  "    update_learning_rate=theano.shared(utils.float32(0.1)),\n" +\
		  "    update_momentum=theano.shared(utils.float32(0.9)),\n" +\
		  "    objective_l2=0.0001,\n" +\
		  "    max_epochs=32000,\n" +\
		  "    verbose=1,\n" +\
		  "    )\n"

	return


def resnet_cnn(layer_size_param):
	"""
	Code generated using print_resnet_cnn to return a resnet cnn of size 6*layer_size_param+2
	"""
	if layer_size_param == 1:
		return NeuralNet(
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

			batch_iterator_train=FlipBatchIterator(batch_size=128),
			update_learning_rate=theano.shared(utils.float32(0.1)),
			update_momentum=theano.shared(utils.float32(0.9)),
			objective_l2=0.0001,
			max_epochs=32000,
			verbose=1,
			)
	elif layer_size_param == 2:
		return NeuralNet(
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
				('conv8', layers.Conv2DLayer),
				('bn8', layers.BatchNormLayer),
				('conv9', layers.Conv2DLayer),
				('bn9', layers.BatchNormLayer),
				('conv10', layers.Conv2DLayer),
				('bn10', layers.BatchNormLayer),
				('conv11', layers.Conv2DLayer),
				('bn11', layers.BatchNormLayer),
				('conv12', layers.Conv2DLayer),
				('bn12', layers.BatchNormLayer),
				('conv13', layers.Conv2DLayer),
				('bn13', layers.BatchNormLayer),
				('globalpool', layers.GlobalPoolLayer),
				('output', layers.DenseLayer),
			],

			input_shape=(None, 3, 32, 32),
			conv1_num_filters=16, conv1_filter_size=(3, 3), conv1_pad=1, conv1_W=HeNormal(),
			conv2_num_filters=16, conv2_filter_size=(3, 3), conv2_pad=1, conv2_W=HeNormal(),
			conv3_num_filters=16, conv3_filter_size=(3, 3), conv3_pad=1, conv3_W=HeNormal(),
			conv4_num_filters=16, conv4_filter_size=(3, 3), conv4_pad=1, conv4_W=HeNormal(),
			conv5_num_filters=16, conv5_filter_size=(3, 3), conv5_pad=1, conv5_W=HeNormal(),

			conv6_stride=2,
			conv6_num_filters=32, conv6_filter_size=(3, 3), conv6_pad=1, conv6_W=HeNormal(),
			conv7_num_filters=32, conv7_filter_size=(3, 3), conv7_pad=1, conv7_W=HeNormal(),
			conv8_num_filters=32, conv8_filter_size=(3, 3), conv8_pad=1, conv8_W=HeNormal(),
			conv9_num_filters=32, conv9_filter_size=(3, 3), conv9_pad=1, conv9_W=HeNormal(),

			conv10_stride=2,
			conv10_num_filters=64, conv10_filter_size=(3, 3), conv10_pad=1, conv10_W=HeNormal(),
			conv11_num_filters=64, conv11_filter_size=(3, 3), conv11_pad=1, conv11_W=HeNormal(),
			conv12_num_filters=64, conv12_filter_size=(3, 3), conv12_pad=1, conv12_W=HeNormal(),
			conv13_num_filters=64, conv13_filter_size=(3, 3), conv13_pad=1, conv13_W=HeNormal(),

			output_num_units=10, output_nonlinearity=softmax,

			batch_iterator_train=FlipBatchIterator(batch_size=128),
			update_learning_rate=theano.shared(utils.float32(0.1)),
			update_momentum=theano.shared(utils.float32(0.9)),
			objective_l2=0.0001,
			max_epochs=32000,
			verbose=1,
		)
	elif layer_size_param == 3:
		return NeuralNet(
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
				('conv8', layers.Conv2DLayer),
				('bn8', layers.BatchNormLayer),
				('conv9', layers.Conv2DLayer),
				('bn9', layers.BatchNormLayer),
				('conv10', layers.Conv2DLayer),
				('bn10', layers.BatchNormLayer),
				('conv11', layers.Conv2DLayer),
				('bn11', layers.BatchNormLayer),
				('conv12', layers.Conv2DLayer),
				('bn12', layers.BatchNormLayer),
				('conv13', layers.Conv2DLayer),
				('bn13', layers.BatchNormLayer),
				('conv14', layers.Conv2DLayer),
				('bn14', layers.BatchNormLayer),
				('conv15', layers.Conv2DLayer),
				('bn15', layers.BatchNormLayer),
				('conv16', layers.Conv2DLayer),
				('bn16', layers.BatchNormLayer),
				('conv17', layers.Conv2DLayer),
				('bn17', layers.BatchNormLayer),
				('conv18', layers.Conv2DLayer),
				('bn18', layers.BatchNormLayer),
				('conv19', layers.Conv2DLayer),
				('bn19', layers.BatchNormLayer),
				('globalpool', layers.GlobalPoolLayer),
				('output', layers.DenseLayer),
			],

			input_shape=(None, 3, 32, 32),
			conv1_num_filters=16, conv1_filter_size=(3, 3), conv1_pad=1, conv1_W=HeNormal(),
			conv2_num_filters=16, conv2_filter_size=(3, 3), conv2_pad=1, conv2_W=HeNormal(),
			conv3_num_filters=16, conv3_filter_size=(3, 3), conv3_pad=1, conv3_W=HeNormal(),
			conv4_num_filters=16, conv4_filter_size=(3, 3), conv4_pad=1, conv4_W=HeNormal(),
			conv5_num_filters=16, conv5_filter_size=(3, 3), conv5_pad=1, conv5_W=HeNormal(),
			conv6_num_filters=16, conv6_filter_size=(3, 3), conv6_pad=1, conv6_W=HeNormal(),
			conv7_num_filters=16, conv7_filter_size=(3, 3), conv7_pad=1, conv7_W=HeNormal(),

			conv8_stride=2,
			conv8_num_filters=32, conv8_filter_size=(3, 3), conv8_pad=1, conv8_W=HeNormal(),
			conv9_num_filters=32, conv9_filter_size=(3, 3), conv9_pad=1, conv9_W=HeNormal(),
			conv10_num_filters=32, conv10_filter_size=(3, 3), conv10_pad=1, conv10_W=HeNormal(),
			conv11_num_filters=32, conv11_filter_size=(3, 3), conv11_pad=1, conv11_W=HeNormal(),
			conv12_num_filters=32, conv12_filter_size=(3, 3), conv12_pad=1, conv12_W=HeNormal(),
			conv13_num_filters=32, conv13_filter_size=(3, 3), conv13_pad=1, conv13_W=HeNormal(),

			conv14_stride=2,
			conv14_num_filters=64, conv14_filter_size=(3, 3), conv14_pad=1, conv14_W=HeNormal(),
			conv15_num_filters=64, conv15_filter_size=(3, 3), conv15_pad=1, conv15_W=HeNormal(),
			conv16_num_filters=64, conv16_filter_size=(3, 3), conv16_pad=1, conv16_W=HeNormal(),
			conv17_num_filters=64, conv17_filter_size=(3, 3), conv17_pad=1, conv17_W=HeNormal(),
			conv18_num_filters=64, conv18_filter_size=(3, 3), conv18_pad=1, conv18_W=HeNormal(),
			conv19_num_filters=64, conv19_filter_size=(3, 3), conv19_pad=1, conv19_W=HeNormal(),

			output_num_units=10, output_nonlinearity=softmax,

			batch_iterator_train=FlipBatchIterator(batch_size=128),
			update_learning_rate=theano.shared(utils.float32(0.1)),
			update_momentum=theano.shared(utils.float32(0.9)),
			objective_l2=0.0001,
			max_epochs=32000,
			verbose=1,
		)
	else:
		print "Please print initialization code for a resnet of layer size param",\
			layer_size_param, "and add it to resnet_cnn()"


def feature_extraction_from_nn(nn, hidden_layer_name, X, filename=None):
	"""
	Return and, if specified, pickle the hidden layer output of a neural network
		over all the training data, as a method of feature extraction

	Feature Extraction code/hidden layer output code credit to Christian Perone
		and his tutorial (http://blog.christianperone.com/2015/08/convolutional-neural-networks-and-feature-extraction-with-python/)
	"""
	# Set up hidden layer outputting
	input_var = nn.layers_['input'].input_var
	hidden_layer = layers.get_output(nn.layers_[hidden_layer_name], deterministic=True)
	f_hidden = theano.function([input_var], hidden_layer)

	# Transform the data, example by example
	Xtracted = np.zeros((X.shape[0], layers.get_output_shape(nn.layers_[hidden_layer_name])[1]))  # only outputs 2d matrix
	for i in range(X.shape[0]):
		if len(X.shape) == 2:
			Xtracted[i] = f_hidden(X[i][None, :])
		elif len(X.shape) == 3:
			Xtracted[i] = f_hidden(X[i][None, :, :])
		elif len(X.shape) == 4:
			Xtracted[i] = f_hidden(X[i][None, :, :, :])


	if filename is not None:
		utils.dump(Xtracted, filename)

	return Xtracted

