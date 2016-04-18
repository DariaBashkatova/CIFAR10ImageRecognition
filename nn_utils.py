from nolearn.lasagne import BatchIterator
import numpy as np


class FlipBatchIterator(BatchIterator):

	def transform(self, Xb, yb):
		Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)

		# Flip half of the images in this batch at random:
		bs = Xb.shape[0]
		indices = np.random.choice(bs, bs / 2, replace=False)
		Xb[indices] = Xb[indices, :, :, ::-1]

		return Xb, yb


class AdjustVariable(object):
	def __init__(self, name, start=0.01, stop=0.001, weight_decay='exp'):
		self.name = name
		self.start, self.stop = start, stop
		self.ls = None

	def __call__(self, nn, train_history):
		if self.ls is None:
			if weight_decay == 'linear':
				self.ls = np.linspace(self.start, self.stop, nn.max_epochs)  # Linear Weight Decay
			else:
				self.ls = np.logspace(self.start, self.stop, nn.max_epochs)  # Exponential Weight Decay

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


def print_resnet_cnn(n):
	"""
	Prints out the python/lasagne code to generate a resnet-specification CIFAR 10 CNN
	- n is the resnet parameter: number of layers is 6n+2, and n=18 is recommended
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
		  "    # Hyperparameters\n" + \
		  "    batch_iterator_train=nn_utils.FlipBatchIterator(batch_size=128),\n" +\
		  "    update_learning_rate=theano.shared(utils.float32(0.1)),\n" +\
		  "    update_momentum=theano.shared(utils.float32(0.9)),\n" +\
		  "    on_epoch_finished=[\n" +\
		  "        # nn_utils.AdjustVariable('update_learning_rate', start=0.1, stop=0.1),\n" +\
		  "        # nn_utils.AdjustVariable('update_momentum', start=0.9, stop=0.9),\n" +\
		  "        nn_utils.EarlyStopping(patience=30),\n" +\
		  "    ],\n" +\
		  "    objective_l2=0.0001,\n" +\
		  "    max_epochs=10000,\n" +\
		  "    verbose=1,\n" +\
		  "    )\n"

	return

# print_resnet_cnn(1)
