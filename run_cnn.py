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
from scipy import stats
from sklearn import model_selection, preprocessing
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
X_train = X_train.astype('float32')  # need this cast to use GPU
X_test = X_test.astype('float32')  # need this case to use GPU


# Train and Test Model
print "Training CNN..."

# nn = NeuralNet(
		# layers=[
		# 	('input', layers.InputLayer),
		# 	('conv1', layers.Conv2DLayer),
		# 	('bn1', layers.BatchNormLayer),
		# 	('conv2', layers.Conv2DLayer),
		# 	('bn2', layers.BatchNormLayer),
		# 	('conv3', layers.Conv2DLayer),
		# 	('bn3', layers.BatchNormLayer),
		# 	('conv4', layers.Conv2DLayer),
		# 	('bn4', layers.BatchNormLayer),
		# 	('conv5', layers.Conv2DLayer),
		# 	('bn5', layers.BatchNormLayer),
		# 	('conv6', layers.Conv2DLayer),
		# 	('bn6', layers.BatchNormLayer),
		# 	('conv7', layers.Conv2DLayer),
		# 	('globalpool', layers.GlobalPoolLayer),
		# 	('output', layers.DenseLayer),
		# 	],
        #
		# input_shape=(None, 3, 32, 32),
		# conv1_num_filters=16, conv1_filter_size=(3, 3), conv1_pad=1, conv1_W=HeNormal(),
		# conv2_num_filters=16, conv2_filter_size=(3, 3), conv2_pad=1, conv2_W=HeNormal(),
		# conv3_num_filters=16, conv3_filter_size=(3, 3), conv3_pad=1, conv3_W=HeNormal(),
        #
		# conv4_num_filters=32, conv4_filter_size=(3, 3), conv4_pad=1, conv4_W=HeNormal(), conv4_stride=2,
		# conv5_num_filters=32, conv5_filter_size=(3, 3), conv5_pad=1, conv5_W=HeNormal(),
        #
		# conv6_num_filters=64, conv6_filter_size=(3, 3), conv6_pad=1, conv6_W=HeNormal(), conv6_stride=2,
		# conv7_num_filters=64, conv7_filter_size=(3, 3), conv7_pad=1, conv7_W=HeNormal(),
        #
		# output_num_units=10, output_nonlinearity=softmax,
        #
		# update_learning_rate=theano.shared(utils.float32(0.01)),
		# update_momentum=theano.shared(utils.float32(0.9)),
        #
		# batch_iterator_train=nn_utils.FlipBatchIterator(batch_size=128),
		# on_epoch_finished=[
		# 	# nn_utils.AdjustVariable('update_learning_rate', start=0.1, stop=0.1),
		# 	# nn_utils.AdjustVariable('update_momentum', start=0.9, stop=0.9),
		# 	nn_utils.EarlyStopping(patience=20),
		# ],
        #
		# max_epochs=500,
		# verbose=1,
		# )


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
        ('conv20', layers.Conv2DLayer),
        ('bn20', layers.BatchNormLayer),
        ('conv21', layers.Conv2DLayer),
        ('bn21', layers.BatchNormLayer),
        ('conv22', layers.Conv2DLayer),
        ('bn22', layers.BatchNormLayer),
        ('conv23', layers.Conv2DLayer),
        ('bn23', layers.BatchNormLayer),
        ('conv24', layers.Conv2DLayer),
        ('bn24', layers.BatchNormLayer),
        ('conv25', layers.Conv2DLayer),
        ('bn25', layers.BatchNormLayer),
        ('conv26', layers.Conv2DLayer),
        ('bn26', layers.BatchNormLayer),
        ('conv27', layers.Conv2DLayer),
        ('bn27', layers.BatchNormLayer),
        ('conv28', layers.Conv2DLayer),
        ('bn28', layers.BatchNormLayer),
        ('conv29', layers.Conv2DLayer),
        ('bn29', layers.BatchNormLayer),
        ('conv30', layers.Conv2DLayer),
        ('bn30', layers.BatchNormLayer),
        ('conv31', layers.Conv2DLayer),
        ('bn31', layers.BatchNormLayer),
        ('conv32', layers.Conv2DLayer),
        ('bn32', layers.BatchNormLayer),
        ('conv33', layers.Conv2DLayer),
        ('bn33', layers.BatchNormLayer),
        ('conv34', layers.Conv2DLayer),
        ('bn34', layers.BatchNormLayer),
        ('conv35', layers.Conv2DLayer),
        ('bn35', layers.BatchNormLayer),
        ('conv36', layers.Conv2DLayer),
        ('bn36', layers.BatchNormLayer),
        ('conv37', layers.Conv2DLayer),
        ('bn37', layers.BatchNormLayer),
        ('conv38', layers.Conv2DLayer),
        ('bn38', layers.BatchNormLayer),
        ('conv39', layers.Conv2DLayer),
        ('bn39', layers.BatchNormLayer),
        ('conv40', layers.Conv2DLayer),
        ('bn40', layers.BatchNormLayer),
        ('conv41', layers.Conv2DLayer),
        ('bn41', layers.BatchNormLayer),
        ('conv42', layers.Conv2DLayer),
        ('bn42', layers.BatchNormLayer),
        ('conv43', layers.Conv2DLayer),
        ('bn43', layers.BatchNormLayer),
        ('conv44', layers.Conv2DLayer),
        ('bn44', layers.BatchNormLayer),
        ('conv45', layers.Conv2DLayer),
        ('bn45', layers.BatchNormLayer),
        ('conv46', layers.Conv2DLayer),
        ('bn46', layers.BatchNormLayer),
        ('conv47', layers.Conv2DLayer),
        ('bn47', layers.BatchNormLayer),
        ('conv48', layers.Conv2DLayer),
        ('bn48', layers.BatchNormLayer),
        ('conv49', layers.Conv2DLayer),
        ('bn49', layers.BatchNormLayer),
        ('conv50', layers.Conv2DLayer),
        ('bn50', layers.BatchNormLayer),
        ('conv51', layers.Conv2DLayer),
        ('bn51', layers.BatchNormLayer),
        ('conv52', layers.Conv2DLayer),
        ('bn52', layers.BatchNormLayer),
        ('conv53', layers.Conv2DLayer),
        ('bn53', layers.BatchNormLayer),
        ('conv54', layers.Conv2DLayer),
        ('bn54', layers.BatchNormLayer),
        ('conv55', layers.Conv2DLayer),
        ('bn55', layers.BatchNormLayer),
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
        conv8_num_filters=16, conv8_filter_size=(3, 3), conv8_pad=1, conv8_W=HeNormal(),
        conv9_num_filters=16, conv9_filter_size=(3, 3), conv9_pad=1, conv9_W=HeNormal(),
        conv10_num_filters=16, conv10_filter_size=(3, 3), conv10_pad=1, conv10_W=HeNormal(),
        conv11_num_filters=16, conv11_filter_size=(3, 3), conv11_pad=1, conv11_W=HeNormal(),
        conv12_num_filters=16, conv12_filter_size=(3, 3), conv12_pad=1, conv12_W=HeNormal(),
        conv13_num_filters=16, conv13_filter_size=(3, 3), conv13_pad=1, conv13_W=HeNormal(),
        conv14_num_filters=16, conv14_filter_size=(3, 3), conv14_pad=1, conv14_W=HeNormal(),
        conv15_num_filters=16, conv15_filter_size=(3, 3), conv15_pad=1, conv15_W=HeNormal(),
        conv16_num_filters=16, conv16_filter_size=(3, 3), conv16_pad=1, conv16_W=HeNormal(),
        conv17_num_filters=16, conv17_filter_size=(3, 3), conv17_pad=1, conv17_W=HeNormal(),
        conv18_num_filters=16, conv18_filter_size=(3, 3), conv18_pad=1, conv18_W=HeNormal(),
        conv19_num_filters=16, conv19_filter_size=(3, 3), conv19_pad=1, conv19_W=HeNormal(),

        conv20_stride=2,
        conv20_num_filters=32, conv20_filter_size=(3, 3), conv20_pad=1, conv20_W=HeNormal(),
        conv21_num_filters=32, conv21_filter_size=(3, 3), conv21_pad=1, conv21_W=HeNormal(),
        conv22_num_filters=32, conv22_filter_size=(3, 3), conv22_pad=1, conv22_W=HeNormal(),
        conv23_num_filters=32, conv23_filter_size=(3, 3), conv23_pad=1, conv23_W=HeNormal(),
        conv24_num_filters=32, conv24_filter_size=(3, 3), conv24_pad=1, conv24_W=HeNormal(),
        conv25_num_filters=32, conv25_filter_size=(3, 3), conv25_pad=1, conv25_W=HeNormal(),
        conv26_num_filters=32, conv26_filter_size=(3, 3), conv26_pad=1, conv26_W=HeNormal(),
        conv27_num_filters=32, conv27_filter_size=(3, 3), conv27_pad=1, conv27_W=HeNormal(),
        conv28_num_filters=32, conv28_filter_size=(3, 3), conv28_pad=1, conv28_W=HeNormal(),
        conv29_num_filters=32, conv29_filter_size=(3, 3), conv29_pad=1, conv29_W=HeNormal(),
        conv30_num_filters=32, conv30_filter_size=(3, 3), conv30_pad=1, conv30_W=HeNormal(),
        conv31_num_filters=32, conv31_filter_size=(3, 3), conv31_pad=1, conv31_W=HeNormal(),
        conv32_num_filters=32, conv32_filter_size=(3, 3), conv32_pad=1, conv32_W=HeNormal(),
        conv33_num_filters=32, conv33_filter_size=(3, 3), conv33_pad=1, conv33_W=HeNormal(),
        conv34_num_filters=32, conv34_filter_size=(3, 3), conv34_pad=1, conv34_W=HeNormal(),
        conv35_num_filters=32, conv35_filter_size=(3, 3), conv35_pad=1, conv35_W=HeNormal(),
        conv36_num_filters=32, conv36_filter_size=(3, 3), conv36_pad=1, conv36_W=HeNormal(),
        conv37_num_filters=32, conv37_filter_size=(3, 3), conv37_pad=1, conv37_W=HeNormal(),

        conv38_stride=2,
        conv38_num_filters=64, conv38_filter_size=(3, 3), conv38_pad=1, conv38_W=HeNormal(),
        conv39_num_filters=64, conv39_filter_size=(3, 3), conv39_pad=1, conv39_W=HeNormal(),
        conv40_num_filters=64, conv40_filter_size=(3, 3), conv40_pad=1, conv40_W=HeNormal(),
        conv41_num_filters=64, conv41_filter_size=(3, 3), conv41_pad=1, conv41_W=HeNormal(),
        conv42_num_filters=64, conv42_filter_size=(3, 3), conv42_pad=1, conv42_W=HeNormal(),
        conv43_num_filters=64, conv43_filter_size=(3, 3), conv43_pad=1, conv43_W=HeNormal(),
        conv44_num_filters=64, conv44_filter_size=(3, 3), conv44_pad=1, conv44_W=HeNormal(),
        conv45_num_filters=64, conv45_filter_size=(3, 3), conv45_pad=1, conv45_W=HeNormal(),
        conv46_num_filters=64, conv46_filter_size=(3, 3), conv46_pad=1, conv46_W=HeNormal(),
        conv47_num_filters=64, conv47_filter_size=(3, 3), conv47_pad=1, conv47_W=HeNormal(),
        conv48_num_filters=64, conv48_filter_size=(3, 3), conv48_pad=1, conv48_W=HeNormal(),
        conv49_num_filters=64, conv49_filter_size=(3, 3), conv49_pad=1, conv49_W=HeNormal(),
        conv50_num_filters=64, conv50_filter_size=(3, 3), conv50_pad=1, conv50_W=HeNormal(),
        conv51_num_filters=64, conv51_filter_size=(3, 3), conv51_pad=1, conv51_W=HeNormal(),
        conv52_num_filters=64, conv52_filter_size=(3, 3), conv52_pad=1, conv52_W=HeNormal(),
        conv53_num_filters=64, conv53_filter_size=(3, 3), conv53_pad=1, conv53_W=HeNormal(),
        conv54_num_filters=64, conv54_filter_size=(3, 3), conv54_pad=1, conv54_W=HeNormal(),
        conv55_num_filters=64, conv55_filter_size=(3, 3), conv55_pad=1, conv55_W=HeNormal(),

        output_num_units=10, output_nonlinearity=softmax,

        # Hyperparameters
        batch_iterator_train=nn_utils.FlipBatchIterator(batch_size=128),
        update_learning_rate=theano.shared(utils.float32(0.01)),
        update_momentum=theano.shared(utils.float32(0.9)),
        # on_epoch_finished=[
            # nn_utils.AdjustVariable('update_learning_rate', start=0.1, stop=0.1),
            # nn_utils.AdjustVariable('update_momentum', start=0.9, stop=0.9),
            # nn_utils.EarlyStopping(patience=20),
        # ],
        max_epochs=1000,
        verbose=1,
        )


nn.fit(X_train, y_train)
utils.dump(nn, "cnn1.pickle")

best_y_test_pred = nn.predict(X_test)
train_pred = nn.predict(X_train)
train_accuracy = np.mean(train_pred == y_train)
print "CNN Train Accuracy: ", train_accuracy

if FINAL_RUN:
	utils.y_to_csv(best_y_test_pred, "data/testLabels.csv")


# Evaluate the best classifier on test set (if results are known)
if y_test is not None:
	best_accuracy = np.mean(best_y_test_pred == y_test)
	print "CNN Test Accuracy: ", best_accuracy
	utils.print_accuracy_report(y_test, best_y_test_pred)


