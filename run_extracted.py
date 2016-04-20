import numpy as np
import nn_utils
import utils
import random
from lasagne import layers
from lasagne.init import HeNormal
from lasagne.nonlinearities import softmax
from nolearn.lasagne import NeuralNet
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn import tree
from scipy import stats
import theano



# Initialize variables
FINAL_RUN = True
model_type = "SVM"  # Supports NB, DT, SVM, NN
print model_type

# X_train = utils.get_X("data/train", 50000, bins=bins)
X_train = np.concatenate((utils.load("X_train_extracted1.pickle"), utils.load("X_train_extracted2.pickle")), axis=1)
y_train = utils.get_y("data/trainLabels.csv")

# Create training, validation, and test data sets
print "Creating Train and Test Sets..."
if FINAL_RUN:  # When running on Training Data and untouched Test Data!
	# X_test = utils.get_X("data/test", 300000)
	X_test = np.concatenate((utils.load("X_test_extracted1.pickle"), utils.load("X_test_extracted2.pickle")), axis=1)
	y_test = None
else:  # When running ONLY on Training Data!
	X_train, X_test, y_train, y_test = model_selection.train_test_split(X_train, y_train, test_size=0.2)


# Train, Predict, and Store Results with Model!
print "Training Model!"
if model_type == "NB":
	nb = GaussianNB()
	best_y_test_pred = nb.fit(X_train, y_train).predict(X_test)

elif model_type == "DT":
	y_test_pred_list = []
	num_decision_trees = 1
	for i in range(num_decision_trees):
		print "DT ", i
		dt = tree.DecisionTreeClassifier()
		y_test_pred = dt.fit(X_train, y_train).predict(X_test)
		y_test_pred_list.append(y_test_pred)

	y_test_pred_matrix = np.array(y_test_pred_list).T
	best_y_test_pred = stats.mode(y_test_pred_matrix, axis=1)[0][:, 0]

elif model_type == "SVM":
	svm_model = svm.SVC()
	best_y_test_pred = svm_model.fit(X_train, y_train).predict(X_test)

elif model_type == "NN":
	X_train = X_train.astype('float32')  # need this cast to use GPU
	X_test = X_test.astype('float32')  # need this case to use GPU
	y_train = y_train.astype(np.uint8)
	if y_test is not None:
		y_test = y_test.astype(np.uint8)
	best_nn = None
	best_accuracy = -1.0
	best_y_test_pred = None
	num_iters = 1

	for i in range(num_iters):
		# Train Model
		num_hidden_layers = 8  # Originally (i % 4) + 1
		# hidden_layer_sizes = utils.rand_hidden_layer_sizes(X_train.shape[1], 10, num_hidden_layers)
		# alpha = random.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
		hidden_layer_sizes = [58, 57, 50, 50, 48]
		print hidden_layer_sizes

		# nn = MLPClassifier(
		# 		activation='relu', algorithm='adam', alpha=alpha, batch_size='auto', beta_1=0.9, beta_2=0.999,
		# 		early_stopping=False, epsilon=1e-08, hidden_layer_sizes=hidden_layer_sizes, learning_rate='constant',
		# 		learning_rate_init=1e-3, max_iter=200, momentum=0.9, nesterovs_momentum=True, power_t=0.5,
		# 		random_state=1, shuffle=True, tol=0.0001, validation_fraction=0.1, verbose=True, warm_start=False)

		nn = NeuralNet(
			layers=[
				('input', layers.InputLayer),
				('hidden1', layers.DenseLayer),
				('bn1', layers.BatchNormLayer),
				('hidden2', layers.DenseLayer),
				('bn2', layers.BatchNormLayer),
				('hidden3', layers.DenseLayer),
				('bn3', layers.BatchNormLayer),
				('hidden4', layers.DenseLayer),
				('bn4', layers.BatchNormLayer),
				('hidden5', layers.DenseLayer),
				('bn5', layers.BatchNormLayer),
				('output', layers.DenseLayer),
			],

			input_shape=(None, 64),
			hidden1_num_units=hidden_layer_sizes[0], hidden1_W=HeNormal(),
			hidden2_num_units=hidden_layer_sizes[1], hidden2_W=HeNormal(),
			hidden3_num_units=hidden_layer_sizes[2], hidden3_W=HeNormal(),
			hidden4_num_units=hidden_layer_sizes[3], hidden4_W=HeNormal(),
			hidden5_num_units=hidden_layer_sizes[4], hidden5_W=HeNormal(),
			output_num_units=10, output_nonlinearity=softmax,

			on_epoch_finished=[nn_utils.EarlyStopping(patience=5)],
			update_learning_rate=theano.shared(utils.float32(0.1)),
			update_momentum=theano.shared(utils.float32(0.9)),
			objective_l2=0.0003,
			max_epochs=1000,
			verbose=1,
		)

		nn.fit(X_train, y_train)

		# Feature extract using the nn, then train over with SVM
		print "Extracting Features using NN!"
		Xtracted_train = nn_utils.feature_extraction_from_nn(nn, 'hidden5', X_train)
		Xtracted_test = nn_utils.feature_extraction_from_nn(nn, 'hidden5', X_test)

		print "Training SVM!"
		svm_model = svm.SVC()
		svm_model.fit(X_train, y_train)

		# Validate Model
		y_test_pred = svm_model.predict(X_test)
		y_train_pred = svm_model.predict(X_train)
		# y_test_pred = nn.predict(X_test)
		# y_train_pred = nn.predict(X_train)

		train_accuracy = np.mean(y_train == y_train_pred)
		print hidden_layer_sizes
		print "t", train_accuracy

		if not FINAL_RUN:
			test_accuracy = np.mean(y_test == y_test_pred)
			print "v", test_accuracy
			utils.print_accuracy_report(y_test, y_test_pred)

			if test_accuracy > best_accuracy:
				best_accuracy = test_accuracy
				best_nn = nn
				best_y_test_pred = y_test_pred

		else:
			if train_accuracy > best_accuracy:
				best_accuracy = train_accuracy
				best_nn = nn
				best_y_test_pred = y_test_pred


# Store final results if necessary
if FINAL_RUN:
	utils.y_to_csv(best_y_test_pred, "data/CNN-NN-SVMtestLabels.csv")


# Evaluate the best softmax classifier on test set (if results are known)
if y_test is not None:
	test_accuracy = np.mean(y_test == best_y_test_pred)
	print model_type, "Test Set Accuracy: ", test_accuracy
	utils.print_accuracy_report(y_test, best_y_test_pred)
