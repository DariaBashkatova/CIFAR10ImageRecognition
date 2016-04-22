import nn_utils
import numpy as np
import theano
import utils
from nolearn.lasagne import visualize
from scipy import stats
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
import sys


# Initialize variables
sys.setrecursionlimit(50000)  # Helps for pickling large amounts of data
training_examples = 50000  # Max = 50000
FINAL_RUN = True
ensemble = None  # "voting", "softmax_average", and None also
feature_extraction = True
read_filename = "cnn25_1-?"
write_filename = "cnn25_2-?"

X_train = utils.load("X2d.pickle")
y_train = utils.get_y("data/trainLabels.csv")[range(training_examples)]
if FINAL_RUN:
	print "FINAL RUN!"
else:
	print "TEST RUN!"


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
if read_filename[4] == "_" and int(read_filename[3]) <= 5:  # Separates out previous models
	X_train /= 255.0  # Scaling to [0, 1] used before
	X_test /= 255.0  # Scaling to [0, 1] used before
else:
	scaler = StandardScaler(with_std=False).fit(X_train.reshape((X_train.shape[0], 3 * 32 * 32)))  # Subtract per-pixel mean from data sets
	X_train = scaler.transform(X_train.reshape((X_train.shape[0], 3 * 32 * 32))).reshape((X_train.shape[0], 3, 32, 32))
	X_test = scaler.transform(X_test.reshape((X_test.shape[0], 3 * 32 * 32))).reshape((X_test.shape[0], 3, 32, 32))
X_train = X_train.astype('float32')  # need this cast to use GPU
X_test = X_test.astype('float32')  # need this case to use GPU
y_train = y_train.astype(np.uint8)
if y_test is not None:
	y_test = y_test.astype(np.uint8)

# Train and Test Model
print "Training CNN..."
nn = nn_utils.resnet_cnn(2)

if ensemble == "voting":  # Voting Scheme
	y_test_pred_list = []
	for nn_file in ["cnn1_1-828", "cnn2_3-826", "cnn3_4-832", "cnn4_1-821", "cnn5_1-819"]:
		print nn_file
		nn = utils.load(nn_file + ".pickle")
		y_test_pred_list.append(nn.predict(X_test))
	y_test_pred_matrix = np.array(y_test_pred_list).T
	best_y_test_pred = stats.mode(y_test_pred_matrix, axis=1)[0][:, 0]
	best_accuracy = np.mean(y_test == best_y_test_pred)

elif ensemble == "softmax_average" or ensemble == "softmax_log_average":  # Take the argmax of the softmax average among models
	softmax_sum = np.zeros((X_test.shape[0], 10))
	nn_files = ["cnn1_1-828", "cnn2_3-826", "cnn3_4-832", "cnn4_1-821", "cnn5_1-819"]
	for nn_file in nn_files:
		print nn_file
		nn = utils.load(nn_file + ".pickle")
		X_train_softmax = nn_utils.feature_extraction_from_nn(
			nn, "output", X_train)  # filename="X_train_extracted_output" + nn_file + ".pickle"
		X_test_softmax = nn_utils.feature_extraction_from_nn(
			nn, "output", X_test)  # filename="X_test_extracted_output" + nn_file + ".pickle"
		if ensemble == "softmax_log_average":
			softmax_sum += np.log(X_test_softmax)  # Softmax log average works a little better
		else:
			softmax_sum += X_test_softmax  # Softmax log average works a little better

	print softmax_sum / (1.0 * len(nn_files))
	best_y_test_pred = np.argmax(softmax_sum, axis=1)

else:  # Using a single model only
	print "Loading Model!"
	nn = utils.load(read_filename + ".pickle")

	# nn.fit(X_train, y_train)
	#
	# print "Pickling Model..."
	# utils.dump(nn, write_filename + ".pickle")

	print "Predicting on Train and Test Sets!"
	best_y_test_pred = nn.predict(X_test)
	train_pred = nn.predict(X_train)
	train_accuracy = np.mean(train_pred == y_train)
	print "CNN Train Accuracy: ", train_accuracy

if feature_extraction:
	print "Extracting Features!"
	nn_utils.feature_extraction_from_nn(nn, "globalpool", X_train, "X_train_extracted" + write_filename + ".pickle")
	nn_utils.feature_extraction_from_nn(nn, "globalpool", X_test, "X_test_extracted" + write_filename + ".pickle")


if FINAL_RUN:
	print "Printing Final Results to File..."
	utils.y_to_csv(best_y_test_pred, "data/" + write_filename + "TestLabels.csv")


# Evaluate the best classifier on test set (if results are known)
if y_test is not None:
	best_accuracy = np.mean(best_y_test_pred == y_test)
	print "CNN Test Accuracy: ", best_accuracy
	utils.print_accuracy_report(y_test, best_y_test_pred)


visualize.plot_conv_weights(nn.layers_['conv1'])  # Code from Christian Perone

