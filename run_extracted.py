import numpy as np
import utils
import random
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn import tree
from scipy import stats



# Initialize variables
FINAL_RUN = False
model_type = "NN"  # Supports NB, DT, SVM, NN
print model_type

# X_train = utils.get_X("data/train", 50000, bins=bins)
X_train = utils.load("X_train_extracted.pickle")
y_train = utils.get_y("data/trainLabels.csv")

# Create training, validation, and test data sets
print "Creating Train and Test Sets..."
if FINAL_RUN:  # When running on Training Data and untouched Test Data!
	# X_test = utils.get_X("data/test", 300000)
	X_test_extracted = utils.load("X_test_extracted.pickle")
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
	print "NN"
	best_nn = None
	best_accuracy = -1.0
	best_y_test_pred = None
	num_iters = 10

	for i in range(num_iters):
		# Train Model
		num_hidden_layers = 8  # Originally (i % 4) + 1
		hidden_layer_sizes = utils.rand_hidden_layer_sizes(X_train.shape[1], 10, num_hidden_layers)

		alpha = random.choice([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0])
		# alpha = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 3e-4, 3e-3, 3e-2, 3e-1, 3e0, 3e1, 3e2, 3e3][i]

		nn = MLPClassifier(
				activation='relu', algorithm='adam', alpha=alpha, batch_size='auto', beta_1=0.9, beta_2=0.999,
				early_stopping=False, epsilon=1e-08, hidden_layer_sizes=hidden_layer_sizes, learning_rate='constant',
				learning_rate_init=1e-3, max_iter=200, momentum=0.9, nesterovs_momentum=True, power_t=0.5,
				random_state=1, shuffle=True, tol=0.0001, validation_fraction=0.1, verbose=True, warm_start=False)
		nn.fit(X_train, y_train)

		# Validate Model
		y_test_pred = nn.predict(X_test)
		y_train_pred = nn.predict(X_train)
		test_accuracy = np.mean(y_test == y_test_pred)
		train_accuracy = np.mean(y_train == y_train_pred)
		print hidden_layer_sizes, alpha, "t", train_accuracy, "v", test_accuracy
		utils.print_accuracy_report(y_test, y_test_pred)

		if test_accuracy > best_accuracy:
			best_accuracy = test_accuracy
			best_nn = nn
			best_y_test_pred = y_test_pred

# Store final results if necessary
if FINAL_RUN:
	utils.y_to_csv(best_y_test_pred, "data/testLabels.csv")


# Evaluate the best softmax classifier on test set (if results are known)
if y_test is not None:
	test_accuracy = np.mean(y_test == best_y_test_pred)
	print model_type, "Test Set Accuracy: ", test_accuracy
	utils.print_accuracy_report(y_test, best_y_test_pred)
