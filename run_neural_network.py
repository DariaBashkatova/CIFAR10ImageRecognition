import numpy as np
import utils
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier



# Initialize variables
FINAL_RUN = False
bins = False

class_to_value_mapping = {"airplane": 0, "automobile": 1, "bird": 2, "cat": 3, "deer": 4,
						"dog": 5, "frog": 6, "horse": 7, "ship": 8, "truck": 9}
value_to_class_mapping = {0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
						5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"}

X_train = utils.get_X("data/train", 50000, bins=bins)
y_train = utils.get_y("data/trainLabels.csv", class_to_value_mapping)


# Create training, validation, and test data sets
print "Creating Train and Test Sets..."
if FINAL_RUN:  # When running on Training Data and untouched Test Data!
	X_test = utils.get_X("data/test", 300000)
	y_test = None
else:  # When running ONLY on Training Data!
	X_train, X_test, y_train, y_test = model_selection.train_test_split(X_train, y_train, test_size=0.2)


# Train, Predict, and Store Results with Neural Network Model!
nn = MLPClassifier(
		activation='relu', algorithm='l-bfgs', alpha=1e-05,
		batch_size='auto', beta_1=0.9, beta_2=0.999, early_stopping=False,
		epsilon=1e-08, hidden_layer_sizes=(15,), learning_rate='constant',
		learning_rate_init=0.001, max_iter=200, momentum=0.9,
		nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
		tol=0.0001, validation_fraction=0.1, verbose=False,
		warm_start=False)
y_test_pred = nn.fit(X_train, y_train).predict(X_test)
utils.y_to_csv(y_test_pred, value_to_class_mapping, "data/testLabels.csv")


# Evaluate the best softmax classifier on test set (if results are known)
if y_test is not None:
	test_accuracy = np.mean(y_test == y_test_pred)
	print "Neural Network Test Set Accuracy: ", test_accuracy

	c_matrix = confusion_matrix(y_test, y_test_pred)
	print c_matrix

	num_songs_actual = np.sum(c_matrix, axis=1)
	accuracy_per_class = np.zeros(10)
	for i in range(10):
		accuracy_per_class[i] = c_matrix[i, i] / (1.0 * num_songs_actual[i])
	print accuracy_per_class.round(3) * 100
