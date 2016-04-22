import numpy as np
import utils
from sklearn import model_selection, preprocessing
from sklearn.neural_network import MLPClassifier
from scipy import stats


# Initialize variables
FINAL_RUN = False
hog_repr = False
training_examples = 50000  # Max = 50000

X_train = utils.get_X("data/train", training_examples, hog_repr=hog_repr)
y_train = utils.get_y("data/trainLabels.csv")[range(training_examples)]

# Create training, validation, and test data sets
print "Creating Train and Test Sets..."
if FINAL_RUN:  # When running on Training Data and untouched Test Data!
	X_test = utils.get_X("data/test", 300000, hog_repr=hog_repr)
	y_test = None
else:  # When running ONLY on Training Data!
	X_train, X_test, y_train, y_test = model_selection.train_test_split(X_train, y_train, test_size=0.2, random_state=0)


if hog_repr:
	print "Preprocessing Data..."
	poly = preprocessing.PolynomialFeatures(1)
	scaler = preprocessing.StandardScaler().fit(X_train)
	X_train = poly.fit_transform(scaler.transform(X_train))
	X_test = poly.fit_transform(scaler.transform(X_test))


# Train, Predict, and Store Results with Neural Network Model!
print "Selecting Hyperparameters..."
best_nn = None
best_accuracy = -1.0
best_y_test_pred = None
num_iters = 0

for i in range(num_iters):
	# Train Model
	num_hidden_layers = 4  # Originally (i % 4) + 1
	# hidden_layer_sizes = utils.rand_hidden_layer_sizes(X_train.shape[1], 10, num_hidden_layers)
	hidden_layer_sizes = [1679, 1270, 71]  # Best for X_HOG [1638, 838, 209, 59]
	# alpha = random.choice([1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3])  # Originally alpha = 1e-05
	alpha = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 3e-4, 3e-3, 3e-2, 3e-1, 3e0, 3e1, 3e2, 3e3][i]  # 0.03 Best!
	nn = MLPClassifier(
			activation='relu', algorithm='adam', alpha=alpha, batch_size='auto', beta_1=0.9, beta_2=0.999,
			early_stopping=True, epsilon=1e-08, hidden_layer_sizes=hidden_layer_sizes, learning_rate='constant',
			learning_rate_init=1e-3, max_iter=40, momentum=0.9, nesterovs_momentum=True, power_t=0.5,
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


# Train Many Neural Networks to vote on outcome
print "Training Voting System..."
# hidden_layer_sizes_list = [[1582], [2413], [1646], [1578, 45], [911, 46], [1664, 53], [2528, 110], [255]]
# hidden_layer_sizes_list = [[103], [255], [599]]
hyperparams_list = [([1638, 838, 209, 59], 0.03), ([2371, 1324, 288, 92], 0.01), ([2371, 1324, 288, 92], 0.1),
					([1679, 1270, 71], 0.00001), ([1508, 1330, 79, 15], 0.00001)]
y_test_pred_list = []

for hyperparams in hyperparams_list:
	nn = MLPClassifier(
			activation='relu', algorithm='adam', alpha=hyperparams[1], batch_size='auto', beta_1=0.9, beta_2=0.999,
			early_stopping=True, epsilon=1e-08, hidden_layer_sizes=hyperparams[0], learning_rate='constant',
			learning_rate_init=1e-3, max_iter=40, momentum=0.9, nesterovs_momentum=True, power_t=0.5,
			random_state=1, shuffle=True, tol=0.0001, validation_fraction=0.1, verbose=True, warm_start=False)
	nn.fit(X_train, y_train)
	y_test_pred = nn.predict(X_test)
	y_test_pred_list.append(y_test_pred)

	if not FINAL_RUN:
		test_accuracy = np.mean(y_test == y_test_pred)
		print hyperparams, test_accuracy
		utils.print_accuracy_report(y_test, y_test_pred)

	else:
		print hyperparams

y_test_pred_matrix = np.array(y_test_pred_list).T
best_y_test_pred = stats.mode(y_test_pred_matrix, axis=1)[0][:, 0]
best_accuracy = np.mean(y_test == best_y_test_pred)
print hyperparams_list, best_accuracy


if FINAL_RUN:
	utils.y_to_csv(best_y_test_pred, "data/testLabels.csv")


# Evaluate the best classifier on test set (if results are known)
if y_test is not None:
	print "Neural Network Test Set Accuracy: ", best_accuracy
	utils.print_accuracy_report(y_test, best_y_test_pred)


