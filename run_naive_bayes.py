import numpy as np
import utils
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB



# Initialize variables
FINAL_RUN = False
NAIVE_BAYES_TYPE = "B"

if NAIVE_BAYES_TYPE == "B":
	nb = BernoulliNB()
	bins = 2
elif NAIVE_BAYES_TYPE == "G":
	nb = GaussianNB()
	bins = False
elif NAIVE_BAYES_TYPE == "M":
	nb = MultinomialNB()
	bins = 256
else:
	print "No such NAIVE_BAYES_TYPE"
	assert(False)

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
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_train, y_train, test_size=0.2)


# Train, Predict, and Store Results with Naive Bayes Model!
y_test_pred = nb.fit(X_train, y_train).predict(X_test)
utils.y_to_csv(y_test_pred, value_to_class_mapping, "data/testLabels.csv")


# Evaluate the best softmax classifier on test set (if results are known)
if y_test is not None:
	test_accuracy = np.mean(y_test == y_test_pred)
	print NAIVE_BAYES_TYPE, "Naive Bayes Test Set Accuracy: ", test_accuracy

	c_matrix = confusion_matrix(y_test, y_test_pred)
	print c_matrix

	num_songs_actual = np.sum(c_matrix, axis=1)
	accuracy_per_class = np.zeros(10)
	for i in range(10):
		accuracy_per_class[i] = c_matrix[i, i] / (1.0 * num_songs_actual[i])
	print accuracy_per_class.round(3) * 100
