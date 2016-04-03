import numpy as np
import utils
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
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


X_train = utils.get_X("data/train", 50000, bins=bins)
y_train = utils.get_y("data/trainLabels.csv")


# Create training, validation, and test data sets
print "Creating Train and Test Sets..."
if FINAL_RUN:  # When running on Training Data and untouched Test Data!
	X_test = utils.get_X("data/test", 300000)
	y_test = None
else:  # When running ONLY on Training Data!
	X_train, X_test, y_train, y_test = model_selection.train_test_split(X_train, y_train, test_size=0.2)


# Train, Predict, and Store Results with Naive Bayes Model!
y_test_pred = nb.fit(X_train, y_train).predict(X_test)
utils.y_to_csv(y_test_pred, "data/testLabels.csv")


# Evaluate the best softmax classifier on test set (if results are known)
if y_test is not None:
	test_accuracy = np.mean(y_test == y_test_pred)
	print NAIVE_BAYES_TYPE, "Naive Bayes Test Set Accuracy: ", test_accuracy
	utils.print_accuracy_report(y_test, y_test_pred)
