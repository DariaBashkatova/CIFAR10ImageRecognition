import numpy as np
import utils
from sklearn import model_selection
from sklearn import svm
from scipy import stats



# Initialize variables
FINAL_RUN = False

X_train = utils.load("X.pickle")
y_train = utils.get_y("data/trainLabels.csv")

# Create training, validation, and test data sets
print "Creating Train and Test Sets..."
if FINAL_RUN:  # When running on Training Data and untouched Test Data!
	X_test = utils.get_X("data/test", 300000)
	y_test = None
else:  # When running ONLY on Training Data!
	X_train, X_test, y_train, y_test = model_selection.train_test_split(X_train, y_train, test_size=0.2)


print "Training SVM..."
svm_model = svm.SVC(verbose=True)
best_y_test_pred = svm_model.fit(X_train, y_train).predict(X_test)


# Store final results if necessary
if FINAL_RUN:
	utils.y_to_csv(best_y_test_pred, "data/CNN-NN-SVMtestLabels.csv")

# Evaluate the best softmax classifier on test set (if results are known)
if y_test is not None:
	test_accuracy = np.mean(y_test == best_y_test_pred)
	print "Test Set Accuracy: ", test_accuracy
	utils.print_accuracy_report(y_test, best_y_test_pred, "SVMConfusionMatrix")
