import numpy as np
from sklearn import model_selection
from sklearn.metrics import confusion_matrix, classification_report
from ova_log import ovaLogisticRegressor
import utils


# Initialize variables
FINAL_RUN = True
X = utils.get_X("data/train", 50000)
y = utils.get_y("data/trainLabels.csv")


# Create training, validation, and test data sets
print "Creating Train, Val, and Test Sets..."
if FINAL_RUN:  # When running on Training Data and untouched Test Data!
	X_train_val = X
	y_train_val = y
	X_test = utils.get_X("data/test", 300000)
	y_test = None
else:  # When running ONLY on Training Data!
	X_train_val, X_test, y_train_val, y_test = model_selection.train_test_split(X, y, test_size=0.2)

X_train, X_val, y_train, y_val = model_selection.train_test_split(X_train_val, y_train_val, test_size=0.2)

# Select Hyperparameters
print "Selecting Hyperparameters..."
penalties = ["l2"]  # "l1" is VERY slow
regularization_strengths = [75.0, 100.0, 125.0, 150.0, 175.0, 200.0]  # 100.0 is best
best_ova_log_reg = None
train_accuracy_of_best_ova_log_reg = -1
best_val_accuracy = -1

for penalty in penalties:
	for reg in regularization_strengths:
		ova_log_reg = ovaLogisticRegressor(np.arange(10))
		ova_log_reg.train(X_train, y_train, reg, penalty)

		y_train_pred = ova_log_reg.predict(X_train)
		train_accuracy = np.mean(y_train_pred == y_train)
		y_val_pred = ova_log_reg.predict(X_val)
		val_accuracy = np.mean(y_val_pred == y_val)
		print "Penalty: ", penalty, " Reg: ", reg, " Train Accuracy: ", train_accuracy, " Val Accuracy: ", val_accuracy

		if val_accuracy > best_val_accuracy:
			best_val_accuracy = val_accuracy
			train_accuracy_of_best_ova_log_reg = train_accuracy
			best_ova_log_reg = ova_log_reg

print "\nBEST OVA LOG REG HYPERPARAMETERS:"
print "Penalty: ", best_ova_log_reg.penalty, " Reg: ", best_ova_log_reg.reg,\
	" Train Accuracy: ", train_accuracy_of_best_ova_log_reg, " Val Accuracy: ", best_val_accuracy


# With ascertained optimal hyperparameters, train new model on Training and Validation Sets
print "Training Final Model..."
best_ova_log_reg_final = ovaLogisticRegressor(np.arange(10))
best_ova_log_reg_final.train(X_train_val, y_train_val, best_ova_log_reg.reg, best_ova_log_reg.penalty)

# Make and store predictions!
print "Making Final Predictions..."
y_test_pred = best_ova_log_reg_final.predict(X_test)
utils.y_to_csv(y_test_pred, "data/testLabels.csv")


# Evaluate the best softmax classifier on test set (if results are known)
if y_test is not None:
	test_accuracy = np.mean(y_test == y_test_pred)
	print "OVA Logistic Regression Test Set Accuracy: ", test_accuracy
	utils.print_accuracy_report(y_test, y_test_pred)
