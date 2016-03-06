import numpy as np
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix, classification_report
from ova_log import ovaLogisticRegressor
import time
import utils


# Initialize variables
FINAL_RUN = True
class_to_value_mapping = {"airplane": 0, "automobile": 1, "bird": 2, "cat": 3, "deer": 4,
						"dog": 5, "frog": 6, "horse": 7, "ship": 8, "truck": 9}
value_to_class_mapping = {0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
						5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"}
X = utils.get_X("data/train", 50000)
y = utils.get_y("data/trainLabels.csv", class_to_value_mapping)


# Create training, validation, and test data sets
print "Creating Train, Val, and Test Sets..."
if FINAL_RUN:  # When running on Training Data and untouched Test Data!
	X_train_val = X
	y_train_val = y
	X_test = utils.get_X("data/test", 300000)
	y_test = None
else:  # When running ONLY on Training Data!
	X_train_val, X_test, y_train_val, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

X_train, X_val, y_train, y_val = cross_validation.train_test_split(X_train_val, y_train_val, test_size=0.2)

# # Select Hyperparameters
# print "Selecting Hyperparameters..."
# penalties = ["l2"]
# regularization_strengths = [25.0, 50.0, 100.0, 200.0, 400.0]
# best_ova_log_reg = None
# train_accuracy_of_best_ova_log_reg = -1
# best_val_accuracy = -1
# # penalty = "l2" and reg = 100.0 works well
# for penalty in penalties:
# 	for reg in regularization_strengths:
# 		ova_log_reg = ovaLogisticRegressor(np.arange(10))
# 		ova_log_reg.train(X_train, y_train, reg, penalty)
#
# 		y_train_pred = ova_log_reg.predict(X_train)
# 		train_accuracy = np.mean(y_train_pred == y_train)
# 		y_val_pred = ova_log_reg.predict(X_val)
# 		val_accuracy = np.mean(y_val_pred == y_val)
# 		print "Penalty: ", penalty, " Reg: ", reg, " Train Accuracy: ", train_accuracy, " Val Accuracy: ", val_accuracy
#
# 		if val_accuracy > best_val_accuracy:
# 			best_val_accuracy = val_accuracy
# 			train_accuracy_of_best_ova_log_reg = train_accuracy
# 			best_ova_log_reg = ova_log_reg
#
# print "\nBEST OVA LOG REG HYPERPARAMETERS:"
# print "Penalty: ", best_ova_log_reg.penalty, " Reg: ", best_ova_log_reg.reg,\
# 	" Train Accuracy: ", train_accuracy_of_best_ova_log_reg, " Val Accuracy: ", best_val_accuracy


# With ascertained optimal hyperparameters, train new model on Training and Validation Sets
print "Training Final Model..."
tic = time.time()
best_ova_log_reg_final = ovaLogisticRegressor(np.arange(10))
# best_ova_log_reg_final.train(X_train_val, y_train_val, best_ova_log_reg.reg, best_ova_log_reg.penalty)
best_ova_log_reg_final.train(X_train_val, y_train_val, 100.0, "l2")
toc = time.time()
print "Training Model Time: ", toc - tic

# Make and store predictions!
print "Making Final Predictions..."
y_test_pred = best_ova_log_reg_final.predict(X_test)
utils.y_to_csv(y_test_pred, value_to_class_mapping, "data/testLabels.csv")


# Evaluate the best softmax classifier on test set (if results are known)
if y_test is not None:
	test_accuracy = np.mean(y_test == y_test_pred)
	print "OVA Logistic Regression Test Set Accuracy: ", test_accuracy

	c_matrix = confusion_matrix(y_test, y_test_pred)
	print c_matrix

	num_songs_actual = np.sum(c_matrix, axis=1)
	accuracy_per_class = np.zeros(10)
	for i in range(10):
		accuracy_per_class[i] = c_matrix[i, i] / (1.0 * num_songs_actual[i])
	print accuracy_per_class.round(3) * 100
