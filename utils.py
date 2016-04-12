import csv
import numpy as np
import cPickle as pickle
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import PolynomialFeatures
from PIL import Image
import math
import random
from skimage.feature import hog
from skimage import color
import sys
import time


class_to_value_mapping = {"airplane": 0, "automobile": 1, "bird": 2, "cat": 3, "deer": 4,
						"dog": 5, "frog": 6, "horse": 7, "ship": 8, "truck": 9}
value_to_class_mapping = {0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
						5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"}

def pngs_to_matrix(filepath, num_pngs, hog_repr=False, include_flipped=False, all_reps=False):
	"""
	Converts a PNG file to a data matrix, without any processing, but perhaps data augmentation if specified
	given the filepath to the folder containing the images.
	-- hog_repr = True: Adds the HOG representation of the data to the X matrix (takes a while)
	-- include_flipped = True: Adds the flipped versions of images to the X matrix
	-- all_reps = True: Returns all possible combinations of representations (only useful when trying to pickle data)
	"""
	print("Reading Raw Data...")

	# Initialize variables
	pic_dim = 32
	X = np.zeros([num_pngs, pic_dim, pic_dim, 3])
	HOG = np.zeros([num_pngs, pic_dim, pic_dim])  # May need to change later depending on HOG params

	# Add a slash to end of filepath if necessary
	if filepath[-1] != '/':
		filepath += '/'

	# Store pixel data for each image in X matrix
	for pic_num in range(1, num_pngs + 1):
		if pic_num % 5000 == 0:
			print(pic_num)
		# RGB Data
		image = Image.open(filepath + str(pic_num) + ".png")
		rgb_values = np.array(list(image.getdata())).reshape([pic_dim, pic_dim, 3])
		X[pic_num - 1] = rgb_values

		# HOG Data
		if hog_repr or all_reps:
			grayscale_values = color.rgb2gray(1.0 * rgb_values)
			fd, hog_values = hog(grayscale_values, orientations=8, pixels_per_cell=(2, 2),  # TODO: Correct Param Values?
				cells_per_block=(1, 1), visualise=True)
			HOG[pic_num - 1] = hog_values

	X = X.reshape(num_pngs, X.size / num_pngs)

	# If all representations need to be returned (useful only really for pickling)
	if all_reps:
		X_with_flipped = np.concatenate((X, np.fliplr(X)))
		HOG = HOG.reshape(num_pngs, HOG.size / num_pngs)
		HOG_with_flipped = np.concatenate((HOG, np.fliplr(HOG)))
		# Returns X, X with flipped data, X with HOG repr, and X with flipped data and in HOG repr
		return X, X_with_flipped, np.concatenate((X, HOG), axis=1),\
				np.concatenate((X_with_flipped, HOG_with_flipped), axis=1)

	# Return data with right form of augmentation
	if hog_repr:
		HOG = HOG.reshape(num_pngs, HOG.size / num_pngs)
		if include_flipped:
			X_with_flipped = np.concatenate((X, np.fliplr(X)))
			HOG_with_flipped = np.concatenate((HOG, np.fliplr(HOG)))
			return np.concatenate((X_with_flipped, HOG_with_flipped), axis=1)
		else:
			return np.concatenate((X, HOG), axis=1)
	else:
		if include_flipped:
			return np.concatenate((X, np.fliplr(X)))
		else:
			return X


def preprocess(X_raw, bins=False):
	"""
	Returns a version of a raw data matrix with the following modifications:
	1. Scales features/Bins features (into specified number of bins, no bins if bins=False)
	2. Adds intercept term
	3. Combines features as desired (if at all)
	"""
	print("Preprocessing Data...")
	poly = PolynomialFeatures(degree=1, include_bias=True)
	if not bins:
		X_scaled = X_raw / 255.0  # Feature scales to [0, 1] -> Later try [-1, 1]?
		return poly.fit_transform(X_scaled)
	else:
		X_scaled = X_raw.astype(int) / (256 / bins)
		return poly.fit_transform(X_scaled).astype(int)


def get_X(pngs_filepath, num_data, hog_repr=False, include_flipped=False, all_reps=False, bins=False):
	"""
	Converts a PNG file to a data matrix, processing data appropriately,
	given the filepath to the folder containing the images
	"""
	if all_reps:
		return pngs_to_matrix(pngs_filepath, num_data, all_reps=all_reps)
	if hog_repr:
		return pngs_to_matrix(pngs_filepath, num_data, hog_repr=hog_repr)
	else:
		return preprocess(pngs_to_matrix(pngs_filepath, num_data), bins)


def get_y(filepath):
	"""
	Reads in a CSV of training data "answers" as a y vector,
	given the filepath to the CSV data
	"""
	lines = []
	with open(filepath) as csvfile:
		reader = csv.reader(csvfile)
		next(reader)  # Skip header
		for row in reader:
			lines.append(class_to_value_mapping[row[1]])
	return np.array(lines)


def y_to_csv(y, filepath):
	"""
	Writes a vector of classification results (y) in CSV format into a file (location specified by filepath),
	using the value_to_class_mapping conversion mapping.
	"""
	file = open(filepath, "w")
	file.write("id,label\n")
	for i in range(len(y)):
		file.write(str(i + 1) + "," + value_to_class_mapping[y[i]] + "\n")
	return


def log_random(min, max):
	"""
	Generates a random integer in the given range on a uniform log scale.
	I.e. the number has the same change of being in the range (1,10) as in the range (10,100).
	"""
	return int(round(math.e ** random.uniform(math.log(min), math.log(max))))


def rand_hidden_layer_sizes(input_layer_size, output_layer_size, num_hidden_layers, method=0):
	"""
	Randomly generates the number of neurons per hidden layer.
	Successive layers get smaller in this implementation.
	"""
	if method == 0:
		layer_sizes = []
		for layer_num in range(num_hidden_layers):
			layer_sizes.append(log_random(output_layer_size, input_layer_size))
		layer_sizes.sort(reverse=True)
		return layer_sizes
	else:
		layer_sizes = [input_layer_size]
		for layer_num in range(num_hidden_layers):
			layer_sizes.append(log_random(layer_sizes[layer_num], output_layer_size))
		return layer_sizes[1:]


def print_accuracy_report(y_test, y_test_pred):
	"""
	Prints a detailed accuracy report on predicted output when compared to actual answers
	"""
	c_matrix = confusion_matrix(y_test, y_test_pred)
	num_data_actual = np.sum(c_matrix, axis=1)
	accuracy_per_class = np.zeros(10)
	for i in range(10):
		accuracy_per_class[i] = c_matrix[i, i] / (1.0 * num_data_actual[i])
	print(c_matrix)
	print(accuracy_per_class.round(3) * 100)
	return


def dump(data, filename):
	"""
	Dump data into pickled form into a file of name filename.
	Filename should end in ".pickle"
	"""
	sys.setrecursionlimit(10000)
	with open(filename, 'wb') as f:
		pickle.dump(data, f, -1)
	return


def load(filename):
	"""
	Load data from pickled form from a file of name filename.
	Filename should end in ".pickle"
	"""
	with open(filename, 'rb') as f:
		return pickle.load(f)
	return


def convert_y_to_matrix(y, num_classes=10):
	"""
	Useful data conversion function for classification/ova schemes
	"""
	y_matrix = np.zeros((len(y), num_classes))
	for i in range(len(y)):
		y_matrix[i, y[i]] = 1.0
	return y_matrix


def load2d(filepath, num_pngs):
	"""
	Loads in 2d image data into a num_examples x num_channels x num_dim x num_dim matrix
	"""
	# TODO!
	print("Reading Raw Data in 2D Form...")

	# Initialize variables
	num_channels = 3
	pic_dim = 32
	X = np.zeros([num_pngs, num_channels, pic_dim, pic_dim])

	# Add a slash to end of filepath if necessary
	if filepath[-1] != '/':
		filepath += '/'

	# Store pixel data for each image in X matrix
	for pic_num in range(1, num_pngs + 1):
		if pic_num % 5000 == 0:
			print(pic_num)
		# RGB Data
		image = Image.open(filepath + str(pic_num) + ".png")
		X[pic_num - 1] = np.array(list(image.getdata())).T.reshape(3, pic_dim, pic_dim)

	return X


def float32(k):
	return np.cast['float32'](k)



# num_images = 50000
# print num_images
# all_representations = get_X("data/train", num_images, all_reps=True)
# dump(all_representations[0], "X.pickle")
# dump(all_representations[1], "X_flipped.pickle")
# dump(all_representations[2], "X_HOG.pickle")
# dump(all_representations[3], "X_HOG_flipped.pickle")

# all_representations = get_X("data/test", 300000, all_reps=True)
# dump(all_representations[0], "XTEST.pickle")
# dump(all_representations[2], "XTEST_HOG.pickle")

# print load2d("data/train", 2)
