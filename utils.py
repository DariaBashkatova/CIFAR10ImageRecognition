import csv
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from PIL import Image


def pngs_to_matrix(filepath, num_pngs):
	"""
	Converts a PNG file to a data matrix, without any processing,
	given the filepath to the folder containing the images
	"""
	print "Reading Raw Data..."

	# Initialize variables
	pic_dim = 32
	num_features = 3 * (pic_dim ** 2)
	X = np.zeros([num_pngs, num_features])

	# Add a slash to end of filepath if necessary
	if filepath[-1] != '/':
		filepath += '/'

	# Store pixel data for each image in X matrix
	for pic_num in range(1, num_pngs + 1):
		image = Image.open(filepath + str(pic_num) + ".png")
		rgb_values = list(image.getdata())
		cur_feature_num = 0
		for (r_val, g_val, b_val) in rgb_values:
			X[pic_num - 1, cur_feature_num] = r_val
			X[pic_num - 1, cur_feature_num + 1] = g_val
			X[pic_num - 1, cur_feature_num + 2] = b_val
			cur_feature_num += 3

	return X


def preprocess(X_raw, bins=False):
	"""
	Returns a version of a raw data matrix with the following modifications:
	1. Scales features/Bins features (into specified number of bins, no bins if bins=False)
	2. Adds intercept term
	3. Combines features as desired (if at all)
	"""
	print "Preprocessing Data..."
	poly = PolynomialFeatures(degree=1, include_bias=True)
	if not bins:
		X_scaled = X_raw / 255.0  # Feature scales to [0, 1] -> Later try [-1, 1]?
		return poly.fit_transform(X_scaled)
	else:
		X_scaled = X_raw.astype(int) / (256 / bins)
		return poly.fit_transform(X_scaled).astype(int)


def get_X(pngs_filepath, num_data, bins=False):
	"""
	Converts a PNG file to a data matrix, processing data appropriately,
	given the filepath to the folder containing the images
	"""
	return preprocess(pngs_to_matrix(pngs_filepath, num_data), bins)


def get_y(filepath, class_to_value_mapping):
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


def y_to_csv(y, value_to_class_mapping, filepath):
	"""
	Writes a vector of classification results (y) in CSV format into a file (location specified by filepath),
	using the value_to_class_mapping conversion mapping.
	"""
	file = open(filepath, "w")
	file.write("id,label\n")
	for i in range(len(y)):
		file.write(str(i + 1) + "," + value_to_class_mapping[y[i]] + "\n")
	return


