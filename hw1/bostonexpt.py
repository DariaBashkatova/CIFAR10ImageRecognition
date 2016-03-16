from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_boston
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import part1.plot_utils, part1.utils, part2.plot_utils, part2.utils
from part2.reg_linear_regressor_multi import RegularizedLinearReg_SquaredLoss

print 'Reading data ...'
bdata = load_boston()
df = pd.DataFrame(data = bdata.data, columns = bdata.feature_names)

# Get Data Matrices
X = df.values
y = bdata.target

# Split Data into Training, Validation, and Testing
X_non_test, X_test, y_non_test, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_non_test, y_non_test, test_size=0.2)

for degree in range(1, 4):
    print "DEGREE = ", degree
    # Add Polynomial Features without Bias
    poly = PolynomialFeatures(degree, include_bias=False)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_val = poly.fit_transform(X_val)
    X_poly_test = poly.fit_transform(X_test)

    # Normalize Features
    X_poly_train_norm = part1.utils.feature_normalize(X_poly_train)[0]
    X_poly_val_norm = part1.utils.feature_normalize(X_poly_val)[0]
    X_poly_test_norm = part1.utils.feature_normalize(X_poly_test)[0]

    # Add Bias Column
    poly = PolynomialFeatures(1, include_bias=True)
    XX_poly_train = poly.fit_transform(X_poly_train_norm)
    XX_poly_val = poly.fit_transform(X_poly_val_norm)
    XX_poly_test = poly.fit_transform(X_poly_test_norm)

    # Generate Unregularized Linear Regression Model
    if degree == 1:
        linear_reg = RegularizedLinearReg_SquaredLoss()
        theta_opt = linear_reg.train(XX_poly_train,y_train,reg=0.0,num_iters=1000)
        print 'Theta at lambda = 0 is ', theta_opt

        test_error = linear_reg.loss(theta_opt, XX_poly_train, y_train, 0.0)
        print 'Lowest achievable error on the test set with lambda = 0 is ', test_error

    # Generate Regularized Linear Regression Model
    reg_vec, error_train, error_val = part2.utils.validation_curve(XX_poly_train, y_train, XX_poly_val, y_val)
    part2.plot_utils.plot_lambda_selection(reg_vec,error_train,error_val)
    plt.savefig('LambdasGraphP=' + str(degree) + '.pdf')
    print 'Test Error for the best lambda for Degree ' + str(degree) + ' is ', np.min(error_val)


