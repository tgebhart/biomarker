import os
import pandas as pd
from data_collection import *
import numpy as np
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA


def ensemble(num_test, seed=1, regressor='tree', meta_model='linear', var_classification = True, meta_classification = True,max_depth=3):
    np.random.seed(seed)

    excel = parse_master_file()
    L = get_filename_list(excel['Associated data'])

    x1 = create_x1_matrix(L)
    x4 = create_x4_matrix(L)
    x5 = create_x5_matrix(L)
    x6 = create_x6_matrix(L)
    x7 = create_x7_matrix(L)

    if var_classification:
        y = excel['Output: logKbucket'].values
    else:
        y = excel['Output: logK'].values
    x10_x17 = excel.iloc[:, 3:-2]

    master,_ = prepare_master(x10_x17)

    test_idxs = np.random.randint(0,len(y),num_test)
    train_idxs = np.ones(y.shape,dtype=bool)
    train_idxs[test_idxs] = False
    y_train = y[train_idxs]
    y_test = y[test_idxs]

    master_train = master[train_idxs]
    master_test = master[test_idxs]

    if regressor == 'linear':
        if var_classification:
            x1_approx_train,_ = linear_classification_approx(x1[train_idxs], y_train)
            x4_approx_train,_ = linear_classification_approx(x4[train_idxs], y_train)
            x5_approx_train,_ = linear_classification_approx(x5[train_idxs], y_train)
            x6_approx_train,_ = linear_classification_approx(x6[train_idxs], y_train)
            x7_approx_train,_ = linear_classification_approx(x7[train_idxs], y_train)

            x1_approx_test,_ = linear_classification_approx(x1[test_idxs], y_test)
            x4_approx_test,_ = linear_classification_approx(x4[test_idxs], y_test)
            x5_approx_test,_ = linear_classification_approx(x5[test_idxs], y_test)
            x6_approx_test,_ = linear_classification_approx(x6[test_idxs], y_test)
            x7_approx_test,_ = linear_classification_approx(x7[test_idxs], y_test)
        else:
            x1_approx_train,_ = linear_regression_approx(x1[train_idxs], y_train)
            x4_approx_train,_ = linear_regression_approx(x4[train_idxs], y_train)
            x5_approx_train,_ = linear_regression_approx(x5[train_idxs], y_train)
            x6_approx_train,_ = linear_regression_approx(x6[train_idxs], y_train)
            x7_approx_train,_ = linear_regression_approx(x7[train_idxs], y_train)

            x1_approx_test,_ = linear_regression_approx(x1[test_idxs], y_test)
            x4_approx_test,_ = linear_regression_approx(x4[test_idxs], y_test)
            x5_approx_test,_ = linear_regression_approx(x5[test_idxs], y_test)
            x6_approx_test,_ = linear_regression_approx(x6[test_idxs], y_test)
            x7_approx_test,_ = linear_regression_approx(x7[test_idxs], y_test)
        
        

    elif regressor == 'tree':
        if var_classification:
            x1_approx_train,_ = decision_tree_approx(x1[train_idxs], y_train)
            x4_approx_train,_ = decision_tree_approx(x4[train_idxs], y_train)
            x5_approx_train,_ = decision_tree_approx(x5[train_idxs], y_train)
            x6_approx_train,_ = decision_tree_approx(x6[train_idxs], y_train)
            x7_approx_train,_ = decision_tree_approx(x7[train_idxs], y_train)

            x1_approx_test,_ = decision_tree_approx(x1[test_idxs], y_test)
            x4_approx_test,_ = decision_tree_approx(x4[test_idxs], y_test)
            x5_approx_test,_ = decision_tree_approx(x5[test_idxs], y_test)
            x6_approx_test,_ = decision_tree_approx(x6[test_idxs], y_test)
            x7_approx_test,_ = decision_tree_approx(x7[test_idxs], y_test)
        else:
            x1_approx_train,_ = regression_tree_approx(x1[train_idxs], y_train)
            x4_approx_train,_ = regression_tree_approx(x4[train_idxs], y_train)
            x5_approx_train,_ = regression_tree_approx(x5[train_idxs], y_train)
            x6_approx_train,_ = regression_tree_approx(x6[train_idxs], y_train)
            x7_approx_train,_ = regression_tree_approx(x7[train_idxs], y_train)

            x1_approx_test,_ = regression_tree_approx(x1[test_idxs], y_test)
            x4_approx_test,_ = regression_tree_approx(x4[test_idxs], y_test)
            x5_approx_test,_ = regression_tree_approx(x5[test_idxs], y_test)
            x6_approx_test,_ = regression_tree_approx(x6[test_idxs], y_test)
            x7_approx_test,_ = regression_tree_approx(x7[test_idxs], y_test)
    else:
        raise ValueError('please choose appropriate regressor type')

    if meta_classification:
        y = excel['Output: logKbucket'].values
        y_train = y[train_idxs]
        y_test = y[test_idxs]
    if meta_model == 'linear':
        if meta_classification:
            regr = SVC(kernel='linear')
        else:
            regr = linear_model.LinearRegression()
    elif meta_model == 'tree':
        if meta_classification:    
            regr = DecisionTreeClassifier(max_depth=max_depth)
        else:
            regr = DecisionTreeRegressor(max_depth=max_depth)
    else:
        raise ValueError('please choose appropriate meta model')
        
    all_xs_train = np.column_stack((x1_approx_train, x4_approx_train, x5_approx_train, x6_approx_train, x7_approx_train, master_train))
    regr.fit(all_xs_train, y_train)

    all_xs_test = np.column_stack((x1_approx_test, x4_approx_test, x5_approx_test, x6_approx_test, x7_approx_test, master_test))
    predictions = regr.predict(all_xs_test)
    print(y_test)
    print(predictions)
    mse = mean_squared_error(y_test, predictions)
    r_squared = r2_score(y_test, predictions)
    # The mean squared error
    print("Mean squared error: %.2f" % mse)
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r_squared)

    if meta_model == 'linear':
        print('Coefficients: \n', regr.coef_)
        return mse, r_squared, regr.coef_, regr
    else:
        return mse, r_squared, None, regr








# end
