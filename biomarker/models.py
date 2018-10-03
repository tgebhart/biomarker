import os

import pandas as pd
from biomarker.data_collection import *
import numpy as np

from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA


class Ensemble(object):

    def __init__(self, seed=1, regressor='tree', meta_model='linear', max_depth=2):

        np.random.seed(seed)
        self.regressor = regressor

        if meta_model == 'linear':
            self.regr = linear_model.LinearRegression()
        elif meta_model == 'tree':
            self.regr = DecisionTreeRegressor(max_depth=max_depth)
        else:
            raise ValueError('please input acceptable meta_model')

        self.master = None

        self.x1_approx = None
        self.x4_approx = None
        self.x5_approx = None
        self.x6_approx = None
        self.x7_approx = None

        self.x1_regr = None
        self.x4_regr = None
        self.x5_regr = None
        self.x6_regr = None
        self.x7_regr = None


    def fit(self, x1, x4, x5, x6, x7, master, y, max_depth=2):
        self.x1 = x1
        self.x4 = x4
        self.x5 = x5
        self.x6 = x6
        self.x7 = x7

        self.master = master

        if self.regressor == 'linear':

            self.x1_approx,self.x1_regr = linear_regression_approx(x1, y)
            self.x4_approx,self.x4_regr = linear_regression_approx(x4, y)
            self.x5_approx,self.x5_regr = linear_regression_approx(x5, y)
            self.x6_approx,self.x6_regr = linear_regression_approx(x6, y)
            self.x7_approx,self.x7_regr = linear_regression_approx(x7, y)

        elif self.regressor == 'tree':

            self.x1_approx,self.x1_regr = regression_tree_approx(x1, y, max_depth=max_depth)
            self.x4_approx,self.x4_regr = regression_tree_approx(x4, y, max_depth=max_depth)
            self.x5_approx,self.x5_regr = regression_tree_approx(x5, y, max_depth=max_depth)
            self.x6_approx,self.x6_regr = regression_tree_approx(x6, y, max_depth=max_depth)
            self.x7_approx,self.x7_regr = regression_tree_approx(x7, y, max_depth=max_depth)

        else:
            raise ValueError('Please set regressor to appropriate type')

        xs = np.column_stack((self.x1_approx, self.x4_approx, self.x5_approx, self.x6_approx, self.x7_approx, self.master))
        self.regr.fit(xs, y)

    def predict(self, x1, x4, x5, x6, x7, master):

        x1_approx = self.x1_regr.predict(x1)
        x4_approx = self.x4_regr.predict(x4)
        x5_approx = self.x5_regr.predict(x5)
        x6_approx = self.x6_regr.predict(x6)
        x7_approx = self.x7_regr.predict(x7)

        xs = np.column_stack((x1_approx, x4_approx, x5_approx, x6_approx, x7_approx, master))
        return self.regr.predict(xs)














# end
