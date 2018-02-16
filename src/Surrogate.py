#!/usr/bin/env python3
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor


class Surrogate:
    """ Surrogate models

    """

    def __init__(self, hparams):
        self.model_name = hparams['name']
        self.normalize = hparams['normalize']
        if self.model_name == 'GP':
            if 'normalize_y' not in hparams:
                hparams['normalize_y'] = False
            self.model = GaussianProcessRegressor(kernel=hparams['kernel'],
                                                  n_restarts_optimizer=hparams['n_restarts_optimizer'],
                                                  normalize_y=hparams['normalize_y'])
        if self.model_name == 'SVR':
            self.model = SVR()
        if self.model_name == 'RF':
            self.model = RandomForestRegressor(max_depth=2, random_state=0,
                                               n_estimators=hparams['n_estimators'])

    def fit(self, X, y):
        """
        Fit surrogate models
        Args:
            X: array-like, shape = (n, d)
               Training data
            y: array-like, shape = (n, [num_lsf])
               Soft output values

        Returns:
            self: returns an instance of self

        """
        if self.normalize:
            self.scaler_X = StandardScaler()
            # self.scaler_y = StandardScaler()
            self.scaler_X.fit(X)
            # self.scaler_y.fit(y)
            X = self.scaler_X.transform(X)
            # y = self.scaler_y.transform(y)
        if self.model_name == 'RF':
            y = y.flatten()
        self.model.fit(X, y)
        return self

    def predict(self, X, *args):
        if self.normalize:
            X = self.scaler_X.transform(X)
        if self.model_name == 'GP':
            return_std = args[0]
            return_cov = args[1]
            y = self.model.predict(X, return_std, return_cov)
            return y
        if self.model_name == 'SVR':
            y = self.model.predict(X).reshape(-1, 1)
            return y
        if self.model_name == 'RF':
            y = self.model.predict(X).reshape(-1, 1)
            return y

