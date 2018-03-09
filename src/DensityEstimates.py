#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 201801 LIU Wangsheng; Email: awang.signup@gmail.com
import numpy as np
import openturns as ot
from sklearn import mixture
from sklearn.cluster import KMeans
from src.gaussian_mixture_weighted import GaussianMixtureW

class DensityEstimates:
    def __init__(self, hparams, random_seed):
        self.random_seed = random_seed
        self.hparams = hparams
        self.name = self.hparams['name']
        if self.name == 'GM-sklearn':
            n_components = self.hparams['n_components']
            max_iter = self.hparams['max_iter']
            n_init = self.hparams['n_init']
            self.model = mixture.GaussianMixture(covariance_type='full', n_components=n_components,
                                                 n_init=n_init, max_iter=max_iter, random_state=self.random_seed)
        elif self.name == 'GM-w':
            n_components = self.hparams['n_components']
            max_iter = self.hparams['max_iter']
            n_init = self.hparams['n_init']
            self.model = GaussianMixtureW(covariance_type='full', n_components=n_components,
                                           n_init=n_init, max_iter=max_iter, random_state=self.random_seed)

    def fit(self, X, **kwargs):
        if self.name == 'GM-sklearn':
            self.model.fit(X)
            return self
        elif self.name == 'GM-w':
            self.model.fit(X, kwargs['weights'])
            return self
        elif self.name == 'GM-mpp':
            n_components = self.hparams['n_components']
            d = X.shape[1]
            dist_u = ot.Normal(d)
            if n_components > 1:
                labels = KMeans(n_clusters=n_components, random_state=self.random_seed).fit_predict(X)
                dist_list = []
                for label in range(n_components):
                    tmp_idx = np.argmax(np.array(dist_u.computeLogPDF(X[labels == label])))
                    tmp_mean = X[labels == label][tmp_idx]
                    dist_list.append(ot.Normal(tmp_mean, [1.0] * d, ot.CorrelationMatrix(d)))
                self.dist_pseudo_best = ot.Mixture(dist_list, [1 / n_components] * n_components)
                return self
            else:
                tmp_idx = np.argmax(np.array(dist_u.computeLogPDF(X)))
                self.dist_pseudo_best = ot.Normal(X[tmp_idx], [1.0] * d, ot.CorrelationMatrix(d))
                return self

    def sample(self, num_samples):
        if self.name == 'GM-sklearn':
            X, _ = self.model.sample(num_samples)
            return X
        elif self.name == 'GM-mpp':
            X = np.array(self.dist_pseudo_best.getSample(num_samples))
            return X

    def score_samples(self, X):
        if self.name == 'GM-mpp':
            return np.array(self.dist_pseudo_best.computeLogPDF(X)).flatten()
        else:
            return self.model.score_samples(X)
